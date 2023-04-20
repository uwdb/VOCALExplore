from collections import defaultdict
from contextlib import contextmanager, ExitStack
import duckdb
import glob
import logging
import numpy as np
import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import threading
from typing import List

from vfe.api.storagemanager import ClipSet
from .abstract import AbstractFeatureStore

def validate_parquet_dir(dir):
    if not os.path.exists(dir):
        return

    for pq_file in glob.glob(os.path.join(dir, '*.parquet')):
        try:
            duckdb.connect().execute("""
                SELECT count(*) FROM '{pq_file}'
            """.format(pq_file=pq_file)).fetchnumpy()
        except:
            logging.warning(f'Removing {pq_file} because it could not be queried')
            os.remove(pq_file)

class ParquetFeatureStore(AbstractFeatureStore):
    def __init__(self, *args, base_dir=None, max_rows=50000, fid_offset=0, **kwargs):
        # Should probably check for malformed files on startup.
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
        self.max_rows = max_rows
        self.fid_offset = fid_offset
        self._feature_to_writer = {}
        self._feature_to_fid = {}
        self._feature_locks = defaultdict(threading.Lock)
        self._feature_to_writer_nrows = {}
        self._validated_features = set()
        self.logger = logging.getLogger(__name__)

    @classmethod
    def create(cls, *args, base_dir=None, dataset=None, **kwargs):
        return cls(*args, base_dir=os.path.join(base_dir, dataset, 'featurestore'), **kwargs)

    def __del__(self):
        print('Closing all feature store writers')
        for writer in self._feature_to_writer.values():
            writer.close()

    def _close_writer_if_necessary(self, feature_name):
        if feature_name in self._feature_to_writer:
            self._feature_to_writer[feature_name].close()
            self._feature_to_writer.pop(feature_name)
            self._feature_to_writer_nrows.pop(feature_name)

    def _get_parquet_file(self, feature_name):
        feature_dir = os.path.join(self.base_dir, feature_name)
        # Prefix version path with '_' so it will be ignored by pyarrow dataset.
        version_path = os.path.join(feature_dir, '_version.txt')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
            with open(version_path, 'w+') as f:
                print('0', file=f)

        # Not thread/process safe.
        with open(version_path, 'r') as f:
            version = int(f.readlines()[0].strip())

        with open(version_path, 'w') as f:
            print(f'{version + 1}', file=f)

        return os.path.join(feature_dir, f'{version}.parquet')

    def get_feature_names(self):
        if not os.path.exists(self.base_dir):
            return []
        return [d.name for d in os.scandir(self.base_dir) if d.is_dir() and d.name != 'tmp']

    @contextmanager
    def get_writer(self, feature_name):
        self._feature_locks[feature_name].acquire()
        if feature_name not in self._feature_to_writer:
            feature_file_path = self._get_parquet_file(feature_name)
            schema = pa.schema([
                ('fid', pa.uint32()),
                ('vid', pa.uint32()),
                ('start_time', pa.float64()),
                ('end_time', pa.float64()),
                ('feature', pa.list_(pa.float32())),
            ])
            self._feature_to_writer[feature_name] = pq.ParquetWriter(feature_file_path, schema=schema)
            self._feature_to_writer_nrows[feature_name] = 0

        try:
            yield self._feature_to_writer[feature_name]
        finally:
            self._feature_locks[feature_name].release()

    @contextmanager
    def get_reader(self, feature_name):
        # Eventually this could support multiple readers. But currently only the main thread reads.
        self._feature_locks[feature_name].acquire()
        self._close_writer_if_necessary(feature_name)
        try:
            yield
        finally:
            self._feature_locks[feature_name].release()

    def get_fid_sequence(self, feature_name, length):
        # Not thread/process safe.
        fid_path = os.path.join(self.base_dir, feature_name, '_fid.txt')
        if feature_name in self._feature_to_fid:
            fid = self._feature_to_fid[feature_name]
        elif os.path.exists(fid_path):
            with open(fid_path, 'r') as f:
                fid = int(f.readlines()[0].strip())
        else:
            fid = self.fid_offset
        stop = fid + length
        fid_sequence = np.arange(fid, stop, dtype=np.uint32)
        with open(fid_path, 'w') as f:
            print(f'{stop}', file=f)
        self._feature_to_fid[feature_name] = stop
        return fid_sequence

    def insert_batch(self, feature_name, vids, starts, ends, features):
        with self.get_writer(feature_name) as writer:
            fids = self.get_fid_sequence(feature_name, len(vids))
            batch = pa.record_batch([fids, vids, starts, ends, features], names=['fid', 'vid', 'start_time', 'end_time', 'feature'])
            writer.write_batch(batch)
            self._feature_to_writer_nrows[feature_name] += len(fids)
            if self._feature_to_writer_nrows[feature_name] > self.max_rows:
                self._close_writer_if_necessary(feature_name)

    def feature_dir(self, feature_name):
        # self.logger.debug(f'Reading features from {os.path.join(self.base_dir, feature_name)}')
        return os.path.join(self.base_dir, feature_name)

    def _validate_feature_name(self, feature_name):
        feature_dir = self.feature_dir(feature_name)
        validate_parquet_dir(feature_dir)

    def read_func(func):
        def wrapped_func(self, *args, feature_name=None, **kwargs):
            with self.get_reader(feature_name):
                if feature_name not in self._validated_features:
                    self._validate_feature_name(feature_name)
                    self._validated_features.add(feature_name)

                return func(self, *args, feature_name=feature_name, **kwargs)
        return wrapped_func

    def read_func_multifeature(func):
        def wrapped_func(self, *args, feature_names=None, **kwargs):
            with ExitStack() as stack:
                # Get the read lock on all features.
                for feature_name in feature_names:
                    stack.enter_context(self.get_reader(feature_name))

                # Call the function now that all read locks are held.
                return func(self, *args, feature_names=feature_names, **kwargs)
        return wrapped_func

    @read_func
    def get_vids(self, feature_name=None):
        feature_dir = self.feature_dir(feature_name)
        if not os.path.exists(feature_dir):
            return np.array([])

        return duckdb.connect().execute("""
            SELECT DISTINCT vid
            FROM '{feature_dir}/*.parquet'
        """.format(feature_dir=feature_dir)).fetchnumpy()['vid']

    def get_nonaggregated_dataset_concat(self, *, feature_names: List[str] = None, vids=None):
        # This is a read function, but we don't have to use @read_func because the calls to get_nonaggregated_dataset
        # will do that for us when we are actually reading the features.
        if len(feature_names) == 1:
            return self.get_nonaggregated_dataset(feature_name=feature_names[0], vids=vids)

        # assert len(feature_names) == 2, f'get_nonaggregated_dataset_concat only supports 2 features; got {feature_names}'

        combined_features = self.get_nonaggregated_dataset(feature_name=feature_names[0], vids=vids)
        for overlay_feat in feature_names[1:]:
            try:
                self.logger.debug(f'Adding {overlay_feat}; combined_len={len(combined_features)}')
            except:
                self.logger.debug(f'Adding {overlay_feat}; combined_len={len(combined_features.to_table())}')
            overlay_featureset = self.get_nonaggregated_dataset(feature_name=overlay_feat, vids=vids)
            combined_features = self._combine_featuresets(combined_features, overlay_featureset)

        self.logger.debug(f'Final length of combined={len(combined_features)}')
        return combined_features

    def _combine_featuresets(self, base_featureset, overlay_featureset):
        con = duckdb.connect() # in memory.
        return con.execute("""
            SELECT fid, vid, bstart AS start_time, bend AS end_time, LIST_CAT(base_feature, overlay_feature) AS feature
            FROM (
                SELECT *, ROW_NUMBER() OVER(
                    PARTITION BY fid
                    ORDER BY position ASC
                ) AS row_number
                FROM (
                    SELECT b.fid, b.vid, b.start_time as bstart, b.end_time as bend,
                        o.start_time as ostart, o.end_time as oend,
                        b.feature AS base_feature, o.feature AS overlay_feature,
                        ABS(0.5 - (o.start_time - b.start_time) / (b.end_time - b.start_time)) AS position
                    FROM base_featureset b, overlay_featureset o
                    WHERE b.vid=o.vid
                        AND b.start_time <= o.start_time
                        AND o.end_time <= b.end_time
                )
           )
           WHERE row_number=1
        """).arrow()

    @read_func
    def get_nonaggregated_dataset(self, *, feature_name=None, vids=None):
        feature_dir = self.feature_dir(feature_name)
        if not os.path.exists(feature_dir):
            return None
        dataset = ds.dataset(os.path.join(self.base_dir, feature_name), format='parquet')
        if vids is not None:
            return dataset.scanner(filter=ds.field('vid').isin(vids))
        return dataset

    def get_features_for_clips_concat(self, *, feature_names: List[str] = None, clips: ClipSet=None):
        # This is a read function, but we don't have to use @read_func because the calls to get_features_for_clips
        # will do that for us when we are actually reading the features.
        if len(feature_names) == 1:
            return self.get_features_for_clips(feature_name=feature_names[0], clips=clips)

        combined_features = self.get_features_for_clips(feature_name=feature_names[0], clips=clips)
        for overlay_feat in feature_names[1:]:
            try:
                self.logger.debug(f"Adding {overlay_feat}: len(combined_features)={len(combined_features)}")
            except:
                self.logger.debug(f"Adding {overlay_feat}: len(combined_features)={len(combined_features.to_table())}")
            overlay_features = self.get_nonaggregated_dataset(feature_name=overlay_feat, vids=clips.to_table()['vid'].unique())
            combined_features = self._combine_featuresets(combined_features, overlay_features)

        self.logger.debug(f"Final: len(combined_features)={len(combined_features)}")
        return combined_features

    @read_func
    def get_features_for_clips(self, *, feature_name=None, clips: ClipSet=None):
        feature_dir = self.feature_dir(feature_name)
        if not os.path.exists(feature_dir):
            return None
        con = duckdb.connect() # in memory.
        result = con.execute("""
            SELECT d.fid, FIRST(d.vid) AS vid, FIRST(d.start_time) AS start_time, FIRST(d.end_time) AS end_time, FIRST(d.feature) AS feature
            FROM '{feature_dir}/*.parquet' d, clips c
            WHERE d.vid=c.vid
                AND d.start_time >= c.start_time
                AND d.end_time <= c.end_time
            GROUP BY d.fid
        """.format(feature_dir=feature_dir)).arrow()
        return result

    @read_func_multifeature
    def get_label_counts(self, feature_names: List[str]=None, dbcon=None):
        if len(feature_names) == 1:
            feature_dir = self.feature_dir(feature_names[0])
            if not os.path.exists(feature_dir):
                return None
            result = dbcon.execute("""
                SELECT a.label, count(distinct d.fid)
                FROM annotations a,
                    '{feature_dir}/*.parquet' d
                WHERE a.vid=d.vid
                    AND d.start_time >= a.start_time
                    AND d.end_time <= a.end_time
                GROUP BY a.label
            """.format(feature_dir=feature_dir)).fetchall()
        else:
            if len(feature_names) > 2:
                self.logger.warn(f'get_label_counts only supports 2 features; using the first two from {feature_names}')
            base_feature_dir = self.feature_dir(feature_names[0])
            overlay_feature_dir = self.feature_dir(feature_names[1])
            if not os.path.exists(base_feature_dir) or not os.path.exists(overlay_feature_dir):
                self.logger.warn(f'get_label_counts: one or more of {feature_names} does not exist')
                return None
            result = dbcon.execute("""
                SELECT a.label, count(distinct d.fid)
                FROM annotations a,
                    '{base_feature_dir}/*.parquet' d,
                    '{overlay_feature_dir}/*.parquet' o
                WHERE a.vid=d.vid
                    AND d.start_time >= a.start_time
                    AND d.end_time <= a.end_time
                    AND d.vid=o.vid
                    AND d.start_time <= o.start_time
                    AND o.end_time <= d.end_time
                GROUP BY a.label
            """.format(base_feature_dir=base_feature_dir, overlay_feature_dir=overlay_feature_dir)).fetchall()

        return {label: label_count for label, label_count in result}

    def get_labels(self, feature_dataset, dbcon, adjust_feature_time=True, full_overlap=True, include_feature=True, ignore_labels=[]):
        # This doesn't need to be in parquet.py.
        return self._get_labels_join(feature_dataset, dbcon, adjust_feature_time=adjust_feature_time, full_overlap=full_overlap, include_feature=include_feature, ignore_labels=ignore_labels)

    def get_nonaggregated_labels(self, clip_dataset, dbcon):
        # This doesn't need to be in parquet.py.
        # Inner join to ignore clips in clip_dataset without any labels.
        return dbcon.execute("""
            SELECT a.vid,
                CASE WHEN c.start_time < a.start_time THEN a.start_time ELSE c.start_time END AS start_time,
                CASE WHEN c.end_time < a.end_time THEN c.end_time ELSE a.end_time END AS end_time,
                a.label
            FROM clip_dataset c
                JOIN annotations a
                ON c.vid=a.vid
                    AND ((c.start_time <= a.start_time AND a.start_time <= c.end_time)
                        OR (c.start_time <= a.end_time AND a.end_time <= c.end_time)
                        OR (a.start_time <= c.start_time AND c.end_time <= a.end_time))
            """).arrow()

    def _get_labels_gby(self, feature_dataset, dbcon):
        dbcon.execute("""
            SELECT f.vid, f.start_time, f.end_time, f.feature, string_agg(a.label, '_') labels
            FROM feature_dataset f, annotations a
            WHERE f.vid=a.vid
                AND f.start_time >= a.start_time
                AND f.end_time <= a.end_time
            GROUP BY f.vid, f.start_time, f.end_time, f.feature
        """)
        return dbcon.arrow()

    @staticmethod
    def _join_query(adjust_feature_time, full_overlap=False, include_feature=True, ignore_labels=[]):
        query_parameters = []
        if adjust_feature_time:
            select = """
                SELECT labels.vid,
                    CASE WHEN labels.labels IS NULL THEN 'none' ELSE labels.labels END labels,
                    labels.vstart + to_microseconds(CAST(labels.start_time * 1e6 AS BIGINT)) start_time,
                    labels.vstart + to_microseconds(CAST(labels.end_time * 1e6 As BIGINT)) end_time
            """
        else:
            select = """
                SELECT labels.vid,
                    CASE WHEN labels.labels IS NULL THEN 'none' ELSE labels.labels END labels,
                    labels.start_time start_time,
                    labels.end_time end_time
            """
        if include_feature:
            select = select + ",f.fid,f.feature"

        join_type = "INNER JOIN" if len(ignore_labels) else "LEFT JOIN"
        base_p1 = """
            FROM feature_dataset f,
                (
                    SELECT fv.*,
                        string_agg(distinct a.label, '_') labels
                    FROM (
                        SELECT f.vid,
                            v.vstart,
                            f.start_time,
                            f.end_time
                        FROM feature_dataset f, video_metadata v
                        WHERE f.vid=v.vid
                    ) fv
                    {join_type} annotations a
            """.format(join_type=join_type)

        if full_overlap:
            join = """
                ON fv.vid=a.vid
                    AND fv.start_time >= a.start_time
                    AND fv.end_time <= a.end_time
            """
        else:
            join = """
                ON fv.vid=a.vid
                    AND (
                        (fv.start_time >= a.start_time AND fv.start_time <= a.end_time)
                        OR (fv.end_time >= a.start_time AND fv.end_time <= a.end_time)
                        OR (fv.start_time <= a.start_time AND a.end_time <= fv.end_time)
                    )
            """

        base_p2 = """
                    GROUP BY fv.vid, fv.vstart, fv.start_time, fv.end_time
                ) labels
            WHERE labels.vid = f.vid
                AND labels.start_time = f.start_time
                AND labels.end_time = f.end_time
            """
        if len(ignore_labels):
            parameters = ','.join(['?' for _ in ignore_labels])
            # Remove ignore labels from joining with features. It's possible to still return "none"
            # because of the left join.
            base_p2 = "AND a.label NOT IN ({parameters})".format(parameters=parameters) + base_p2
            query_parameters.extend(ignore_labels)

        return select + base_p1 + join + base_p2, query_parameters

    def _get_labels_join(self, feature_dataset, dbcon, adjust_feature_time=True, full_overlap=True, include_feature=True, ignore_labels=[]):
        # Combine from v.vstart in global time and f.start/end_time in local-video time.
        # f.start/end_time is a double representing seconds.
        return dbcon.execute(*self._join_query(adjust_feature_time=adjust_feature_time, full_overlap=full_overlap, include_feature=include_feature, ignore_labels=ignore_labels)).arrow()

    def flatten_features_and_labels(self, labels_and_feature_table):
        pass
