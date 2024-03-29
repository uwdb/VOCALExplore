import atexit
import datetime
import duckdb
from functools import lru_cache
import logging
import numpy as np
import os
import pyarrow as pa
import signal
import sqlite3
import threading
import traceback
from typing import Iterable, List, Tuple, Dict, Union

from vfe import core
from vfe.featurestore.parquet import ParquetFeatureStore as FeatureStore

from .abstract import AbstractStorageManager, VidType, FeatureSet, LabeledFeatureSet, LabelInfo, ClipSet, LabeledClipSet, LabelSet, ModelInfo, ClipInfo, ClipInfoWithPath, PredictionRow

def add_delta(dt, seconds):
    if isinstance(dt, datetime.datetime):
        return dt + datetime.timedelta(seconds=seconds)
    elif isinstance(dt, np.datetime64):
        return dt + np.timedelta64(seconds, 's')

class BasicStorageManager(AbstractStorageManager):
    def __init__(self, db_dir, features_dir, models_dir, read_only=False, fid_offset=0, full_overlap=True):
        # Metadata and annotations.
        # Initialize db_dir, _con, _con_type before running setup statements.
        self.db_dir = db_dir
        self.full_overlap = full_overlap
        self._con = None
        self._cursors = {}
        self._sqlite_connections = {}
        self._con_type = read_only
        if not read_only:
            self._run_setup_stmts()

        self.logger = logging.getLogger(__name__)

        # Features.
        self.features_dir = features_dir
        self.featurestore = FeatureStore(base_dir=self.features_dir, fid_offset=fid_offset)

        # Models.
        self.models_dir = models_dir

        atexit.register(self.shutdown)

    def shutdown(self):
        self.logger.info('Clearing feature store')
        self.featurestore = None

    @property
    def metadata_db_path(self):
        return os.path.join(self.db_dir, 'annotations.duckdb')

    @property
    def predictions_db_path(self):
        return os.path.join(self.db_dir, 'predictions.db')

    def get_cursor(self, read_only=False):
        # DuckDB's documentation seems to indicate that connections are threadsafe, but in practice
        # it throws an error: RuntimeError: DuckDB objects created in a thread can only be used in that same thread.

        # Eventually switch to a read_only connection if possible so more concurrent queries can execute.
        if self._con is None:
            self._con = duckdb.connect(self.metadata_db_path, read_only=self._con_type)
            self._cursors = {}
        thread_id = threading.get_ident()
        if thread_id not in self._cursors:
            self._cursors[thread_id] = self._con.cursor()
        return self._cursors[thread_id]

    def get_sqlite_cursor(self, read_only=False):
        thread_id = threading.get_ident()
        if thread_id not in self._sqlite_connections:
            self._sqlite_connections[thread_id] = sqlite3.connect(self.predictions_db_path)
        return self._sqlite_connections[thread_id]

    def _close(self):
        self._con = None
        self._cursors = {}

    def _run_setup_stmts(self):
        conn = self.get_cursor()

        # video_metadata table.
        conn.execute("CREATE SEQUENCE IF NOT EXISTS vid_seq")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_metadata(
                vid BIGINT PRIMARY KEY,
                vpath VARCHAR UNIQUE,
                vstart TIMESTAMP,
                vduration DOUBLE,
                thumbpath VARCHAR UNIQUE DEFAULT NULL,
            )
        """)

        # dataset_splits table.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_splits(
                name VARCHAR,
                split_type VARCHAR,
                split_idx UINTEGER,
                split VARCHAR,
                vids BIGINT[],
                PRIMARY KEY(name, split_type, split_idx, split)
            )
        """)

        # annotations table.
        conn.execute("CREATE SEQUENCE IF NOT EXISTS annotation_seq")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations(
                aid BIGINT DEFAULT(nextval('annotation_seq')),
                vid BIGINT,
                start_time DECIMAL,
                end_time DECIMAL,
                label VARCHAR,
                label_time TIMESTAMP,
                UNIQUE (vid, start_time, end_time, label),
                FOREIGN KEY (vid) REFERENCES video_metadata(vid)
            )
        """)

        # models table.
        conn.execute("CREATE SEQUENCE IF NOT EXISTS mid_seq")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models(
                mid BIGINT PRIMARY KEY,
                model_type VARCHAR,
                feature_name VARCHAR,
                creation_time TIMESTAMP,
                batch_size UINTEGER,
                epochs UINTEGER,
                learningrate DOUBLE,
                ntrain UINTEGER,
                labels VARCHAR[],
                model_path VARCHAR,
                labels_path VARCHAR,
                f1_threshold DOUBLE,
            )
        """)

        # predictions table.
        sqlite_cursor = self.get_sqlite_cursor()
        sqlite_cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions(
                mid INTEGER,
                vid INTEGER,
                start_time DECIMAL(10,5),
                end_time DECIMAL(10,5),
                label TEXT,
                probability REAL,
                -- We can't actually specify these foreign keys because it's a different database.
                -- FOREIGN KEY (mid) REFERENCES models(mid),
                -- FOREIGN KEY (vid) REFERENCES video_metadata(vid)
                UNIQUE(mid, vid, start_time, end_time, label, probability)
            )
        """)

    def _add_videos(self, video_df, include_thumbpath=False):
        conn = self.get_cursor()

        # Import video_info_csv_path.
        conn.execute("BEGIN TRANSACTION")
        conn.execute("""
            CREATE TEMP TABLE video_metadata_raw
            AS SELECT * FROM video_df
        """)
        # "returning" statement isn't working.
        # Work around by getting a set of vids before and after then differencing them.
        old_vids = set(conn.execute("""
            SELECT vid
            FROM video_metadata
        """).fetchnumpy()['vid'].tolist()) # tolist to convert from numpy type to python type.
        extra_select = ", thumbpath" if include_thumbpath else ", NULL"
        conn.execute(f"""
            INSERT INTO video_metadata
            SELECT nextval('vid_seq') vid, path, start, duration {extra_select}
            FROM video_metadata_raw
            WHERE path NOT IN (
                SELECT vpath
                FROM video_metadata
            )
        """)
        conn.execute("DROP TABLE video_metadata_raw")
        new_vids = set(conn.execute("""
            SELECT vid
            FROM video_metadata
        """).fetchnumpy()['vid'].tolist()) # tolist to convert from numpy type to python type.
        conn.execute("COMMIT")
        return new_vids - old_vids

    def _add_video(self, video_path, start_time, duration, thumb_path = None):
        return self.get_cursor().execute("""
            INSERT INTO video_metadata
            (vid, vpath, vstart, vduration, thumbpath)
            VALUES (nextval('vid_seq'), ?, ?, ?, ?)
            RETURNING vid as vid
        """, [video_path, start_time, duration, thumb_path]).fetchall()[0][0]

    def add_video(self, path, start_time, duration, thumbnail_path=None) -> VidType:
        if start_time and not isinstance(start_time, str):
            start_time = str(start_time)
        return self._add_video(path, start_time, duration, thumbnail_path)

    def add_videos(self, video_df, include_thumbpath=False) -> Iterable[VidType]:
        return self._add_videos(video_df, include_thumbpath=include_thumbpath)

    def update_thumbnail_path(self, vid, thumbpath):
        self.get_cursor().execute("""
            UPDATE video_metadata
            SET thumbpath=?
            WHERE vid=?
        """, [thumbpath, vid])

    def get_video_paths(self, vids, thumbnails=False) -> Iterable[Tuple[VidType, str, Union[str, None]]]:
        select_str = "SELECT vid, vpath"
        if thumbnails:
            select_str += ", thumbpath"
        if vids is None or not len(vids):
            return self.get_cursor(read_only=True).execute("""
                {select_str}
                FROM video_metadata
            """.format(select_str=select_str)).fetchall()

        parameters = ','.join('?' for _ in vids)
        return self.get_cursor(read_only=True).execute("""
            {select_str}
            FROM video_metadata
            WHERE vid=ANY([{parameters}])
        """.format(parameters=parameters, select_str=select_str), [int(v) for v in vids]).fetchall()

    def _update_label(self, vid, start, end, add_label=None, remove_label=None) -> bool:
        conn = self.get_cursor()

        success = False
        # We'll do at most two loops of this before success will always be True.
        # The returned value may be True even if the query failed for some reason.
        # Actually returning a useful success value will require inspecting the exeptions.
        # For example, failing to insert a label if it doesn't exist yet is a true error.
        # But failing to insert a label that already exists is not really an error.
        while not success:
            success = True
            if add_label and not remove_label:
                try:
                    conn.execute("BEGIN TRANSACTION")
                    conn.execute("""
                        INSERT INTO annotations (vid, start_time, end_time, label, label_time)
                        VALUES (?, ?, ?, ?, now())
                    """, [vid, start, end, add_label])
                    # Get all annotations for the same vid and label that overlap in time.
                    conn.execute("""
                        CREATE TEMP TABLE overlapping_annotations AS
                        SELECT *
                        FROM annotations
                        WHERE vid=?
                            AND label=?
                            AND (
                                (start_time<=? AND ?<=end_time)
                                OR (start_time<=? AND ? <= end_time)
                                OR (?<=start_time AND end_time<=?)
                            )
                    """, [vid, add_label, start, start, end, end, start, end])
                    # If there is more than one row in overlapping_annotations, delete these rows from annotations
                    # and insert one row with the maximum annotations extent.
                    n_overlapping = conn.execute("SELECT COUNT(*) FROM overlapping_annotations").fetchall()[0][0]
                    # We Should wait until after delete/insert to commit, but duckdb fails when deleting/updating
                    # in a single transaction: https://github.com/duckdb/duckdb/issues/3789
                    # For now, commit early. If insert worked above, there shouldn't be any uniqueness
                    # issues with the delete/insert statements.
                    conn.execute("COMMIT")
                    if n_overlapping > 1:
                        self.logger.debug(f'Coalescing labels; found {n_overlapping} overlapping')
                        conn.execute("""
                            DELETE FROM annotations
                            WHERE aid IN (
                                SELECT aid FROM overlapping_annotations
                            )
                        """)
                        conn.execute("""
                            INSERT INTO annotations
                            (vid, start_time, end_time, label, label_time)
                            SELECT vid, min(start_time), max(end_time), label, now()
                            FROM overlapping_annotations
                            GROUP BY vid, label
                        """)
                except Exception as e:
                    print(f'Failed to add label with exception {e}')
                    conn.execute("ROLLBACK")
                finally:
                    conn.execute("DROP TABLE IF EXISTS overlapping_annotations")
            elif not add_label and remove_label:
                try:
                    conn.execute("""
                        DELETE FROM annotations
                        WHERE vid=? AND start_time=? AND end_time=? AND label=?
                    """, [vid, start, end, remove_label])
                except Exception as e:
                    print(f'Failed to remove label with exception {e}')
            else:
                try:
                    conn.execute("""
                        UPDATE annotations
                        SET label=?
                        WHERE vid=? AND start_time=? AND end_time=? AND label=?
                    """, [add_label, vid, start, end, remove_label])
                except Exception as e:
                    # Updating failed, possibly because add_label already exists in the table.
                    # In that case, try to just remove the old label.
                    print(f'Update failed; falling back to attempt removing old label. Exception {e}')
                    add_label = None
                    success = False
        return success

    def update_label(self, vid, start, end, add_label=None, remove_label=None) -> bool:
        if add_label is not None and '_' in add_label:
            raise RuntimeError(f'Labels should not contain underscores: "_"; attempted to add {add_label}')

        return self._update_label(vid, start, end, add_label, remove_label)

    def add_labels(self, labels: Iterable[LabelInfo]) -> bool:
        for label_info in labels:
            self._update_label(label_info.vid, label_info.start_time, label_info.end_time, label_info.label)

    def update_labels(self, labels: Iterable[LabelInfo]) -> None:
        for label_info in labels:
            conn = self.get_cursor()
            conn.execute("""
                UPDATE annotations
                SET label=?, start_time=?, end_time=?
                WHERE aid=?
            """, [label_info.label, label_info.start_time, label_info.end_time, label_info.lid])

    def delete_labels(self, labels: Iterable[LabelInfo]) -> None:
        for label_info in labels:
            conn = self.get_cursor()
            conn.execute("""
                DELETE FROM annotations
                WHERE aid=?
            """, [label_info.lid])

    def remove_label(self, vid, start, end, label) -> bool:
        return self.update_label(vid, start, end, remove_label=label)

    def add_labels_bulk(self, label_csv_path) -> None:
        conn = self.get_cursor()

        conn.execute("BEGIN TRANSACTION")
        conn.execute("""
            CREATE TEMP TABLE annotations_raw
            AS SELECT * FROM read_csv_auto('{csv_path}')
        """.format(csv_path=label_csv_path))
        try:
            conn.execute("""
                INSERT INTO annotations
                    (vid, start_time, end_time, label, label_time)
                    SELECT vm.vid, ar.start, ar.end, ar.label, now()
                    FROM video_metadata vm, annotations_raw ar
                    WHERE ar.path=vm.vpath
            """)
            conn.execute("COMMIT")
        except Exception as e:
            print(f'Bulk label insertion failed with exception {e}')
            conn.execute("ROLLBACK")

    def add_feature_batch(self, feature_name, vids, starts, ends, feature_vectors) -> None:
        if self._con_type:
           self.logger.warning('Should not add features when storage manager is read only.')
        self.featurestore.insert_batch(feature_name, vids, starts, ends, feature_vectors)

    def add_feature(self, feature_name, vid, start, end, feature_vector) -> None:
        if self._con_type:
            raise RuntimeError('Cannot add features when storage manager is read only.')
        self.add_feature_batch(feature_name, [vid], [start], [end], [feature_vector])

    def get_labels(self, vids=None, before_label_time=None, ignore_labels=[]) -> Iterable[LabelInfo]:
        if before_label_time is not None:
            where_clause = "WHERE label_time < ?"
            query_parameters = [before_label_time]
        else:
            where_clause = "WHERE true"
            query_parameters = []

        if vids is not None:
            parameters = ','.join('?' for _ in vids)
            base_query = """
                SELECT aid, vid, start_time, end_time, label
                FROM annotations
                {where_clause}
                    AND vid=ANY([{parameters}])
            """.format(where_clause=where_clause, parameters=parameters)
            query_parameters.extend(core.typecheck.ensure_list(vids))
        else:
            base_query = """
                SELECT aid, vid, start_time, end_time, label
                FROM ANNOTATIONS
                {where_clause}
            """.format(where_clause=where_clause)

        if len(ignore_labels):
            parameters = ','.join('?' for _ in ignore_labels)
            base_query += "AND label NOT IN ({parameters})".format(parameters=parameters)
            query_parameters.extend(ignore_labels)

        results = self.get_cursor(read_only=True).execute(base_query, query_parameters).fetchall()
        return map(LabelInfo._make, results)

    def get_vids_with_labels(self) -> Iterable[int]:
        return self.get_cursor(read_only=True).execute("SELECT DISTINCT vid FROM annotations").fetchnumpy()['vid']

    def get_all_vids(self) -> Iterable[int]:
        return self.get_cursor(read_only=True).execute("SELECT DISTINCT vid FROM video_metadata").fetchnumpy()['vid']

    def get_vids_for_paths(self, paths_csv) -> Iterable[VidType]:
        return self.get_cursor(read_only=True).execute("""
            SELECT vid
            FROM video_metadata v, read_csv_auto('{paths_csv}', header=True) paths
            WHERE v.vpath=paths.vpath
        """.format(paths_csv=paths_csv)).fetchnumpy()['vid']

    def get_distinct_labels(self) -> List[str]:
        return self.get_cursor(read_only=True).execute("SELECT DISTINCT label FROM annotations").fetchnumpy()['label']

    def get_labels_for_features(self, featureset: FeatureSet, ignore_labels=[]) -> LabeledFeatureSet:
        conn = self.get_cursor(read_only=True)
        return self.featurestore.get_labels(featureset, conn, adjust_feature_time=False, include_feature=True, ignore_labels=ignore_labels, full_overlap=self.full_overlap)

    def get_labels_for_clips_aggregated_fulloverlap(self, clipset: ClipSet, full_overlap=True) -> LabeledClipSet:
        if full_overlap != self.full_overlap:
            self.logger.warning(f"full_overlap={full_overlap}, but overriding to {self.full_overlap}")
        conn = self.get_cursor(read_only=True)
        return self.featurestore.get_labels(clipset, conn, full_overlap=self.full_overlap, adjust_feature_time=False, include_feature=False)

    def get_labels_for_clips_nonaggregated_overlapping(self, clipset: ClipSet) -> LabelSet:
        conn = self.get_cursor(read_only=True)
        return self.featurestore.get_nonaggregated_labels(clipset, conn)

    def get_label_counts(self, feature_names: Union[str, List[str]]) -> Dict[str, int]:
        feature_names = core.typecheck.ensure_list(feature_names)
        conn = self.get_cursor(read_only=True)
        return self.featurestore.get_label_counts(feature_names=feature_names, dbcon=conn, full_overlap=self.full_overlap)

    def get_unique_labels(self) -> Iterable[str]:
        rows = self.get_cursor(read_only=True).execute("""
            SELECT DISTINCT label FROM annotations
        """).fetchall()
        return [row[0] for row in rows]

    def get_total_label_time(self) -> Dict[str, float]:
        conn = self.get_cursor(read_only=True)
        label_dur = conn.execute("""
            SELECT label, SUM(end_time - start_time) AS duration
            FROM annotations
            GROUP BY label
        """).fetchall()
        return {label: duration for label, duration in label_dur}

    def get_stored_feature_vids(self, feature_names: Union[str, List[str]]) -> Iterable[VidType]:
        # Return the vids that have all specified features.
        feature_names = core.typecheck.ensure_list(feature_names)
        base_vids = set(self.featurestore.get_vids(feature_name=feature_names[0]))
        for feature_name in feature_names[1:]:
            base_vids &= set(self.featurestore.get_vids(feature_name=feature_name))
        return base_vids

    def get_feature_names(self) -> Iterable[str]:
        return self.featurestore.get_feature_names()

    def get_features(self, feature_names: Union[str, List[str]], vids=None) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)
        return self.featurestore.get_nonaggregated_dataset_concat(feature_names=feature_names, vids=vids)

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)
        return self.featurestore.get_features_for_clips_concat(feature_names=feature_names, clips=clipset, full_overlap=self.full_overlap)

    def _add_model(self, model_type, feature_name, creation_time, batch_size, epochs, learningrate, ntrain, labels, model_path, labels_path, f1_threshold):
        conn = self.get_cursor()

        conn.execute("""
            INSERT INTO models
            (mid, model_type, feature_name, creation_time, batch_size, epochs, learningrate, ntrain, labels, model_path, labels_path, f1_threshold)
            VALUES (nextval('mid_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [model_type, feature_name, creation_time, batch_size, epochs, learningrate, ntrain, labels, model_path, labels_path, f1_threshold])

    def get_models_dir(self):
        return self.models_dir

    def add_model(
        self,
        model_type: str,
        feature_name: str,
        creation_time: np.datetime64,
        batch_size: int,
        epochs: int,
        learningrate: float,
        ntrain: int,
        labels: List[str],
        model_path: str,
        labels_path: str,
        f1_threshold: float,
    ) -> None:
        self._add_model(model_type, feature_name, creation_time, batch_size, epochs, learningrate, ntrain, labels, model_path, labels_path, f1_threshold)

    def get_model_info(self, feature_name, ignore_labels=[], include_labels=[]) -> ModelInfo:
        base_query = """
            SELECT model_type, model_path, labels, feature_name, f1_threshold, mid
            FROM models
        """
        if feature_name is not None:
            base_query += " WHERE feature_name=?"
            query_parameters = [feature_name]
        else:
            query_parameters = []

        if ignore_labels:
            # Filter to rows where labels does not contain any of the ignored labels. These rows will have
            # the same labels length after filtering out the ignored labels.
            parameters = ','.join(['?' for _ in ignore_labels])
            base_query += """
                AND LENGTH(labels) = LENGTH(LIST_FILTER(labels, x -> NOT LIST_CONTAINS([{parameters}], x)))
            """.format(parameters=parameters)
            query_parameters.extend(ignore_labels)

        if include_labels:
            # Filter to rows where labels contains all of the specified include labels.
            parameters = ','.join(['?' for _ in include_labels])
            base_query += """
                AND ? = LENGTH(LIST_FILTER(labels, x -> LIST_CONTAINS([{parameters}, x])))
            """.format(parameters=parameters)
            query_parameters.append(len(include_labels))
            query_parameters.extend(include_labels)

        base_query += """
            ORDER BY creation_time DESC
            LIMIT 1
        """
        results = self.get_cursor(read_only=True).execute(base_query, query_parameters).fetchall()
        return ModelInfo._make(results[0]) if len(results) else None

    def get_model_info_for_mid(self, mid) -> ModelInfo:
        results = self.get_cursor(read_only=True).execute("""
            SELECT model_type, model_path, labels, feature_name, f1_threshold, mid
            FROM models
            WHERE mid=?
        """, [mid]).fetchall()
        # We always expect the specified mid to exist. If not, this indexing will throw an error.
        return ModelInfo._make(results[0])

    @lru_cache
    def get_clip_splits(self, vids, clip_duration) -> Iterable[ClipInfo]:
        # This isn't ideal because if the specified duration is longer than any single video, nothing will
        # be returned even if clip_duration could be achieved by concatenating consecutive videos.
        parameters = ','.join(['?' for _ in vids])
        results = self.get_cursor(read_only=True).execute("""
            SELECT vid, vstart, clip_start, clip_start + ? AS clip_end
            FROM (
                SELECT vid, vstart, UNNEST(RANGE(CAST(vduration / ? AS BIGINT))) * ? AS clip_start
                FROM video_metadata
                WHERE vid=ANY([{parameters}])
            ) starts
        """.format(parameters=parameters), [clip_duration, clip_duration, clip_duration, *map(int, vids)]).fetchall()
        # Return a list rather than `map` so that the result isn't depleted if we get a cached result.
        return [ClipInfo._make(result) for result in results]

    def get_adjacent_vids(self, vids) -> Iterable[VidType]:
        parameters = ','.join(['?' for _ in vids])
        results = self.get_cursor(read_only=True).execute("""
            WITH vmr AS (SELECT vid, row_number() OVER (ORDER BY vstart) AS "row" FROM video_metadata)
            SELECT DISTINCT vid
            FROM vmr
            WHERE "row" in (
                SELECT UNNEST(LIST_VALUE("row", "row"+1, "row"-1))
                FROM vmr
                WHERE vid=ANY([{parameters}])
            )
        """.format(parameters=parameters), vids).fetchnumpy()
        return results['vid']

    def get_physical_clips_for_clip(self, clip_info: ClipInfo) -> Iterable[ClipInfoWithPath]:
        if clip_info.vstart is None:
            vstart = self.get_cursor(read_only=True).execute("SELECT vstart FROM video_metadata WHERE vid=?", [clip_info.vid]).fetchall()[0][0]
        else:
            vstart = clip_info.vstart

        if vstart is None:
            start = max(0, clip_info.start_time)
            video_duration, vpath, thumbpath = self.get_cursor(read_only=True).execute("SELECT vduration, vpath, thumbpath FROM video_metadata WHERE vid=?", [clip_info.vid]).fetchall()[0]
            end = min(video_duration, clip_info.end_time)
            returnvals= [ClipInfoWithPath(clip_info.vid, vstart, start, end, vpath, thumbpath)]
        else:
            realtime_start = add_delta(vstart, clip_info.start_time)
            realtime_end = add_delta(vstart, clip_info.end_time)
            returnvals= self.get_clips_for_timespan(realtime_start, realtime_end)
        returnvals = list(returnvals)
        return returnvals

    def get_clipinfo_with_path(self, vids: List[VidType]) -> List[ClipInfoWithPath]:
        qmarks = ", ".join(["?" for _ in vids])
        results = self.get_cursor(read_only=True).execute(
            f"SELECT vid, vstart, 0, vduration, vpath, thumbpath FROM video_metadata WHERE vid IN ({qmarks})",
            vids
        ).fetchall()
        return list(map(ClipInfoWithPath._make, results))

    def get_clips_for_timespan(self, realtime_start, realtime_end) -> Iterable[ClipInfoWithPath]:
        # Not ideal to format string, but it's a read-only query and it's cleaner.
        results = self.get_cursor(read_only=True).execute("""
            SELECT vid,
                vstart,
                CASE WHEN vstart > start_timestamp THEN 0 ELSE DATE_SUB('milliseconds', vstart, start_timestamp) / 1000.0 END AS clip_start,
                CASE WHEN vstart + TO_SECONDS(CAST(vduration AS BIGINT)) < end_timestamp THEN vduration ELSE DATE_SUB('milliseconds', vstart, end_timestamp) / 1000.0 END AS clip_end,
                vpath,
                thumbpath
            FROM video_metadata, (
                SELECT CAST('{start}' AS TIMESTAMP) AS start_timestamp,
                    CAST('{end}' AS TIMESTAMP) AS end_timestamp
            ) timestamps
            WHERE (start_timestamp <= vstart AND vstart < end_timestamp)
                OR (start_timestamp < vstart + TO_SECONDS(CAST(vduration AS BIGINT)) AND vstart + TO_SECONDS(CAST(vduration AS BIGINT)) <= end_timestamp)
                OR (vstart <= start_timestamp AND vstart + TO_SECONDS(CAST(vduration AS BIGINT)) >= end_timestamp)
            ORDER BY vstart
        """.format(start=realtime_start, end=realtime_end)).fetchall()
        return map(ClipInfoWithPath._make, results)

    def _add_predictions(self, values, parameters):
        query = """
            INSERT INTO predictions
                (mid, vid, start_time, end_time, label, probability)
            VALUES {values}
        """
        values_str = ','.join(values)
        conn = self.get_sqlite_cursor()
        # Use a context manager to commit after adding values.
        # If this fails because of a unique error, go 1-by-1 and add the ones that are not unique.
        try:
            with conn:
                conn.execute(query.format(values=values_str), parameters)
        except Exception as e:
            self.logger.exception(f"Bulk-adding predictions failed with exception {e}")
            offset = 0
            for qmarks in values:
                nparams = qmarks.count('?')
                try:
                    with conn:
                        conn.execute(f"""
                            INSERT INTO predictions
                                (mid, vid, start_time, end_time, label, probability)
                            VALUES {qmarks}
                        """, parameters[offset:offset+nparams])
                except:
                    # May fail if this is a duplicate.
                    pass
                offset += nparams

    def add_predictions(self, mid, predictionset_list):
        self.logger.debug(f'Adding {len(predictionset_list)} predictions')
        # predictionset: (vid, start_time, end_time, predictions {label: prob,})

        parameters = []
        values = []
        for predictionset in predictionset_list:
            for label, probability in predictionset.predictions.items():
                values.append('(?, ?, ?, ?, ?, ?)')
                vals = (mid, predictionset.vid, predictionset.start_time, predictionset.end_time, label, probability)
                parameters.extend(vals)

            # Try to avoid error "Too many sql parameters"
            if len(parameters) > 30_000:
                self._add_predictions(values, parameters)
                parameters, values = [], []

        if len(parameters):
            self._add_predictions(values, parameters)

    def get_vids_with_predictions(self, mid) -> List[VidType]:
        conn = self.get_sqlite_cursor(read_only=True)
        rows = conn.execute("SELECT DISTINCT vid FROM predictions WHERE mid=?", [mid]).fetchall()
        return [row[0] for row in rows]

    def get_predictions(self, mid, labels_and_features):
        conn = self.get_sqlite_cursor(read_only=True)
        vids = labels_and_features['vid'].to_pylist()
        vid_params = ', '.join('?' for _ in vids)
        rows = conn.execute(f"""
            SELECT vid, start_time, end_time, JSON_GROUP_OBJECT(label, probability) as pred_dict
            FROM predictions
            WHERE mid=?
                AND vid IN ({vid_params})
            GROUP BY vid, start_time, end_time
        """, [mid, *vids]).fetchall()
        tupled_rows = [pr._asdict() for pr in map(PredictionRow._make, rows)]
        if tupled_rows:
            return pa.Table.from_pylist(tupled_rows)
        else:
            return pa.table({
                "vid": [],
                "start_time": [],
                "end_time": [],
                "pred_dict": [],
            })

    def search_videos(self, date_range=None, labels=None, predictions=None, prediction_confidence=None, mid=None) -> List[ClipInfo]:
        params = []
        where_clause = ""
        if date_range:
            # date_range is a 2-element array of [start, end].
            # If we don't cast the datetime's to strings, we get a weird date out-of-range error.
            where_clause += "AND ? <= vstart AND vstart <= ?"
            params.extend([str(d) for d in date_range])

        if labels:
            joined_labels = ", ".join(["?" for _ in labels])
            params.extend(labels)
            where_clause += f"""AND vid IN (
                SELECT vid
                FROM annotations
                WHERE label IN ({joined_labels})
            )"""
        if predictions:
            # Get vids. Then add them to the where clause.
            assert mid is not None
            joined_predictions = ", ".join(["?" for _ in predictions])
            pred_params = []
            # Copy predictions because otherwise the typing is weird.
            pred_params.extend(predictions)
            pred_params.append(prediction_confidence)
            pred_params.append(mid)
            pred_vids = self.get_sqlite_cursor(read_only=True).execute(f"""
                SELECT DISTINCT vid
                FROM predictions
                WHERE label IN ({joined_predictions})
                    AND probability > ?
                    AND mid = ?
            """, pred_params).fetchall()

            if pred_vids:
                vids = ", ".join(str(row[0]) for row in pred_vids)
                where_clause += f"""AND vid IN ({vids})"""
            else:
                where_clause += "AND False"

        query = f"""
            SELECT DISTINCT v.vid, vstart, 0 as clip_start, vduration as clip_end
            FROM video_metadata v
            WHERE True
                {where_clause}
        """
        results = self.get_cursor(read_only=True).execute(query, params).fetchall()
        return [ClipInfo._make(result) for result in results]

    def reset_annotations(self):
        # Intended for debugging only.
        conn = self.get_cursor()
        conn.execute("truncate table models")
        conn.execute("truncate table annotations")
        conn.commit()

        pconn = self.get_sqlite_cursor()
        with pconn:
            pconn.execute("delete from predictions")
