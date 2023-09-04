from collections import defaultdict
import duckdb
import logging
import numpy as np
from scipy.spatial import distance
from typing import Iterable, List

from vfe import core
from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer
from vfe.api.scheduler import UserPriority

class CoresetsExplorer(AbstractExplorer):
    def __init__(self, rng=None, missing_vids_X=-1):
        self.rng = np.random.default_rng(rng)
        self.missing_vids_X = missing_vids_X
        self.logger = logging.getLogger(__name__)
        self._min_distances = {}
        self._shown_fids = defaultdict(set)
        self._coreset_fids = defaultdict(set)

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None, step=None) -> Iterable[ClipInfo]:
        self.logger.debug(f'Coresets explore (features {feature_names}) over unlabeled vids with features')

        feature_name_str = core.typecheck.ensure_str(feature_names)

        # Make sure there are enough features extracted.
        vids_with_features, vids_without_features = featuremanager.get_extracted_features_info(feature_names)
        labeled_vids = set(modelmanager.get_vids_with_labels())
        unlabeled_vids_with_features = set(vids_with_features) - labeled_vids
        self.logger.debug(f'{len(unlabeled_vids_with_features)} unlabeled vids with features; {len(vids_without_features)} vids without features')
        desired_vids = max(k, self.missing_vids_X)
        if len(unlabeled_vids_with_features) < desired_vids:
            candidate_vids = set(vids_without_features) - labeled_vids
            if candidate_vids:
                vids_to_extract = self.rng.choice(np.array([*candidate_vids]), size=min(desired_vids, len(candidate_vids)), replace=False)
                self.logger.info(f'Before doing coresets, extracting features from {len(vids_to_extract)} new vids')
                featuremanager.get_features(feature_names, vids_to_extract, priority=UserPriority.priority)

        # Reading feature vectors is actually only necessary when we can't use the cached distances.
        # We could instead initially read just the clip info metadata for the features, and then
        # only add the features column when we determine it's necessary.
        features = featuremanager.get_features(feature_names, vids=vids)
        try:
            # This will fail if features is already a table.
            features = features.to_table()
        except:
            pass
        features_plus_labels = modelmanager.get_labels_for_clips(features, full_overlap=True)
        features_plus_labels = duckdb.connect().execute("""
            SELECT l.vid, l.start_time, l.end_time, l.labels, f.feature, f.fid
            FROM features_plus_labels l, features f
            WHERE l.vid=f.vid AND l.start_time=f.start_time AND l.end_time=f.end_time
            ORDER BY f.fid
        """).arrow()
        X = np.vstack(features_plus_labels['feature'].to_numpy())
        labeled_idxs = np.where(features_plus_labels['labels'].to_numpy() != 'none')[0]
        self._shown_fids[feature_name_str] |= set(features_plus_labels['fid'].take(labeled_idxs).unique().to_pylist())
        coreset_idxs = np.where(np.isin(features_plus_labels['fid'].to_numpy(), np.array([*self._coreset_fids[feature_name_str]])))[0]
        if len(coreset_idxs) and (feature_name_str not in self._min_distances or len(X) != len(self._min_distances[feature_name_str])):
            # Reshape from (len(X),) to (len(X), 1). Otehrwise np.minimum leads to a (len(X), len(X)) array.
            self._min_distances[feature_name_str] = np.expand_dims(np.amin(distance.cdist(X, X[coreset_idxs]), axis=1), axis=1)

        clips = []
        while len(clips) < k:
            if len(coreset_idxs) == 0:
                furthest_clip_idx = self.rng.choice(len(X))
                furthest_distance = -1
            else:
                furthest_clip_idx = np.argmax(self._min_distances[feature_name_str])
                furthest_distance = self._min_distances[feature_name_str][furthest_clip_idx]

            distance_to_furthest = distance.cdist(X, np.expand_dims(X[furthest_clip_idx], axis=0))
            if feature_name_str in self._min_distances:
                self._min_distances[feature_name_str] = np.minimum(self._min_distances[feature_name_str], distance_to_furthest)
            else:
                self._min_distances[feature_name_str] = distance_to_furthest
            coreset_idxs = np.append(coreset_idxs, furthest_clip_idx)
            row = features_plus_labels.take([furthest_clip_idx]).to_pylist()[0]
            self._coreset_fids[feature_name_str].add(row['fid'])
            if row['fid'] in self._shown_fids[feature_name_str]:
                self.logger.debug(f'Skipping ({row["vid"]}, ({row["start_time"]:.2f}-{row["end_time"]:.2f}) because it was already shown')
                continue
            self._shown_fids[feature_name_str].add(row['fid'])
            clip = ClipInfo(vid=row['vid'], vstart=None, start_time=row['start_time'], end_time=row['end_time'])
            self.logger.debug(f'Adding clip ({clip.vid}, {clip.start_time:.2f}-{clip.end_time:.2f}), distance={float(furthest_distance):.3f}')
            clips.append(clip)
        return clips
