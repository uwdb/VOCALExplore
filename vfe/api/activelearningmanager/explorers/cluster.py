from collections import defaultdict
import logging
import numpy as np
from scipy.spatial import distance
import sklearn
from sklearn_extra import cluster
from typing import Iterable, List

from vfe import core
from vfe.api.storagemanager import ClipInfo, clipinfo_to_clipset
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

class ClusterExplorer(AbstractExplorer):
    def __init__(self, rng=None):
        self.random_state = np.random.RandomState(rng)
        self.logger = logging.getLogger(__name__)
        self._kmedoids_failed = defaultdict(bool)
        self._nclusters = -1

    def _cluster_kmeans(self, X, n_clusters):
        self.logger.debug(f'KMeans clustering, n_clusters={n_clusters}')
        clusterer = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=self.random_state).fit(X)
        labels = clusterer.labels_
        cluster_centers = clusterer.cluster_centers_
        medoid_indices = []
        for i, center in enumerate(cluster_centers):
            idxs = np.where(labels == i)[0]
            distances = distance.cdist(X[idxs], np.expand_dims(center, axis=0))
            closest = np.argmin(distances)
            medoid_indices.append(idxs[closest])
        return labels, medoid_indices

    def _cluster_kmedoids(self, X, n_clusters):
        self.logger.debug(f'KMedoids clustering, n_clusters={n_clusters}')
        clusterer = cluster.KMedoids(n_clusters=n_clusters, init='k-medoids++', metric='euclidean', method='alternate', random_state=self.random_state).fit(X)
        return clusterer.labels_, clusterer.medoid_indices_

    def _cluster(self, feature_names, X, n_clusters):
        feature_names_str = core.typecheck.ensure_str(feature_names)
        X_mem = X.shape[0]**2 * 4 / 1024 / 1024 / 1024 # GB
        if not self._kmedoids_failed[feature_names_str] and X_mem < core.consts.KMEDOID_CLUSTER_MAX_MEM:
            try:
                return self._cluster_kmedoids(X, n_clusters)
            except Exception as e:
                self.logger.warning(f'KMedoids clustering failed with exception: {e}')
                self._kmedoids_failed[feature_names_str] = True
        return self._cluster_kmeans(X, n_clusters)

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        self.logger.info(f'Cluster explore (features {feature_names})')

        # Make sure there are enough features extracted.
        vids_with_features, vids_without_features = featuremanager.get_extracted_features_info(feature_names)
        labeled_vids = set(modelmanager.get_vids_with_labels())
        unlabeled_vids_with_features = set(vids_with_features) - labeled_vids
        self.logger.debug(f'{len(unlabeled_vids_with_features)} unlabeled vids with features; {len(vids_without_features)} vids without features')
        if len(unlabeled_vids_with_features) < k:
            candidate_vids = set(vids_without_features) - labeled_vids
            vids_to_extract = self.random_state.choice(np.array([*candidate_vids]), size=min(k, len(candidate_vids)), replace=False)
            self.logger.info(f'Before doing clustering, extracting features from {len(vids_to_extract)} new vids')
            featuremanager.get_features(feature_names, vids_to_extract)

        features = featuremanager.get_features(feature_names, vids).to_table()
        X = np.vstack(features['feature'].to_numpy())
        if self._nclusters <= 0:
            self._nclusters = k
        else:
            # If k is larger than the last number of clusters we tried, bump up the value.
            # If the next k value is smaller, self._nclusters may be too big.
            self._nclusters = max(self._nclusters, k)
        clips = []
        while len(clips) < k:
            # Keep trying until we have k centroids that aren't already labeled.
            self.logger.debug(f'Clustering with n_clusters={self._nclusters}')
            labels, medoid_indices = self._cluster(feature_names, X, self._nclusters)
            # Iterate over centroids from largest to smallest cluster.
            clusters, cluster_counts = np.unique(labels, return_counts=True)
            clips = []
            for cluster_idx in np.argsort(-1 * cluster_counts):
                centroid_row = features.take([medoid_indices[cluster_idx]]).to_pylist()[0]
                clip = ClipInfo(vid=centroid_row['vid'], vstart=None, start_time=centroid_row['start_time'], end_time=centroid_row['end_time'])
                # Check if centroid is labeled.
                label = modelmanager.get_labels_for_clips(clipinfo_to_clipset([clip]), full_overlap=False).to_pylist()[0]['labels']
                if label != 'none':
                    self.logger.debug(f'Skipping clip ({clip.vid}, {clip.start_time:.2f}-{clip.end_time:.2f}) because it is already labeled')
                    continue
                self.logger.debug(f'Adding clip ({clip.vid}, {clip.start_time:.2f}-{clip.end_time:.2f})')
                clips.append(clip)
                if len(clips) == k:
                    break
            if len(clips) < k:
                self._nclusters *= 2
        return clips
