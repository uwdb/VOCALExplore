import numpy as np
from sklearn import cluster, preprocessing

from .abstractcoordinator import AbstractCoordinator

class BaseKMeansCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState(rng)
        # For now assume we have a single feature.
        self.X = next(iter(self.Xs.items()))[1]
        self.cluster_assignments, self.cluster_centers = self._cluster(self.X)
        self._not_handled_clusters = set(np.unique(self.cluster_assignments))

    def _pick_cluster_element(self, cluster_elements, cluster_center):
        raise NotImplementedError

    def _interaction_round(self):
        next_cluster = self._not_handled_clusters.pop()
        cluster_elements = np.where(self.cluster_assignments == next_cluster)[0]
        # Pick an arbitrary element from this cluster.
        idx_to_label = self._pick_cluster_element(cluster_elements, self.cluster_centers[next_cluster])
        new_label = self.labeler.label(self.X[idx_to_label], idx_to_label)
        return dict(
            new_y=new_label * np.ones(len(cluster_elements)),
            new_y_idxs=cluster_elements,
            idxs_shown_to_labeler=np.array([idx_to_label]),
            num_propagated=len(cluster_elements) - 1,
            cluster_assignments=self.cluster_assignments,
        )

    def _cluster(self, X):
        n_clusters = len(X) // self.base_cluster_size
        clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=self.rng)
        self.X = preprocessing.StandardScaler().fit_transform(X)
        clusterer = clusterer.fit(self.X)
        return clusterer.predict(self.X), clusterer.cluster_centers_

    def _closest_element(self, cluster_elements, cluster_center):
        distances = np.sqrt(
            np.sum(
                np.square(self.X[cluster_elements] - cluster_center)
            , axis=-1)
        )
        return cluster_elements[np.argmin(distances)]
class GreedyKMeansCoordinator(BaseKMeansCoordinator):
    @classmethod
    def name(cls):
        return 'greedykmeans'

    def _pick_cluster_element(self, cluster_elements, cluster_center):
        return cluster_elements[0]

class CenterKMeansCoordinator(BaseKMeansCoordinator):
    @classmethod
    def name(cls):
        return 'centerkmeans'

    def _pick_cluster_element(self, cluster_elements, cluster_center):
        return self._closest_element(cluster_elements, cluster_center)
