import math
import numpy as np
import pandas as pd
import scipy
from sklearn import cluster, preprocessing, mixture
import torch

from .abstractcoordinator import AbstractCoordinator
from .model_utils import train_model, predict_model_logits

class BaseClusterer:
    def cluster(self, X, expected_size=20, rng=None):
        # Return cluster_assignments, cluster_centers.
        raise NotImplementedError

class KMeansClusterer(BaseClusterer):
    def cluster(self, X, expected_size=20, rng=None):
        n_clusters = math.ceil(len(X) / expected_size)
        assert n_clusters > 1, f'math.ceil({len(X)} / {expected_size}) <= 1'
        clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=rng)
        clusterer = clusterer.fit(X)
        return clusterer.predict(X), clusterer.cluster_centers_

class GMMClusterer(BaseClusterer):
    def cluster(self, X, expected_size=20, rng=None):
        n_components = math.ceil(len(X) / expected_size)
        assert n_components > 1, f'math.ceil({len(X)} / {expected_size}) <= 1'
        clusterer = mixture.GaussianMixture(n_components=n_components, random_state=rng)
        clusterer.fit(X)
        return clusterer.predict(X), clusterer.means_

class BaseRanker:
    def cluster_to_split(self, coordinator):
        raise NotImplementedError

    def clusters_with_unlabeled_elements(self, coordinator):
        min_label = pd.Series(coordinator._hard_labels).groupby(coordinator.cluster_assignments).min()
        clusters_with_unlabeled = min_label[min_label == -1].index
        elements_in_unlabeled_clusters = np.where(np.isin(coordinator.cluster_assignments, clusters_with_unlabeled))
        return clusters_with_unlabeled, elements_in_unlabeled_clusters

class SizeRanker(BaseRanker):
    def cluster_to_split(self, coordinator):
        clusters_with_unlabeled, elements_in_unlabeled_clusters = self.clusters_with_unlabeled_elements(coordinator)
        return pd.Series(coordinator.cluster_assignments[elements_in_unlabeled_clusters]).groupby(coordinator.cluster_assignments[elements_in_unlabeled_clusters]).count().idxmax()

class DistanceRanker(BaseRanker):
    def cluster_to_split(self, coordinator):
        clusters_with_unlabeled, elements_in_unlabeled_clusters = self.clusters_with_unlabeled_elements(coordinator)
        # For each cluster, compute the distance from each element to the center.
        distances = np.sqrt(
            np.sum(
                np.square(coordinator.transformedX - coordinator.cluster_centers[coordinator.cluster_assignments])
            , axis=-1)
        )
        percluster_distances = pd.Series(distances[elements_in_unlabeled_clusters]).groupby(coordinator.cluster_assignments[elements_in_unlabeled_clusters]).quantile(0.75)

        # Pick the cluster with the largest 75th percentile distance.
        return percluster_distances.idxmax()

class DisagreementRanker(BaseRanker):
    def cluster_to_split(self, coordinator):
        label_groups = pd.Series(coordinator._hard_labels).groupby(coordinator.cluster_assignments)
        max_label = label_groups.max()
        min_label = label_groups.min()
        num_labels = label_groups.nunique()
        sizes = label_groups.count()
        # If there are any clusters without labels, pick the biggest one.
        clusters_without_labels = max_label[max_label == -1]
        if len(clusters_without_labels):
            return sizes[clusters_without_labels.index].idxmax()
        else:
            # Otherwise, if all clusters have at least one labeled item, pick the one with the most conflicting label types.
            # Filter out clusters where everything is labeled.
            return num_labels[min_label == -1].idxmax()

class BaseElementChooser:
    def pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        # Filter out elements that have already been labeled by the user.
        hard_labels = coordinator._hard_labels[cluster_elements]
        cluster_elements = cluster_elements[hard_labels == -1]
        if len(cluster_elements) == 0:
            # If everything in the cluster is labeled, there's no nothing else to show to the user.
            return []
        if len(cluster_elements) == 1:
            # There is just one choice of what to show to the user that's new.
            return cluster_elements
        return self._pick_cluster_element(coordinator, cluster_elements, cluster_center)

    def _pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        raise NotImplementedError

    def get_distances(self, coordinator, cluster_elements, cluster_center):
        return np.sqrt(
            np.sum(
                np.square(coordinator.transformedX[cluster_elements] - cluster_center)
            , axis=-1)
        )

class CenterElementChooser(BaseElementChooser):
    def _pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        distances = self.get_distances(coordinator, cluster_elements, cluster_center)
        return np.array([cluster_elements[np.argmin(distances)]])

class CenterOutsideElementChooser(BaseElementChooser):
    def _pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        distances = self.get_distances(coordinator, cluster_elements, cluster_center)
        return cluster_elements[[np.argmin(distances), np.argmax(distances)]]

class FarthestApartElementChooser(BaseElementChooser):
    def _pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        cluster_points = coordinator.transformedX[cluster_elements]
        distances = scipy.spatial.distance_matrix(cluster_points, cluster_points)
        # furthest_idx is a 2 dimensional tuple indicating which combination of points has the largest distance.
        furthest_idx = np.unravel_index(np.argmax(distances), distances.shape)
        return cluster_elements[[furthest_idx[0], furthest_idx[1]]]

class UncertaintyRanker(BaseRanker):
    def _train_model(self, coordinator, labeled_idxs):
        trainer, model = train_model(coordinator.X[labeled_idxs], coordinator.y[labeled_idxs], coordinator.nclasses)
        return predict_model_logits(trainer, model, coordinator.X, logits=False)

    def cluster_to_split(self, coordinator):
        # Train a model on the labeled data and get the confidences of each point.
        y_pred_probs = self._train_model(coordinator, np.where(coordinator._hard_labels != -1)[0])
        # Get the probability of the most likely class.
        y_pred_probs_max = torch.max(y_pred_probs, dim=1)[0]

        # Filter out clusters where all of the elements are labeled.
        # The minimum will be -1 if there is an unlabeled element in the cluster.
        clusters_with_unlabeled, elements_in_unlabeled_clusters = self.clusters_with_unlabeled_elements(coordinator)

        # Pick the cluster with the lowest confidence.
        return pd.Series(y_pred_probs_max[elements_in_unlabeled_clusters]).groupby(coordinator.cluster_assignments[elements_in_unlabeled_clusters]).median().idxmin()

class BaseLabelPropagator:
    def propagate_labels(self, coordinator):
        # Update coordinator.y_pred
        raise NotImplementedError

class OracleLabelPropagator(BaseLabelPropagator):
    def propagate_labels(self, coordinator):
        # Only get the mode from the elements that are actually labeled.
        # This will not affect elements in clusters that contain only unlabeled items, because those clusters won't show up in cluster_modes.
        labeled_idxs = np.where(coordinator._hard_labels != -1)
        # Set dtype=object to work around https://github.com/pandas-dev/pandas/issues/38534.
        cluster_modes_hardlabels = pd.Series(coordinator._hard_labels[labeled_idxs], dtype=object).groupby(coordinator.cluster_assignments[labeled_idxs]).agg(pd.Series.mode)
        for i in np.arange(len(coordinator.y_pred)):
            if coordinator._hard_labels[i] != -1:
                # Don't modify the label of something the user has labeled.
                continue
            cluster_assignment = coordinator.cluster_assignments[i]
            if cluster_assignment not in cluster_modes_hardlabels:
                continue
            else:
                element_mode = cluster_modes_hardlabels[cluster_assignment]
            try:
                # If there are multiple modes, cluster_modes will be a list.
                # If there are multiple modes, pick one randomly to avoid biasing the labels towards any particular class.
                coordinator.y_pred[i] = coordinator.rng.choice(element_mode)
            except:
                # If there is a single mode, trying to index a scalar will throw an exception.
                coordinator.y_pred[i] = element_mode

class BaseModelLabelPropagator(BaseLabelPropagator):
    def __init__(self, oracle_propagator):
        self.oracle_propagator = oracle_propagator

    def propagate_labels(self, coordinator):
        self.oracle_propagator.propagate_labels(coordinator)
        labeled_idxs = np.where(coordinator.y_pred != -1)
        trainer, model = train_model(coordinator.X[labeled_idxs], coordinator.y_pred[labeled_idxs].astype(int), coordinator.nclasses)

        y_pred = predict_model_logits(trainer, model, coordinator.X, logits=True)
        not_oracle_labeled_idxs = np.where(coordinator._hard_labels == -1)
        coordinator.y_pred[not_oracle_labeled_idxs] = y_pred[not_oracle_labeled_idxs]

class OracleAndModelLabelPropagator(BaseModelLabelPropagator):
    def __init__(self):
        super().__init__(OracleLabelPropagator())

class BaseClusterSoftLabelCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, scale_before_cluster=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState(rng)
        self.scale_before_cluster = scale_before_cluster
        self._clusterer = None
        self._ranker = None
        self._element_chooser = None
        self._label_propagator = None

        # For now assume we have a single feature.
        self.X = next(iter(self.Xs.items()))[1]
        self.transformedX = self.X.copy()
        self.cluster_assignments, self.cluster_centers = self._cluster(np.arange(len(self.X)))
        self._not_handled_clusters = list(np.unique(self.cluster_assignments))
        self._hard_labels = -1 * np.ones_like(self.y_pred)
        self._max_cluster = np.max(self.cluster_assignments) + 1

    @property
    def clusterer_cls(self):
        raise NotImplementedError

    @property
    def ranker_cls(self):
        raise NotImplementedError

    @property
    def element_chooser_cls(self):
        return CenterElementChooser

    @property
    def label_propagator_cls(self):
        return OracleLabelPropagator

    def clusterer(self) -> BaseClusterer:
        if self._clusterer is None:
            self._clusterer = self.clusterer_cls()
        return self._clusterer

    def ranker(self) -> BaseRanker:
        if self._ranker is None:
            self._ranker = self.ranker_cls()
        return self._ranker

    def element_chooser(self) -> BaseElementChooser:
        if self._element_chooser is None:
            self._element_chooser = self.element_chooser_cls()
        return self._element_chooser

    def label_propagator(self) -> BaseLabelPropagator:
        if self._label_propagator is None:
            self._label_propagator = self.label_propagator_cls()
        return self._label_propagator

    @classmethod
    def name(cls):
        raise NotImplementedError

    def done_labeling(self):
        if self.budget == -1:
            return len(np.where(self._hard_labels == -1)[0]) == 0
        else:
            return self.round >= self.budget

    def _interaction_round(self):
        found_unlabeled_item = False
        while not found_unlabeled_item:
            next_cluster = self._next_cluster()
            cluster_elements = np.where(self.cluster_assignments == next_cluster)[0]
            idxs_to_label = self.element_chooser().pick_cluster_element(self, cluster_elements, self.cluster_centers[next_cluster])
            found_unlabeled_item = len(idxs_to_label)
        new_label = self.labeler.label(self.X[idxs_to_label], idxs_to_label)

        # Remove already-labeled elements from what the new label will be propagated to.
        hard_labels = self._hard_labels[cluster_elements]
        cluster_elements = cluster_elements[hard_labels == -1]

        assert np.all(self._hard_labels[idxs_to_label] == -1), f'Already labeled {idxs_to_label}'
        self._hard_labels[idxs_to_label] = new_label
        self.label_propagator().propagate_labels(self)
        return dict(
            new_y=new_label,
            new_y_idxs=idxs_to_label,
            idxs_shown_to_labeler=idxs_to_label,
            num_propagated=0,
            cluster_assignments=self.cluster_assignments,
            hard_labels=self._hard_labels,
        )

    def _cluster(self, idxs, expected_size=None):
        if expected_size is None:
            expected_size = self.base_cluster_size

        if len(idxs) == 1:
            return np.array([0]), self.transformedX[idxs]

        if self.scale_before_cluster:
            self.transformedX[idxs] = preprocessing.StandardScaler().fit_transform(self.X[idxs])
        cluster_assignments, cluster_centers = self.clusterer().cluster(self.transformedX[idxs], expected_size=expected_size, rng=self.rng)
        # If there is just a single cluster found, randomly split the points into clusters.
        if len(set(cluster_assignments)) == 1:
            shuffled_idxs = np.arange(len(idxs))
            self.rng.shuffle(shuffled_idxs)
            for cluster_id, start in enumerate(range(0, len(idxs)+expected_size, expected_size)):
                if start >= len(idxs):
                    break
                stop = min(len(idxs), start + expected_size)
                idxs_in_cluster = idxs[shuffled_idxs[start:stop]]
                cluster_assignments[idxs_in_cluster] = cluster_id
                cluster_centers[cluster_id] = np.mean(self.transformedX[idxs_in_cluster], axis=0)
        return cluster_assignments, cluster_centers

    def _cluster_to_split(self):
        return self.ranker().cluster_to_split(self)

    def _next_cluster(self):
        if len(self._not_handled_clusters):
            return self._not_handled_clusters.pop()
        else:
            # Pick an existing cluster and re-cluster it.
            cluster_to_split = self._cluster_to_split()
            idxs = np.where(self.cluster_assignments == cluster_to_split)[0]
            assert len(idxs), f'No elements are assigned to cluster {cluster_to_split}'
            # Split the cluster in two.
            # Use ceil to avoid potentially dividing by zero in _cluster.
            expected_size = math.ceil(len(idxs) / 2)
            cluster_assignments, cluster_centers = self._cluster(idxs, expected_size)
            # Update the cluster IDs so they don't overlap with previous ones.
            cluster_assignments = cluster_assignments + self._max_cluster
            self.cluster_assignments[idxs] = cluster_assignments
            self._max_cluster = np.max(self.cluster_assignments) + 1
            self.cluster_centers = np.vstack([self.cluster_centers, cluster_centers])
            self._not_handled_clusters = list(np.unique(cluster_assignments))
            # Return an item from _not_handled_clusters.
            return self._not_handled_clusters.pop()

class SizeKMeansSoftLabelCoordinator(BaseClusterSoftLabelCoordinator):
    @classmethod
    def name(cls):
        return 'size-softlabel-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return SizeRanker

class DistanceKMeansSoftLabelCoordinator(BaseClusterSoftLabelCoordinator):
    @classmethod
    def name(cls):
        return 'distance-softlabel-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DistanceRanker

class DistanceKMeansSoftLabelTrainingCoordinator(BaseClusterSoftLabelCoordinator):
    @classmethod
    def name(cls):
        return 'distance-softlabel-training-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DistanceRanker

    @property
    def label_propagator_cls(self):
        return OracleAndModelLabelPropagator

class DistanceGMMSoftLabelCoordinator(BaseClusterSoftLabelCoordinator):
    @classmethod
    def name(cls):
        return 'distance-softlabel-gmm'

    @property
    def clusterer_cls(self):
        return GMMClusterer

    @property
    def ranker_cls(self):
        return DistanceRanker

# Disagreement metric? (hard labels within a cluster disagree)

# Active learning metric: average confidence for points a model is trained on.
class UncertaintyKMeansSoftLabelCoordinator(BaseClusterSoftLabelCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nclasses = len(set(self.y))

    @classmethod
    def name(cls):
        return 'uncertainty-softlabel-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return UncertaintyRanker

class UncertaintyGMMSoftLabelCoordinator(BaseClusterSoftLabelCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nclasses = len(set(self.y))

    @classmethod
    def name(cls):
        return 'uncertainty-softlabel-gmm'

    @property
    def clusterer_cls(self):
        return GMMClusterer

    @property
    def ranker_cls(self):
        return UncertaintyRanker
