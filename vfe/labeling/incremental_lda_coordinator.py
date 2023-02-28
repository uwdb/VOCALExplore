import math

import numpy as np
import pandas as pd
from sklearn import decomposition, discriminant_analysis

from .softkmeans_coordinator import (BaseClusterSoftLabelCoordinator,
                                     BaseElementChooser, BaseRanker,
                                     CenterElementChooser,
                                     CenterOutsideElementChooser,
                                     DisagreementRanker, DistanceRanker,
                                     FarthestApartElementChooser,
                                     KMeansClusterer,
                                     OracleLabelPropagator)
from .lda_coordinator import OracleAndPropagatedLabelPropagator


class BaseLabeledElementChooser:
    def idxs_and_labels(self, coordinator):
        raise NotImplementedError

class OracleLabeledElementChooser(BaseLabeledElementChooser):
    def idxs_and_labels(self, coordinator):
        labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        return labeled_idxs, coordinator._hard_labels[labeled_idxs]

class OracleAndPropagatedLabeledElementChooser(BaseLabeledElementChooser):
    def idxs_and_labels(self, coordinator):
        hard_labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        # Only get propagated labels for the elements where we don't have oracle labels.
        propagated_labeled_idxs = np.where((coordinator.y_pred != -1) & (coordinator._hard_labels == -1))[0]
        return np.concatenate([hard_labeled_idxs, propagated_labeled_idxs]), np.concatenate([coordinator._hard_labels[hard_labeled_idxs], coordinator.y_pred[propagated_labeled_idxs]])

class BaseReclusterCoordinator:
    def should_recluster(self, coordinator):
        raise NotImplementedError

class EveryIterationReclusterCoordinator(BaseReclusterCoordinator):
    def should_recluster(self, coordinator):
        oracle_labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        if len(oracle_labeled_idxs) / len(coordinator.X) < 0.01:
            return False
        return True

class IncrementalIterationReclusterCoordinator(BaseReclusterCoordinator):
    def __init__(self, initial_threshold, increment):
        self.next_threshold = initial_threshold
        self.increment = increment

    def should_recluster(self, coordinator):
        oracle_labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        if len(oracle_labeled_idxs) / len(coordinator.X) < self.next_threshold:
            return False
        self.next_threshold += self.increment
        return True

class BaseIncrementalLDACoordinator(BaseClusterSoftLabelCoordinator):
    def __init__(self, *args, n_pca_components=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_transform = decomposition.PCA(n_components=n_pca_components, random_state=self.rng).fit(self.X)
        self.lda_transform = None
        self.nclasses = len(set(self.y))
        self.n_lda_components = min(self.nclasses - 1, self.X.shape[1])
        self.n_pca_components = n_pca_components
        self._labeled_element_chooser = None
        self._recluster_coordinator = None

    @property
    def clusterer_cls(self):
        raise NotImplementedError

    @property
    def ranker_cls(self):
        raise NotImplementedError

    @property
    def labeled_element_chooser_cls(self):
        raise NotImplementedError

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    def recluster_instance(self):
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    def labeled_element_chooser(self) -> BaseLabeledElementChooser:
        if self._labeled_element_chooser is None:
            self._labeled_element_chooser = self.labeled_element_chooser_cls()
        return self._labeled_element_chooser

    def recluster_coordinator(self) -> BaseReclusterCoordinator:
        if self._recluster_coordinator is None:
            self._recluster_coordinator = self.recluster_instance()
        return self._recluster_coordinator

    def _update_dimensionality_transform(self):
        if not self.recluster_coordinator().should_recluster(self):
            self._repropagate_labels()
            return

        labeled_idxs, labeled_values = self.labeled_element_chooser().idxs_and_labels(self)
        self.lda_transform = discriminant_analysis.LinearDiscriminantAnalysis(n_components=self.n_lda_components).fit(self.X[labeled_idxs], labeled_values)

        X_lda = self.lda_transform.transform(self.X)
        X_pca = self.pca_transform.transform(self.X)
        self.transformedX = np.hstack([X_pca, X_lda]).astype(np.float32)
        self.scale_before_cluster = False

        # Re-cluster everything.
        # Expected_size has to get smaller.
        # Use a linear decay assuming maxrounds.
        maxrounds = math.ceil(1.1 * self.budget) if self.budget > -1 else len(self.y)
        expected_size = max(1, math.ceil((maxrounds - self.round) / maxrounds * self.base_cluster_size))
        cluster_assignments, cluster_centers = self._cluster(np.arange(len(self.X)), expected_size)
        self.cluster_assignments = cluster_assignments
        self.cluster_centers = cluster_centers
        self._max_cluster = np.max(self.cluster_assignments) + 1

        # Re-propagate labels based on the new membership of the labels.
        self._repropagate_labels()

        # Set _not_handled_clusters to the cluster we should prioritize so it will be labeled next.
        prioritized_cluster = self._cluster_to_split()
        self._not_handled_clusters = [prioritized_cluster]

    def _repropagate_labels(self):
        self.label_propagator().propagate_labels(self)

    def _next_cluster(self):
        if len(self._not_handled_clusters):
            return self._not_handled_clusters.pop()
        else:
            # Don't actually split the cluster. The clusters get smaller as we recluster and the rounds increase.
            return self._cluster_to_split()

    def _interaction_round(self):
        found_unlabeled_item = False
        while not found_unlabeled_item:
            next_cluster = self._next_cluster()
            cluster_elements = np.where(self.cluster_assignments == next_cluster)[0]
            idxs_to_label = self.element_chooser().pick_cluster_element(self, cluster_elements, self.cluster_centers[next_cluster])
            found_unlabeled_item = len(idxs_to_label)
        new_labels = self.labeler.label(self.X[idxs_to_label], idxs_to_label)

        assert np.all(self._hard_labels[idxs_to_label] == -1), f'Already labeled {idxs_to_label}'
        self._hard_labels[idxs_to_label] = new_labels
        # Set y_pred here so we can use the value in when propagating labels.
        self.y_pred[idxs_to_label] = new_labels

        # This propagates the labels and re-clusters if necessary.
        self._update_dimensionality_transform()

        return dict(
            new_y=new_labels,
            new_y_idxs=idxs_to_label,
            idxs_shown_to_labeler=idxs_to_label,
            num_propagated=0,
            cluster_assignments=self.cluster_assignments,
            hard_labels=self._hard_labels,
            transformedX=(self.transformedX if self.lda_transform is not None else np.zeros((len(self.X), self.n_pca_components + self.n_lda_components))),
        )

class DisagreementIncrementalLDACoordinator(BaseIncrementalLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-incremental-lda'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def labeled_element_chooser(self):
        return OracleLabeledElementChooser

    def recluster_instance(self):
        return IncrementalIterationReclusterCoordinator(0.01, 0.01)

class DisagreementOAPLDACoordinator(BaseIncrementalLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-oap-lda'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def labeled_element_chooser(self):
        return OracleAndPropagatedLabeledElementChooser

    def recluster_instance(self):
        return EveryIterationReclusterCoordinator()

class DisagreementIncrementalOAPLDACoordinator(BaseIncrementalLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-incremental-oap-lda'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def labeled_element_chooser(self):
        return OracleAndPropagatedLabeledElementChooser

    def recluster_instance(self):
        return IncrementalIterationReclusterCoordinator(0.01, 0.01)

class DisagreementIncrementalOAPOLPLDACoordinator(BaseIncrementalLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-incremental-oap-olp-lda'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def labeled_element_chooser(self):
        return OracleAndPropagatedLabeledElementChooser

    @property
    def label_propagator_cls(self):
        return OracleLabelPropagator

    def recluster_instance(self):
        return IncrementalIterationReclusterCoordinator(0.01, 0.01)
