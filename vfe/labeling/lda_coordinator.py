import math
import numpy as np
import pandas as pd
from sklearn import decomposition, discriminant_analysis

from .softkmeans_coordinator import KMeansClusterer, DistanceRanker, BaseClusterSoftLabelCoordinator, DisagreementRanker, CenterElementChooser, CenterOutsideElementChooser, FarthestApartElementChooser, BaseElementChooser, BaseRanker, BaseLabelPropagator, BaseModelLabelPropagator

class LDAElementChooser(BaseElementChooser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup = CenterElementChooser()

    def _pick_cluster_element(self, coordinator, cluster_elements, cluster_center):
        if coordinator.lda_transform is not None:
            probs = coordinator.lda_transform.predict_proba(coordinator.X[cluster_elements])
            max_prob = np.max(probs, axis=1)
            return np.array([cluster_elements[np.argmin(max_prob)]])
        else:
            return self.backup._pick_cluster_element(coordinator, cluster_elements, cluster_center)

class LDAIncorrectRanker(BaseRanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup = DisagreementRanker()

    def cluster_to_split(self, coordinator):
        if coordinator.lda_transform is not None:
            clusters_with_unlabeled, elements_in_unlabeled_clusters = self.clusters_with_unlabeled_elements(coordinator)
            labeled_idxs = np.where(coordinator._hard_labels != -1)
            predictions = coordinator.lda_transform.predict(coordinator.X[labeled_idxs])
            incorrect_predictions = predictions != coordinator.y[labeled_idxs]
            # Pick the cluster with the most incorrect predictions.
            num_incorrect_predictions = pd.Series(incorrect_predictions).groupby(coordinator.cluster_assignments[labeled_idxs]).sum()
            num_incorrect_predictions = num_incorrect_predictions.filter(items=clusters_with_unlabeled, axis=0)
            if len(num_incorrect_predictions):
                return num_incorrect_predictions.idxmax()
            else:
                return self.backup.cluster_to_split(coordinator)
        else:
            return self.backup.cluster_to_split(coordinator)

class OracleAndPropagatedLabelPropagator(BaseLabelPropagator):
    def propagate_labels(self, coordinator):
        # Only get the mode from the elements that are actually labeled.
        # This will not affect elements in clusters that contain only unlabeled items, because those clusters won't show up in cluster_modes.
        labeled_idxs = np.where(coordinator._hard_labels != -1)
        # Set dtype=object to work around https://github.com/pandas-dev/pandas/issues/38534.
        cluster_modes_hardlabels = pd.Series(coordinator._hard_labels[labeled_idxs], dtype=object).groupby(coordinator.cluster_assignments[labeled_idxs]).agg(pd.Series.mode)
        cluster_modes_softlabels = pd.Series(coordinator.y_pred[coordinator.y_pred != -1], dtype=object).groupby(coordinator.cluster_assignments[coordinator.y_pred != -1]).agg(pd.Series.mode)
        for i in np.arange(len(coordinator.y_pred)):
            if coordinator._hard_labels[i] != -1:
                # Don't modify the label of something the user has labeled.
                continue
            cluster_assignment = coordinator.cluster_assignments[i]
            if cluster_assignment not in cluster_modes_hardlabels:
                # If there are no hard labels in the current cluster, use the soft labels.
                if cluster_assignment not in cluster_modes_softlabels:
                    continue
                element_mode = cluster_modes_softlabels[cluster_assignment]
            else:
                element_mode = cluster_modes_hardlabels[cluster_assignment]
            try:
                # If there are multiple modes, cluster_modes will be a list.
                # Pick the last one so that if there are equal numbers of labeled and unlabeled items,
                # the labeled value will come after -1 in sorted order.
                coordinator.y_pred[i] = coordinator.rng.choice(element_mode)
            except:
                # If there is a single mode, trying to index a scalar will throw an exception.
                coordinator.y_pred[i] = element_mode

class OracleAndPropagatedLabelTrainerPropagator(BaseModelLabelPropagator):
    def __init__(self):
        super().__init__(OracleAndPropagatedLabelPropagator())

class BaseLDACoordinator(BaseClusterSoftLabelCoordinator):
    def __init__(self, *args, n_pca_components=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_transform = decomposition.PCA(n_components=n_pca_components, random_state=self.rng).fit(self.X)
        self.lda_transform = None
        self.nclasses = len(set(self.y))
        self.n_lda_components = min(self.nclasses - 1, self.X.shape[1])
        self.n_pca_components = n_pca_components

    @property
    def clusterer_cls(self):
        raise NotImplementedError

    @property
    def ranker_cls(self):
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    def _update_dimensionality_transform(self):
        labeled_idxs = np.where(self._hard_labels != -1)[0]

        if len(labeled_idxs) / len(self.X) < 0.01:
            # If we haven't labeled even 1% of our data, keep labeling for a bit more before trying to learn anything.
            self._repropagate_labels()
            return

        self.lda_transform = discriminant_analysis.LinearDiscriminantAnalysis(n_components=self.n_lda_components).fit(self.X[labeled_idxs], self.y[labeled_idxs])

        # Check if the transformation is good by training a model.
        # Actually, for now just assume the transformation is good.
        X_lda = self.lda_transform.transform(self.X)
        X_pca = self.pca_transform.transform(self.X)
        self.transformedX = np.hstack([X_pca, X_lda]).astype(np.float32)
        self.scale_before_cluster = False

        # Re-cluster everything.
        # Expected_size has to get smaller.
        # Use a linear decay assuming maxrounds.
        maxrounds = 1000
        expected_size = max(1, math.ceil((maxrounds - self.round) / maxrounds * self.base_cluster_size))
        cluster_assignments, cluster_centers = self._cluster(np.arange(len(self.X)), expected_size)
        self.cluster_assignments = cluster_assignments
        self.cluster_centers = cluster_centers
        self._max_cluster = np.max(self.cluster_assignments) + 1

        # Re-propagate labels based on the new membership of the hard labels.
        # If there are multiple different hard labels within a cluster, pick the one that shows up more.
        self._repropagate_labels()

        # Set _not_handled_clusters to the cluster we should prioritize so it will be labeled next.
        prioritized_cluster = self._cluster_to_split()
        self._not_handled_clusters = [prioritized_cluster]

    def _repropagate_labels(self):
        self.label_propagator().propagate_labels(self)

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

        # This propagates the labels.
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

class DisagreementKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-lda-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

class DistanceKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'distance-lda-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DistanceRanker

class DistanceKMeansLDATrainerCoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'distance-lda-trainer-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DistanceRanker

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelTrainerPropagator

class DisagreementMultiPointKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-lda-multipoint-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def element_chooser_cls(self):
        return CenterOutsideElementChooser

class DisagreementFarthestKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-lda-farthest-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def element_chooser_cls(self):
        return FarthestApartElementChooser

class DisagreementLDAALKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'disagreement-lda-ldaal-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return DisagreementRanker

    @property
    def element_chooser_cls(self):
        return LDAElementChooser

class IncorrectLDAALKMeansLDACoordinator(BaseLDACoordinator):
    @classmethod
    def name(cls):
        return 'incorrect-lda-ldaal-kmeans'

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return LDAIncorrectRanker

    @property
    def element_chooser_cls(self):
        return LDAElementChooser
