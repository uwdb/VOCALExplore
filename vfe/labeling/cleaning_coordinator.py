import math
import numpy as np
from scipy import stats
from sklearn import decomposition, discriminant_analysis
import pandas as pd

from .softkmeans_coordinator import BaseClusterSoftLabelCoordinator, KMeansClusterer, BaseRanker, DisagreementRanker, BaseLabelPropagator
from .model_utils import train_model, predict_model_logits
from .lda_coordinator import OracleAndPropagatedLabelPropagator, OracleAndPropagatedLabelTrainerPropagator
from .incremental_lda_coordinator import OracleLabeledElementChooser

class BaseLabelPropagator:
    def propagate_labels(self, coordinator):
        raise NotImplementedError

class BaseLabelFixer:
    def fix_labels(self, coordinator):
        raise NotImplementedError

class Classifier:
    def predict_proba(self, X):
        raise NotImplementedError

class ClassifierClusterLabelFixer(BaseLabelFixer):
    def __init__(self, ask_oracle_prob, conf_threshold=0.99, rng=None):
        self.ask_oracle_prob = ask_oracle_prob
        self.conf_threshold = conf_threshold
        self.rng = np.random.default_rng(rng)

    def fix_labels(self, coordinator):
        if len(coordinator._not_handled_clusters):
            # Don't re-set the list because we're working through previously scheduled clusters to ask the oracle about.
            return

        clf = coordinator.classifier
        predictions = clf.predict_proba(coordinator.transformedX)

        cluster_assignments = coordinator.cluster_assignments
        labels = coordinator.y_pred
        clusters_for_labeler = []
        for cluster_id in range(coordinator._max_cluster):
            agg_predictions = np.mean(predictions[cluster_assignments == cluster_id], axis=0)
            top_prediction = np.max(agg_predictions)
            if top_prediction >= self.conf_threshold:
                classifier_predicted_class = np.argmax(agg_predictions)
                # mode returns (array of modal values, array of counts for each mode).
                # Since we're taking the mode of a one-dimensional array, there is just a single modal value for us to get.
                current_predicted_class = stats.mode(labels[cluster_assignments == cluster_id])[0][0]
                # Get a user label if the classifier disagrees with the current predicted class, or if they agree but nothing has been labeled yet, so we don't know if the classifier is wrong.
                if classifier_predicted_class != current_predicted_class \
                        or np.max(coordinator._hard_labels[cluster_assignments == cluster_id]) == -1:
                    if self.rng.random() < self.ask_oracle_prob:
                        clusters_for_labeler.append(cluster_id)
                    else:
                        coordinator.y_pred[np.where((cluster_assignments == cluster_id) & (coordinator._hard_labels == -1))] = classifier_predicted_class

        if len(clusters_for_labeler):
            coordinator._not_handled_clusters = clusters_for_labeler

class BaseReclusterCoordinator:
    def should_recluster(self, coordinator):
        raise NotImplementedError

class ReclusterIfNoPendingToLabel(BaseReclusterCoordinator):
    def __init__(self):
        self.performed_first_recluster = False

    def should_recluster(self, coordinator):
        # Don't recluster if we're in the early phase and don't have enough labels yet, or if there are pending clusters to label.
        # The first time we hit enough labels to recluster, clear the pending clusters to label so that we will recluster.
        oracle_labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        if len(oracle_labeled_idxs) / len(coordinator.X) < 0.01:
            return False
        if not self.performed_first_recluster:
            coordinator._not_handled_clusters = []
            self.performed_first_recluster = True
        return len(coordinator._not_handled_clusters) == 0

class BaseLabeledElementChooser:
    def idxs_and_labels(self, coordinator):
        raise NotImplementedError

class OracleThenPropagatedLabeledElementChooser(BaseLabeledElementChooser):
    def idxs_and_labels(self, coordinator):
        hard_labeled_idxs = np.where(coordinator._hard_labels != -1)[0]
        hard_labeled_labels = coordinator._hard_labels[hard_labeled_idxs]
        if len(hard_labeled_idxs) / len(coordinator.X) < 0.02:
            return hard_labeled_idxs, hard_labeled_labels

        # Only get propagated labels for the elements where we don't have oracle labels.
        propagated_labeled_idxs = np.where((coordinator.y_pred != -1) & (coordinator._hard_labels == -1))[0]
        return np.concatenate([hard_labeled_idxs, propagated_labeled_idxs]), np.concatenate([hard_labeled_labels, coordinator.y_pred[propagated_labeled_idxs]])

class BaseTrainer:
    def get_classifier(self, coordinator) -> Classifier:
        raise NotImplementedError

class QDATrainer(BaseTrainer):
    def get_classifier(self, coordinator) -> Classifier:
        clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
        train_idxs = np.where(coordinator._hard_labels != -1)[0]
        clf.fit(coordinator.transformedX[train_idxs], coordinator._hard_labels[train_idxs])
        return clf

class PytorchClassifier(Classifier):
    def __init__(self, trainer, model):
        self.trainer = trainer
        self.model = model

    def predict_proba(self, X):
        return predict_model_logits(self.trainer, self.model, X).numpy()

class LinearModelTrainer(BaseTrainer):
    def get_classifier(self, coordinator):
        train_idxs = np.where(coordinator._hard_labels != -1)[0]
        trainer, model = train_model(coordinator.transformedX[train_idxs], coordinator._hard_labels[train_idxs].astype(int), coordinator.nclasses)
        return PytorchClassifier(trainer, model)

class ClassifierALRanker(BaseRanker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup = DisagreementRanker()

    def cluster_to_split(self, coordinator):
        # Return the cluster with the lowest average prediction probability. This should prioritize clusters with uncertain predictions.
        if coordinator.classifier is not None:
            clusters_with_unlabeled, elements_in_unlabeled_clusters = self.clusters_with_unlabeled_elements(coordinator)
            y_pred = coordinator.classifier.predict_proba(coordinator.transformedX)
            predicted_class_prob = np.max(y_pred, axis=1)
            average_predicted_probability = pd.Series(predicted_class_prob).groupby(coordinator.cluster_assignments).mean()
            return average_predicted_probability.filter(items=clusters_with_unlabeled, axis=0).idxmin()
        else:
            return self.backup.cluster_to_split(coordinator)


class BaseCleaningLDACoordinator(BaseClusterSoftLabelCoordinator):
    def __init__(self, *args, n_pca_components=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_transform = decomposition.PCA(n_components=n_pca_components, random_state=self.rng).fit(self.X)
        self.lda_transform = None
        self.nclasses = len(set(self.y))
        self.n_lda_components = min(self.nclasses - 1, self.X.shape[1])
        self.n_pca_components = n_pca_components
        self.classifier = None

        self._labeled_element_chooser = None
        self._recluster_coordinator = None
        self._label_propagator = None
        self._label_fixer = None
        self._trainer = None

        self.cleaning_threshold = 0.02

    @classmethod
    def name(cls):
        raise NotImplementedError

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
    def trainer_cls(self):
        raise NotImplementedError

    def label_fixer_instance(self):
        raise NotImplementedError

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    def recluster_instance(self):
        raise NotImplementedError

    def labeled_element_chooser(self) -> BaseLabeledElementChooser:
        if self._labeled_element_chooser is None:
            self._labeled_element_chooser = self.labeled_element_chooser_cls()
        return self._labeled_element_chooser

    def recluster_coordinator(self) -> BaseReclusterCoordinator:
        if self._recluster_coordinator is None:
            self._recluster_coordinator = self.recluster_instance()
        return self._recluster_coordinator

    def label_propagator(self) -> BaseLabelPropagator:
        if self._label_propagator is None:
            self._label_propagator = self.label_propagator_cls()
        return self._label_propagator

    def trainer(self) -> BaseTrainer:
        if self._trainer is None:
            self._trainer = self.trainer_cls()
        return self._trainer

    def label_fixer(self) -> BaseLabelFixer:
        if self._label_fixer is None:
            self._label_fixer = self.label_fixer_instance()
        return self._label_fixer

    def _propagate_labels(self):
        self.label_propagator().propagate_labels(self)

    def _fix_labels(self):
        self.label_fixer().fix_labels(self)

    def _update_classifier(self):
        self.classifier = self.trainer().get_classifier(self)

    def _update_dimensionality_transform(self):
        if not self.recluster_coordinator().should_recluster(self):
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
        maxrounds = math.ceil(1.1 * (self.budget if self.budget > -1 else len(self.y)))
        expected_size = max(1, math.ceil((maxrounds - self.round) / maxrounds * self.base_cluster_size))
        cluster_assignments, cluster_centers = self._cluster(np.arange(len(self.X)), expected_size)
        self.cluster_assignments = cluster_assignments
        self.cluster_centers = cluster_centers
        self._max_cluster = np.max(self.cluster_assignments) + 1

        # If cluster memberships have changed, ropagate labels to new neighbors.
        self._propagate_labels()

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
        # Set y_pred here so we can use the value when propagating labels.
        self.y_pred[idxs_to_label] = new_labels

        # Propagate the label.
        self._propagate_labels()

        # Update the dimensionality transform.
        self._update_dimensionality_transform()

        # Once we have a sufficient amount of labeled data, try to improve upon the propagated labels.
        if len(self._hard_labels[self._hard_labels != -1]) / len(self.X) > self.cleaning_threshold:
            # Update the cleaning classifier.
            self._update_classifier()

            # Fix labels.
            self._fix_labels()

        return dict(
            new_y=new_labels,
            new_y_idxs=idxs_to_label,
            idxs_shown_to_labeler=idxs_to_label,
            num_propagated=0,
            cluster_assignments=self.cluster_assignments,
            hard_labels=self._hard_labels,
            transformedX=(self.transformedX if self.lda_transform is not None else np.zeros((len(self.X), self.n_pca_components + self.n_lda_components))),
        )


class CleanWithProbLDACoordinator(BaseCleaningLDACoordinator):
    @property
    def ask_oracle_prob(self) -> float:
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return ClassifierALRanker

    @property
    def labeled_element_chooser(self):
        return OracleThenPropagatedLabeledElementChooser

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    @property
    def trainer_cls(self):
        return QDATrainer

    def label_fixer_instance(self):
        return ClassifierClusterLabelFixer(ask_oracle_prob=self.ask_oracle_prob)

    def recluster_instance(self):
        return ReclusterIfNoPendingToLabel()

class BasicCleaningLDACoordinator(CleanWithProbLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 0

    @classmethod
    def name(cls):
        return 'basiccleaning-lda'

class CheckOracleCleaningLDACoordinator(CleanWithProbLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 1

    @classmethod
    def name(cls):
        return 'checkoracle-cleaning-lda'

class CheckOracle50CleaningLDACoordinator(CleanWithProbLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 0.5

    @classmethod
    def name(cls):
        return 'checkoracle50-cleaning-lda'

class CleanWithProbLMLDACoordinator(BaseCleaningLDACoordinator):
    @property
    def ask_oracle_prob(self) -> float:
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return ClassifierALRanker

    @property
    def labeled_element_chooser(self):
        return OracleThenPropagatedLabeledElementChooser

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    @property
    def trainer_cls(self):
        return LinearModelTrainer

    def label_fixer_instance(self):
        return ClassifierClusterLabelFixer(ask_oracle_prob=self.ask_oracle_prob)

    def recluster_instance(self):
        return ReclusterIfNoPendingToLabel()

class BasicCleaningLMLDACoordinator(CleanWithProbLMLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 0

    @classmethod
    def name(cls):
        return 'basiccleaning-lm-lda'


class CheckOracleCleaningLMLDACoordinator(CleanWithProbLMLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 1

    @classmethod
    def name(cls):
        return 'checkoracle-cleaning-lm-lda'

class CheckOracle50CleaningLMLDACoordinator(CleanWithProbLMLDACoordinator):
    @property
    def ask_oracle_prob(self):
        return 0.5

    @classmethod
    def name(cls):
        return 'checkoracle50-cleaning-lm-lda'

class CleanWithProbLDAWithLabeledCoordinator(BaseCleaningLDACoordinator):
    @property
    def ask_oracle_prob(self) -> float:
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    def clusterer_cls(self):
        return KMeansClusterer

    @property
    def ranker_cls(self):
        return ClassifierALRanker

    @property
    def labeled_element_chooser(self):
        return OracleLabeledElementChooser

    @property
    def label_propagator_cls(self):
        return OracleAndPropagatedLabelPropagator

    @property
    def trainer_cls(self):
        return QDATrainer

    def label_fixer_instance(self):
        return ClassifierClusterLabelFixer(ask_oracle_prob=self.ask_oracle_prob)

    def recluster_instance(self):
        return ReclusterIfNoPendingToLabel()

class CheckOracle50CleaningLDAWithLabeledCoordinator(CleanWithProbLDAWithLabeledCoordinator):
    @property
    def ask_oracle_prob(self):
        return 0.5

    @classmethod
    def name(cls):
        return 'checkoracle50-cleaning-lda-with-labeled'
