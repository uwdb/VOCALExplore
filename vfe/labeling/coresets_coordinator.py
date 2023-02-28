import numpy as np
from scipy.spatial import distance_matrix

from .abstractcoordinator import AbstractCoordinator
from .activelearning_coordinator import LinearModelTrainer

class CoresetsALCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = next(iter(self.Xs.items()))[1]
        self.nclasses = len(set(self.y))
        self.rng = np.random.default_rng(rng)
        self._hard_labels = -1 * np.ones_like(self.y_pred)
        self._trainer = LinearModelTrainer()

    @classmethod
    def name(cls):
        return 'coresetsal'

    def _coreset(self, labeled_idxs):
        # Find the index of the element that is farthest from a labeled index.
        distances = distance_matrix(self.X, self.X[labeled_idxs])
        min_distance = np.min(distances, axis=1)
        return np.argmax(min_distance)

    def _propagate_labels(self, y_pred_logits):
        unlabeled_idxs = np.where(self._hard_labels == -1)[0]
        self.y_pred[unlabeled_idxs] = y_pred_logits[unlabeled_idxs]

    def _interaction_round(self):
        labeled_idxs = np.where(self._hard_labels != -1)[0]

        # Random sample until 1% of the dataset is labeled.
        if len(labeled_idxs) > 0.01 * len(self.y_pred):
            el_idx = self._coreset(labeled_idxs)
            assert el_idx not in labeled_idxs, f'Already labeled {el_idx}'

            y_pred_probs = self._trainer.get_y_probs(self, labeled_idxs)
            self._propagate_labels(np.argmax(y_pred_probs, axis=1))
        else:
            el_idx = self.rng.choice(len(self.X))

        new_label = self.labeler.label(self.X[el_idx], el_idx)
        self._hard_labels[el_idx] = new_label
        return dict(
            new_y=new_label,
            new_y_idxs=el_idx,
            idxs_shown_to_labeler=np.array([el_idx]),
            num_propagated=0,
            hard_labels=self._hard_labels,
        )
