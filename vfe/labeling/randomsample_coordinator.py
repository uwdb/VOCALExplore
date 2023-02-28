import numpy as np
import time

from .abstractcoordinator import AbstractCoordinator
from .activelearning_coordinator import LinearModelTrainer

class RandomSampleCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(rng)
        # The features don't matter, so just pick one.
        self.X = next(iter(self.Xs.items()))[1]
        self._hard_labels = -1 * np.ones_like(self.y_pred)
        self._trainer = LinearModelTrainer()
        self.nclasses = len(set(self.y))

    @classmethod
    def name(cls):
        return 'randomsampleal'

    def _propagate_labels(self, y_pred_logits):
        unlabeled_idxs = np.where(self._hard_labels == -1)[0]
        self.y_pred[unlabeled_idxs] = y_pred_logits[unlabeled_idxs]

    def _interaction_round(self):
        unlabeled_indices = np.where(self._hard_labels == -1)[0]
        idx_to_label = self.rng.choice(unlabeled_indices)
        new_label = self.labeler.label(self.X[idx_to_label], idx_to_label)

        self._hard_labels[idx_to_label] = new_label
        y_pred_probs = self._trainer.get_y_probs(self, np.where(self._hard_labels != -1)[0])
        self._propagate_labels(np.argmax(y_pred_probs, axis=1))

        return dict(
            new_y=new_label,
            new_y_idxs=idx_to_label,
            idxs_shown_to_labeler=np.array([idx_to_label]),
            num_propagated=0,
        )
