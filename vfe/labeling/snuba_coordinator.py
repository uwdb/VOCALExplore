import numpy as np
from sklearn import decomposition

from reef.program_synthesis.heuristic_generator import HeuristicGenerator
from reef.lancet.heuristic_generator import HeuristicGenerator as LHeuristicGenerator

from .abstractcoordinator import AbstractCoordinator

class BaseIndividualElementChooser:
    def pick_next_element(self, coordinator):
        raise NotImplementedError

class RandomIndividualElementChooser:
    def pick_next_element(self, coordinator):
        unlabeled_idxs = np.where(coordinator._hard_labels == -1)[0]
        # Specifyf size=1 so that we return a numpy array vs. an integer.
        return coordinator.rng.choice(unlabeled_idxs, size=1)

class BaseSnubaCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(rng)
        self.X = next(iter(self.Xs.items()))[1]
        self._hard_labels = -1 * np.ones_like(self.y_pred)
        self._element_chooser = None
        self.propagate_every_round = 20
        self.X_lowd = decomposition.PCA(n_components=20, random_state=rng).fit_transform(self.X)

    @classmethod
    def name(cls):
        return 'base-snuba'

    @property
    def element_chooser_cls(self):
        return RandomIndividualElementChooser

    @property
    def heuristic_generator_cls(self):
        return HeuristicGenerator

    def element_chooser(self) -> BaseIndividualElementChooser:
        if self._element_chooser is None:
            self._element_chooser = self.element_chooser_cls()
        return self._element_chooser

    def _propagate_labels(self):
        # Based on flow in https://github.com/HazyResearch/reef/blob/master/%5B1%5D%20generate_reef_labels.ipynb.
        train_primitive_matrix = self.X_lowd[self._hard_labels == -1]
        val_primitive_matrix = self.X_lowd[self._hard_labels != -1]
        val_ground = self._hard_labels[self._hard_labels != -1]
        hg = self.heuristic_generator_cls(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground=None, b=0.5)
        idx = None

        for i in range(20):
            keep = 3 if i == 0 else 1
            hg.run_synthesizer(max_cardinality=1, idx=None, keep=keep, model='lr')
            hg.run_verifier()

            hg.find_feedback()
            idx = hg.feedback_idx

            if len(idx) == 0:
                break

        # Value is in (0, 1) -> probability of being class 1.
        train_marginals = hg.vf.train_marginals

        if len(train_marginals.shape) > 1:
            y_pred_logits = np.argmax(train_marginals, axis=1)
            self.y_pred[self._hard_labels == -1] = np.where(
                np.isclose(np.min(train_marginals, axis=1), np.max(train_marginals, axis=1), atol=1e-2),
                -1,
                y_pred_logits
            )
        else:
            # If the class is 50/50, don't assign a label. Else, assign it 0 or 1 depending on which side of 0.5 the probability is on.
            self.y_pred[self._hard_labels == -1] = np.where(
                train_marginals == 0.5,
                    -1,
                    np.where(
                        train_marginals > 0.5,
                            1,
                            0
                    )
                )

    def _interaction_round(self):
        idxs_to_label = self.element_chooser().pick_next_element(self)
        new_label = self.labeler.label(self.X[idxs_to_label], idxs_to_label)
        self._hard_labels[idxs_to_label] = new_label

        # Use Snuba to generate y_pred.
        if self.round and self.round % self.propagate_every_round == 0:
            self._propagate_labels()

        return dict(
            new_y=new_label,
            new_y_idxs=idxs_to_label,
            idxs_shown_to_labeler=idxs_to_label,
            num_propagated=0,
            hard_labels=self._hard_labels,
        )

class LancetSnubaCoordinator(BaseSnubaCoordinator):
    @classmethod
    def name(cls):
        return 'base-lancet-snuba'

    @property
    def heuristic_generator_cls(self):
        return LHeuristicGenerator
