import numpy as np
import pytorch_lightning as pl
from sklearn import discriminant_analysis
import torch

from .abstractcoordinator import AbstractCoordinator
from vfe.models import linearmodel

class BaseTrainer:
    def get_y_probs(self, coordinator, labeled_idxs):
        raise NotImplementedError

class LinearModelTrainer(BaseTrainer):
    def _train_model(self, coordinator, labeled_idxs):
        model = linearmodel.LinearModel(coordinator.X.shape[1], coordinator.nclasses)
        trainer = pl.Trainer(max_epochs=100, devices=1, accelerator='cpu', enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False, logger=False)

        X_train = coordinator.X[labeled_idxs]
        y_train = coordinator.y[labeled_idxs]
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0))

        predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(coordinator.X), torch.from_numpy(coordinator.y))
        predictions = trainer.predict(model, ckpt_path=None, dataloaders=torch.utils.data.DataLoader(predict_dataset, num_workers=0))
        y_pred_probs = torch.stack(predictions).squeeze()
        return y_pred_probs.numpy()

    def get_y_probs(self, coordinator, labeled_idxs):
        return self._train_model(coordinator, labeled_idxs)

class QDATrainer(BaseTrainer):
    def get_y_probs(self, coordinator, labeled_idxs):
        clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
        clf.fit(coordinator.X[labeled_idxs], coordinator.y[labeled_idxs])
        return clf.predict_proba(coordinator.X)

class BaseUncertaintyALCoordinator(AbstractCoordinator):
    def __init__(self, *args, rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        # For now, assume we have a single feature.
        self.X = next(iter(self.Xs.items()))[1]
        self.nclasses = len(set(self.y))
        self.rng = np.random.default_rng(rng)
        self._hard_labels = -1 * np.ones_like(self.y_pred)
        self._trainer = None

    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    def trainer_cls(self):
        raise NotImplementedError

    def trainer(self):
        if self._trainer is None:
            self._trainer = self.trainer_cls()
        return self._trainer

    def _propagate_labels(self, y_pred_logits):
        unlabeled_idxs = np.where(self._hard_labels == -1)[0]
        self.y_pred[unlabeled_idxs] = y_pred_logits[unlabeled_idxs]

    def _interaction_round(self):
        labeled_idxs = np.where(self._hard_labels != -1)[0]

        # Random sample until we have 1% of the dataset labeled.
        if len(labeled_idxs) > 0.01 * len(self.y_pred):
            # Train model.
            y_pred_probs = self.trainer().get_y_probs(self, labeled_idxs)
            y_pred_probs_max = np.max(y_pred_probs, axis=1)

            # Propagate labels.
            self._propagate_labels(np.argmax(y_pred_probs, axis=1))

            # Pick most uncertain element that isn't labeled yet.
            # el_idx is a tensor, so use item() to get the value before checking for membership.
            for el_idx in np.argsort(y_pred_probs_max):
                if el_idx.item() not in labeled_idxs:
                    break
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

class LinearUncertaintyALCoordinator(BaseUncertaintyALCoordinator):
    @classmethod
    def name(cls):
        return 'uncertaintyal'

    @property
    def trainer_cls(self):
        return LinearModelTrainer

class QDAUncertaintyALCoordinator(BaseUncertaintyALCoordinator):
    @classmethod
    def name(cls):
        return 'uncertaintyal-qda'

    @property
    def trainer_cls(self):
        return QDATrainer
