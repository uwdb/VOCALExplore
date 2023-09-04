from collections import namedtuple
import datetime
from enum import Enum
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
import sys
import torch
from torch.utils import data
import torchmetrics

import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

from vfe import datasets
from . import linearmodel, mlp

TrainingOutput = namedtuple('TrainingOutput', ['trainer', 'model', 'labels', 'to_multilabel', 'ckpt_outfile', 'classes_file', 'f1_threshold'], defaults=(None,))

class BaseTrainingInfo:
    def __init__(self,
            name: str,
            label_col: str,
            train_split: str,
            eval_split: str,
            feature: str):
        self.name = name
        self.label_col = label_col
        self.train_split = train_split
        self.eval_split = eval_split
        self.feature = feature

    def __str__(self):
        return f'{self.name}_{self.label_col}_{self.train_split}_{self.eval_split}_{self.feature}'

    def nested_pieces(self):
        return f'{self.name}_{self.label_col}_{self.train_split}_{self.eval_split}', self.feature

class TrainingInfo(BaseTrainingInfo):
    def __init__(self,
            dataset: datasets.VFEDataset,
            label_col: str,
            train_split: str,
            eval_split: str,
            feature: str):
        super().__init__(dataset.name(), label_col, train_split, eval_split, feature)

class ModelType(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'

    def get_cls(self):
        if self == ModelType.LINEAR:
            return linearmodel.LinearModel
        elif self == self.MLP:
            return mlp.MLP
        else:
            assert False, f'Unrecognized model type: {self}'

def numpy_to_dataset(X, y=None):
    if y is None:
        y = np.zeros(len(X))
    return data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

def to_multilabel_gen(labels):
    mlb = preprocessing.MultiLabelBinarizer().fit([labels])
    return mlb.classes_, lambda y: mlb.transform([l.split('_') for l in y])

def to_softlabel(y):
    return np.where(y == 0, 0.1, y)

class AbstractStrategy:
    def __init__(self, outdir, training_info: BaseTrainingInfo, n_jobs=1, enable_progress_bar=False, model: ModelType=ModelType.LINEAR, save_models=False, batch_size=32, learning_rate=0.02, epochs=200, use_cosine_annealing=False, budgets=None, autolr=False, results_fns=[], outfile_suffix='', softlabel=False, tensorboard=None, deterministic=False, seed=None):
        self.outdir = outdir
        self.training_info = training_info
        self.n_jobs = n_jobs
        self._cached_df = None
        self.enable_progress_bar=enable_progress_bar
        self.model = model
        self.save_models = save_models
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_cosine_annealing = use_cosine_annealing
        self.autolr = autolr
        self.results_fns = results_fns
        self.outfile_suffix = outfile_suffix
        self.softlabel = softlabel
        if tensorboard is None:
            self.tensorboard = self.save_models
        else:
            self.tensorboard = tensorboard
        if deterministic:
            assert seed is not None, 'If training deterministically, must specify a seed.'
        if seed is not None:
            pl.seed_everything(seed, workers=True)
        self.deterministic = deterministic

        self.budgets = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75] if budgets is None else budgets

        self.logger = logging.getLogger(__name__)

    def train_and_evaluate_pytorch_model(self, X_train, y_train, X_test, y_test, it=0, model_name_suffix='', stratify_test=None, labels=None, device=None, f1_val=0.0):
        return self.train_and_evaluate_pytorch_model_multieval(X_train, y_train, {'test': (X_test, y_test)}, it=it, model_name_suffix=model_name_suffix, stratify_test=stratify_test, labels=labels, device=device, f1_val=f1_val)

    def train_pytorch_model(self, X_train, y_train, labels, device, it=0, model_name_suffix='', X_test=None, y_test=None, label_encoder=None, f1_val=0.0):
        logging.debug(f'Training model with X: {X_train.shape}, y: {y_train.shape}')

        # First, train model with entire X_train, y_train. This is the model that will be saved and used for making predictions.
        trainer, model, labels, to_multilabel = self._train_pytorch_model_base(X_train, y_train, labels, device, save_models=self.save_models, it=it, model_name_suffix=model_name_suffix, label_encoder=label_encoder)

        # If we are doing cross-validation to determine the f1 threshold, train a second model with a subset of train to find the threshold.
        if f1_val:
            if len(X_train) < 10:
                logging.debug(f'Tuning f1 threshold on entire training set (len={len(X_train)}) because too few samples to split into validation set')
                # If we're using the entire trianing set for f1 validation, we don't have to train a new model.
                X_train_val, y_train_val = X_train, y_train
                val_trainer, val_model = trainer, model
            else:
                logging.debug(f'Holding out {f1_val * 100}% of data to tune f1 threshold')
                try:
                    X_train, X_train_val, y_train, y_train_val = model_selection.train_test_split(X_train, y_train, test_size=f1_val, stratify=y_train)
                except ValueError as e:
                    if 'The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.' in str(e) \
                        or 'should be greater or equal to the number of classes' in str(e):
                        X_train, X_train_val, y_train, y_train_val = model_selection.train_test_split(X_train, y_train, test_size=f1_val)
                    else:
                        raise e

                # Train model on 80% X_train, y_train (or 100% if there isn't enough data).
                val_trainer, val_model, *val_rest = self._train_pytorch_model_base(X_train, y_train, labels, device, save_models=False, label_encoder=label_encoder)

            # Use held out 20% of original X_train, y_train to find best f1 threshold.
            f1_dataset = numpy_to_dataset(X_train_val)
            y_pred_probs = torch.stack(val_trainer.predict(val_model, ckpt_path=None, dataloaders=data.DataLoader(f1_dataset, num_workers=0))).squeeze()
            y_train_transformed_val = to_multilabel(y_train_val) if len(y_train_val.shape) == 1 else y_train_val
            f1_threshold = self._find_f1_threshold(y_train_transformed_val, y_pred_probs)
            # logging.debug(f'F1 threshold = {f1_threshold:0.2f} computed over {len(y_pred_probs)} samples')
            model_name_suffix += f'_f1th{f1_threshold:0.2f}_'
        else:
             f1_threshold = None

        if self.save_models:
            base_outfile = os.path.splitext(self.outfile)[0] +  f'_numdone{it}' + model_name_suffix
            ckpt_outfile = base_outfile + '.ckpt'
            trainer.save_checkpoint(ckpt_outfile)

            classes_file = None

            return_vals = [ckpt_outfile, classes_file]
        else:
            return_vals = [None, None]

        return TrainingOutput(trainer, model, labels, to_multilabel, *return_vals, f1_threshold)

    def _train_pytorch_model_base(self, X_train, y_train, labels, device,
        save_models=False,
        it=0,
        model_name_suffix='',
        label_encoder=None,
    ):
        logging.debug(f'Training model with X: {X_train.shape}, y: {y_train.shape}')

        if save_models:
            base_outfile = os.path.splitext(self.outfile)[0]
            checkpoint_root = base_outfile + f'_numdone{it}' + model_name_suffix + '_ckpt'
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_root,
                save_last=True,
                train_time_interval=datetime.timedelta(seconds=600),
            )
            callbacks=[checkpoint_callback]
            ckpt_path = os.path.join(checkpoint_root, 'last.ckpt')
            if not os.path.exists(ckpt_path):
                ckpt_path=None
        else:
            ckpt_path=None
            callbacks=[]
            checkpoint_root=None

        if self.tensorboard:
            # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
            log_dir=os.path.join(self.outdir, 'tensorboard', self.log_subdir + model_name_suffix)
            logger = pl.loggers.TensorBoardLogger(log_dir, name=f'lr{self.learning_rate}', default_hp_metric=False)
            print('Logging to', log_dir)
            sys.stdout.flush()
        else:
            logger = False

        trainer = pl.Trainer(
            # deterministic=self.deterministic,
            max_epochs=self.epochs,
            devices=1,
            accelerator=device,
            enable_progress_bar=self.enable_progress_bar,
            enable_checkpointing=self.save_models,
            enable_model_summary=False,
            logger=logger,
            default_root_dir=checkpoint_root,
            callbacks=callbacks,
            check_val_every_n_epoch=5,
            log_every_n_steps=min(50, max(1, len(X_train)-1)),
            auto_lr_find=self.autolr,
        )

        labels, to_multilabel = to_multilabel_gen(labels)
        if label_encoder and np.any(labels != label_encoder.classes_):
            raise RuntimeError(f'MultiLabelBinarizer classes do not equal provided label_encoder classes: {labels} != {label_encoder.classes_}')

        dim_in = X_train.shape[1]
        dim_out = len(labels)
        if self.model == ModelType.LINEAR:
            model = linearmodel.LinearModel(dim_in, dim_out, self.learning_rate, self.training_info.feature, epochs=self.epochs, batch_size=self.batch_size, use_cosine_annealing=self.use_cosine_annealing)
        elif self.model == ModelType.MLP:
            model = mlp.MLP(dim_in, dim_out)
        else:
            raise RuntimeError('Unhandled model type', self.model)

        y_train_transformed = to_multilabel(y_train) if len(y_train.shape) == 1 else y_train
        if self.softlabel:
            y_train_transformed = to_softlabel(y_train_transformed)

        logging.debug(f'y_train (len={len(y_train_transformed)}): {[(label, count) for label, count in zip(labels, np.sum(y_train_transformed, axis=0))]}')
        train_dataset = numpy_to_dataset(X_train, y_train_transformed)
        train_dataloader = data.DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        if self.autolr:
            print('Tuning model for autolr')
            trainer.tune(model, train_dataloaders=train_dataloader)

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=None,
            ckpt_path=ckpt_path
        )

        return trainer, model, labels, to_multilabel

    @staticmethod
    def _find_f1_threshold(y, y_pred_probs):
        logging.debug(f'Finding f1 threshold (y.shape={y.shape})')

        num_labels = y.shape[1]
        if num_labels == 1:
            logging.error(f'Tried to find f1 threshold with a single class; returning default of 0.5')
            return 0.5

        precisions, recalls, thresholds = torchmetrics.functional.classification.multilabel_precision_recall_curve(y_pred_probs, torch.tensor(y), num_labels=num_labels, thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        denom = precisions + recalls
        denom[denom == 0] = 1.0
        f1s = (2 * precisions * recalls) / denom
        f1s = f1s.mean(dim=0)
        logging.debug(f'F1s at candidate thresholds: {f1s.tolist()}')
        return thresholds[torch.argmax(f1s)].item()

    def train_and_evaluate_pytorch_model_multieval(self, X_train, y_train, eval_info, it=0, model_name_suffix='', stratify_test=None, label_encoder=None, labels=None, device=None, f1_val=0.0):
        # Convert y's to class indices.
        # It's possible that y_train doesn't contain all of the labels, so use y_test.
        X_test, y_test = eval_info['test']
        # eval_info['l+u'][1] is the y-values for everything not in y-test. By using this we ensure labels contains all possible labels.
        y_not_test = y_train if 'l+u' not in eval_info else eval_info['l+u'][1]
        # to_class_index = lambda y: np.array([labels.index(label) for label in y])
        if labels is None:
            labels = [l for l in sorted(set(y_test) | set(y_not_test)) if '_' not in l] if label_encoder is None else label_encoder.classes_

        if device is None:
            device = 'gpu' if torch.cuda.is_available() else 'cpu'
        training_output = self.train_pytorch_model(X_train, y_train, labels, device, it=it, model_name_suffix=model_name_suffix, X_test=X_test, y_test=y_test, label_encoder=label_encoder, f1_val=f1_val)
        trainer = training_output.trainer
        model = training_output.model
        labels = training_output.labels
        to_multilabel = training_output.to_multilabel
        f1_threshold = training_output.f1_threshold

        eval_info['train'] = (X_train, y_train)
        results = {}
        for split, (X, y) in eval_info.items():
            y_transformed = to_multilabel(y) if len(y.shape) == 1 else y
            split_dataset = numpy_to_dataset(X, y_transformed)
            y_pred_probs = torch.stack(trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(split_dataset, num_workers=0))).squeeze()

            y_pred_probs_og = y_pred_probs.detach().clone()
            if len(y_pred_probs.shape) == 0:
                # One value. Add a dimension.
                y_pred_probs = torch.unsqueeze(y_pred_probs, dim=1)
                self.logger.debug(f'Before shape: {y_pred_probs_og.shape}; after shape: {y_pred_probs.shape} (y.shape: {y.shape})')
            if len(y_pred_probs.shape) == 1:
                # One class. Add a dimension.
                # Example: one row with two possible values. Transform from tensor([p1, p2]) to tensor([[p1, p2]]).
                y_pred_probs = torch.unsqueeze(y_pred_probs, dim=0)
                self.logger.debug(f'Before shape: {y_pred_probs_og.shape}; after shape: {y_pred_probs.shape} (y.shape: {y.shape})')


            y_pred = np.argmax(y_pred_probs, axis=1)

            # if self.save_models and split == 'test':
            #     base_outfile = os.path.splitext(self.outfile)[0]
            #     ypred_outfile = base_outfile + f'_numdone{it}' + model_name_suffix + '_ypred.npy'
            #     np.save(ypred_outfile, y_pred)

            results.update(
                self._results_for_split(split, y_transformed, y_pred, y_pred_probs, stratify=stratify_test if split == 'test' else None, nclasses=len(labels), labels=labels, multilabel_threshold=f1_threshold)
            )

            for results_fn in self.results_fns:
                results.update(
                    results_fn(split, y_transformed, y_pred, y_pred_probs, labels)
                )

        return results

    @staticmethod
    def evaluate_model(X, trainer, model):
        dataset = numpy_to_dataset(X, np.zeros(len(X)))
        y_pred_probs = torch.stack(trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(dataset, num_workers=0))).squeeze()
        return y_pred_probs

    @staticmethod
    def _results_for_split(split, y, y_pred, y_pred_probs, stratify=None, nclasses=None, labels=None, multilabel_threshold=0.4):
        multilabel_threshold = multilabel_threshold if multilabel_threshold is not None else 0.4

        logging.debug(f'Results for split {split} (len={len(y)}): {[(label, count) for label, count in zip(labels, np.sum(y, axis=0))]}')

        y = torch.tensor(y)
        if labels is not None and not nclasses:
            nclasses = len(labels)

        if nclasses == 1:
            return {'nclasses': nclasses}

        average_precision_class = torchmetrics.functional.classification.multilabel_average_precision(y_pred_probs, y, num_labels=nclasses, average=None)
        if len(average_precision_class.shape) == 0:
            # If there is one class, then average_precision_class will be a single tensor rather than a list of tensors.
            average_precision_class = [average_precision_class]

        average_precision_scores = {
            f'{split}_avgprecision_{labels[i]}': float(average_precision_class[i].item())
            for i in range(len(average_precision_class))
        }
        average_precision_scores[f'{split}_mAP'] = float(torchmetrics.functional.classification.multilabel_average_precision(y_pred_probs, y, num_labels=nclasses, average='macro'))

        # stratify_results = {}
        # if stratify is not None:
        #     for group in set(stratify):
        #         group_idxs = np.where(stratify == group)
        #         y_true_group = y[group_idxs]
        #         y_pred_group = y_pred[group_idxs]
        #         y_pred_probs_group = y_pred_probs[group_idxs]
        #         stratify_results.update({
        #             f'{split}_gtest_accuracy_g{group}': metrics.accuracy_score(y_true_group, y_pred_group),
        #             f'{split}_gtest_f1_macro_g{group}': metrics.f1_score(y_true_group, y_pred_group, average='macro'),
        #             f'{split}_gmAP_g{group}': float(torchmetrics.functional.average_precision(y_pred_probs_group, torch.tensor(y_true_group), num_classes=nclasses, average='macro')),
        #         })

        # Compare validation f1 threshold vs. actual best on test set.
        f1_threshold = AbstractStrategy._find_f1_threshold(y, y_pred_probs)
        logging.debug(f'Passed multilabel_threshold={multilabel_threshold:0.2f}; best on split={f1_threshold:0.2f}')

        y_top1 = torch.argmax(y, dim=1)
        # logging.debug(f'Maximum y_pred_probs: {torch.max(y_pred_probs, dim=0).values}')
        # logging.debug(f'Minimum y_pred_probs: {torch.min(y_pred_probs, dim=0).values}')
        results = {
            'nclasses': nclasses,
            f'{split}_ml_accuracy_macro': torchmetrics.functional.classification.multilabel_accuracy(y_pred_probs, y, num_labels=nclasses, average='macro', threshold=multilabel_threshold).item(),
            f'{split}_ml_accuracy_micro': torchmetrics.functional.classification.multilabel_accuracy(y_pred_probs, y, num_labels=nclasses, average='micro', threshold=multilabel_threshold).item(),
            f'{split}_ml_f1_score_macro': torchmetrics.functional.classification.multilabel_f1_score(y_pred_probs, y, num_labels=nclasses, average='macro', threshold=multilabel_threshold).item(),
            f'{split}_ml_f1_score_micro': torchmetrics.functional.classification.multilabel_f1_score(y_pred_probs, y, num_labels=nclasses, average='micro', threshold=multilabel_threshold).item(),
           **{f'{split}_ml_f1_score_{label}': val.item() for label, val in zip(labels, torchmetrics.functional.classification.multilabel_f1_score(y_pred_probs, y, num_labels=nclasses, average='none', threshold=multilabel_threshold))},
            f'{split}_mc_accuracy_macro': torchmetrics.functional.classification.multiclass_accuracy(y_pred_probs, y_top1, num_classes=nclasses, average='macro').item(),
            f'{split}_mc_accuracy_micro': torchmetrics.functional.classification.multiclass_accuracy(y_pred_probs, y_top1, num_classes=nclasses, average='micro').item(),
            f'{split}_mc_f1_score_macro': torchmetrics.functional.classification.multiclass_f1_score(y_pred_probs, y_top1, num_classes=nclasses, average='macro').item(),
            f'{split}_mc_f1_score_micro': torchmetrics.functional.classification.multiclass_f1_score(y_pred_probs, y_top1, num_classes=nclasses, average='micro').item(),
           **{f'{split}_mc_f1_score_{label}': val.item() for label, val in zip(labels, torchmetrics.functional.classification.multiclass_f1_score(y_pred_probs, y_top1, num_classes=nclasses, average='none'))},
            f'n_{split}': len(y),
            **average_precision_scores,
            # **stratify_results,
        }
        if nclasses > 5:
            results.update({
                f'{split}_mc_accuracy_top5_macro': torchmetrics.functional.classification.multiclass_accuracy(y_pred_probs, y_top1, num_classes=nclasses, top_k=5, average='macro').item(),
                f'{split}_mc_accuracy_top5_micro': torchmetrics.functional.classification.multiclass_accuracy(y_pred_probs, y_top1, num_classes=nclasses, top_k=5, average='micro').item(),
            })
        return results

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        # The number of folds can't be more than the minimum number of times any label appears.
        cv = min(5, pd.Series(y_train).groupby(y_train).count().min())
        solver = 'lbfgs' if len(X_train) <= 10000 else 'saga'
        pca_args = [
            preprocessing.StandardScaler(),
            linear_model.LogisticRegressionCV(Cs=[1, 10, 100], solver=solver, multi_class='multinomial', max_iter=500, cv=cv, n_jobs=self.n_jobs)
        ]
        clf = pipeline.make_pipeline(*pca_args)
        clf.fit(X_train, y_train)
        predicted_test = clf.predict(X_test)
        results = self._results(y_test, predicted_test, y_train, clf.predict(X_train))
        results['solver'] = solver
        if results['nclasses'] > 100:
            results['test_top5_accuracy'] = metrics.top_k_accuracy_score(y_test, clf.predict_proba(X_test), k=5)
            results['train_top5_accuracy'] = metrics.top_k_accuracy_score(y_train, clf.predict_proba(X_train), k=5)
        return results

    @property
    def outfile(self):
        model_str = '' if self.model == ModelType.LINEAR else f'_{self.model.value}'
        softlabel_str = '_sl' if self.softlabel else ''
        return os.path.join(self.outdir, f'{self.name()}{model_str}{softlabel_str}_{self.training_info}{self.outfile_suffix}.pkl')

    @property
    def log_subdir(self):
        model_str = '' if self.model == ModelType.LINEAR else f'_{self.model.value}'
        softlabel_str = '_sl' if self.softlabel else ''
        prefix, subdir = self.training_info.nested_pieces()
        return os.path.join(f'{self.name()}{model_str}{softlabel_str}_{prefix}', subdir)

    def _augment_results(self, results):
        return {
            **results,
            'dataset': self.training_info.name,
            'label_col': self.training_info.label_col,
            'train_split': self.training_info.train_split,
            'eval_split': self.training_info.eval_split,
            'feature': self.training_info.feature,
            'strategy': self.name(),
            'model': self.model.value,
            'learningrate': self.learning_rate,
            'epochs': self.epochs,
            'cosineanneal': self.use_cosine_annealing,
            'autolr': self.autolr,
            'batch_size': self.batch_size,
            'softlabel': self.softlabel,
        }

    @property
    def results_df(self):
        if self._cached_df is None:
            self._cached_df = pd.read_pickle(self.outfile)
        return self._cached_df

    @results_df.setter
    def results_df(self, results_df):
        self._cached_df = results_df

    def save_results(self, results):
        results = map(self._augment_results, results)
        outfile = self.outfile
        if not os.path.exists(outfile):
            self.results_df = pd.DataFrame(results)
        else:
            self.results_df = pd.concat([self.results_df, pd.DataFrame(results)])
        self.results_df.to_pickle(outfile)

    def num_done(self, budget=None):
        if not os.path.exists(self.outfile):
            return 0
        df = self.results_df
        # Assumes each file will contain results for a single model type.
        filtered = df[
            (df.dataset == self.training_info.dataset.name())
            & (df.label_col == self.training_info.label_col)
            & (df.train_split == self.training_info.train_split)
            & (df.eval_split == self.training_info.eval_split)
            & (df.feature == self.training_info.feature)
            & (df.strategy == self.name())
        ]
        if budget is not None:
            filtered = filtered[(filtered.budget == budget)]
        return len(filtered)

    def check_Xy(self, X_train, y_train, X_test, y_test):
        if X_test is None:
            random_state = 0 if self.save_models else None
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state)
        return X_train, y_train, X_test, y_test

    @classmethod
    def name(cls):
        return ''

    def process_it(self, X_train, y_train, X_test, y_test, desired_count, stratify_train=None, stratify_test=None):
        raise NotImplementedError

    def process(self, X_train_in, y_train_in, X_test_in, y_test_in, desired_count=10, stratify_train=None, stratify_test=None):
        for _ in range(desired_count):
            # Check_Xy should be splitting stratify_train as well, but it shouldn't be an issue since we're using separate train/test splits, so we don't have to split anything.
            X_train, y_train, X_test, y_test = self.check_Xy(X_train_in, y_train_in, X_test_in, y_test_in)
            self.process_it(X_train, y_train, X_test, y_test, desired_count, stratify_train, stratify_test)

    def process_dataset(self, dataset):
        raise NotImplementedError
