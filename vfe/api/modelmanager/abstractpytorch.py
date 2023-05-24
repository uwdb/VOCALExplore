import datetime
import duckdb
import functools
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytorch_lightning as pl
from sklearn import model_selection
import torch
from torch.utils import data
from typing import Iterable, Dict, Union, Callable, List

from vfe import core
from vfe.core.timing import logtime
from vfe import models

from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.scheduler import Priority
from vfe.api.storagemanager import AbstractStorageManager, ModelInfo, LabelInfo, clipinfo_to_clipset, ClipSet, VidType
from .abstract import AbstractModelManager, PredictionSet

logger = logging.getLogger(__name__)

def _warn_for_unlabeled(logger, y, caller):
    n_notlabeled = len(np.where(y == 'none')[0])
    if n_notlabeled:
        logger.warn(f'{caller} found {n_notlabeled} clips that are not labeled')

class AbstractPytorchModelManager(AbstractModelManager):
    def __init__(self,
        storagemanager: AbstractStorageManager,
        featuremanager: AbstractFeatureManager,
        device=None,
        random_state=None,
        deterministic=False,
        epochs=100,
        batch_size=32,
        learningrate=0.0002,
    ):
        self.storagemanager = storagemanager
        self.featuremanager = featuremanager
        self.device = device if device is not None else \
            'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logger
        self.random_state = np.random.RandomState(random_state)
        self.deterministic = deterministic
        self.seed = random_state

        # Training configuration.
        self.batch_size = batch_size
        self.model_type = models.ModelType.LINEAR
        self.learningrate = learningrate
        self.epochs = epochs

        self.ignore_labels = set()


        # For evaluation.
        self._feature_to_last_prediction_mid = {}

    def latest_prediction_mid(self, feature_names: List[str]):
        feature_name = core.typecheck.ensure_str(feature_names)
        return self._feature_to_last_prediction_mid[feature_name]

    def add_labels(self, labels: Iterable[LabelInfo]):
        raise NotImplementedError

    def get_predictions(self, *, vids=None, start=None, end=None, feature_names:  Union[str, List[str]]=None, ignore_labeled=False) -> Iterable[PredictionSet]:
        raise NotImplementedError

    def get_label_counts(self, feature_names: Union[str, List[str]]) -> Dict[str, int]:
        return self.storagemanager.get_label_counts(feature_names)

    def get_total_label_time(self) -> Dict[str, float]:
        return self.storagemanager.get_total_label_time()

    def get_labels_for_clips(self, clipset: ClipSet, full_overlap=True):
        return self.storagemanager.get_labels_for_clips_aggregated_fulloverlap(clipset, full_overlap=full_overlap)

    def get_vids_with_labels(self) -> Iterable[VidType]:
        return self.storagemanager.get_vids_with_labels()

    def ignore_label_in_predictions(self, label) -> None:
        self.logger.debug(f'Ignoring label: {label}')
        self.ignore_labels.add(label)

    def check_label_quality(self, feature_names: Union[str, List[str]], n_splits=5, min_size=-1, f1_val=0.0) -> Dict[str, float]:
        # The keys of the dictionary contain the variables.
        # The values of the dictionary contain the median value for each variable across the splits.
        labels_and_features, unique_labels = self._get_all_labels_and_features(feature_names)

        return self._check_label_quality_base(
            feature_names, n_splits=n_splits,
            labels_and_features=labels_and_features,
            unique_labels=unique_labels,
            model_type=self.model_type,
            batch_size=self.batch_size,
            learningrate=self.learningrate,
            epochs=self.epochs,
            deterministic=self.deterministic,
            seed=self.seed,
            random_state=self.random_state,
            logger=self.logger,
            min_size=min_size,
            f1_val=f1_val,
            device=self.device,
        )

    @staticmethod
    def eval_splits(splits, X=None, y=None, feature_name: str=None, unique_labels=None, model_type=None, batch_size=None, learningrate=None, epochs=None, deterministic=None, seed=None, logger=None, f1_val=0.0, device=None):
        # feature_name is a str here because it has already been combined into a single string in the case
        # of concatenated features. It's just metadata information now.
        assert isinstance(feature_name, str), f'feature_name must be a string, but is {type(feature_name)}'

        if len(unique_labels) == 1:
            return [{}]

        if not isinstance(splits, list):
            # splits could be a generator.
            splits = list(splits)
        logger.debug(f'Evaluating {feature_name} on {len(splits)} splits')
        trainer = models.AbstractStrategy(
            outdir=None,
            training_info=models.BaseTrainingInfo(name=None, label_col=None, train_split=None, eval_split=None, feature=feature_name),
            model=model_type,
            save_models=False,
            batch_size=batch_size,
            learning_rate=learningrate,
            epochs=epochs,
            budgets=[],
            deterministic=deterministic,
            seed=seed,
        )
        results = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            results.append(trainer.train_and_evaluate_pytorch_model(X_train, y_train, X_test, y_test, labels=unique_labels, f1_val=f1_val, device=device))
        return results

    @staticmethod
    def postprocess_eval_split_results(results):
        results_df = pd.DataFrame(results)
        return results_df.median().to_dict()

    @staticmethod
    def _check_label_quality_base(feature_names: Union[str, List[str]], n_splits=5, labels_and_features=None, unique_labels=None, model_type=None, batch_size=None, learningrate=None, epochs=None, deterministic=None, seed=None, random_state=None, logger=None, return_tasks=False, min_size=-1, f1_val=0.0, device=None):
        if len(labels_and_features) <= n_splits \
                or len(unique_labels) <= n_splits:
            return None

        if min_size > 0 and len(labels_and_features) < min_size:
            return None

        if n_splits == 1:
            # Special case: do a 80/20 train/test split.
            n_splits = 5
            keep_first = True
        else:
            keep_first = False

        X = np.vstack(labels_and_features['feature'].to_numpy())
        y = labels_and_features['labels'].to_numpy()
        _, label_counts = np.unique(y, return_counts=True)
        if max(label_counts) < n_splits:
            logger.warn(f'Reducing n_splits from {n_splits} to {max(label_counts)} because not enough examples of each class')
            n_splits = max(label_counts)
            if n_splits == 1:
                logger.warn('Returning without checking label quality because there are too few examples of each class')
                return None
        _warn_for_unlabeled(logger, y, 'check_label_quality')
        if len(set(labels_and_features['vid'].to_numpy())) == 1:
            kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            kf = model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        splits = kf.split(X, y=labels_and_features['labels'].to_numpy(), groups=labels_and_features['vid'].to_numpy())
        if keep_first:
            splits = [list(splits)[0]]
        eval_kwds = dict(
            X=X,
            y=y,
            feature_name=core.typecheck.ensure_str(feature_names),
            unique_labels=unique_labels,
            model_type=model_type,
            batch_size=batch_size,
            learningrate=learningrate,
            epochs=epochs,
            deterministic=deterministic,
            seed=seed,
            logger=logger,
            f1_val=f1_val,
            device=device,
        )
        if return_tasks:
            return [functools.partial(AbstractPytorchModelManager.eval_splits, [split], **eval_kwds) for split in splits]

        results = AbstractPytorchModelManager.eval_splits(splits, **eval_kwds)
        return AbstractPytorchModelManager.postprocess_eval_split_results(results)

    def _get_all_labels_and_features(self, feature_names: Union[str, List[str]], vids=None, only_already_extracted=False):
        all_labels = list(self.storagemanager.get_labels(vids=vids, ignore_labels=self.ignore_labels))

        if not all_labels:
            return [], set()

        labelset = clipinfo_to_clipset(all_labels)
        labeled_features = self.featuremanager.get_features_for_clips(feature_names, labelset, only_already_extracted=only_already_extracted)
        labels_and_features = self.storagemanager.get_labels_for_features(labeled_features, ignore_labels=self.ignore_labels)
        unique_labels = set([l.label for l in all_labels])
        return labels_and_features, unique_labels

    def _train_model(self, feature_names: Union[str, List[str]], save=True, labels=None, train_kwargs={}):
        labels_and_features, unique_labels = self._get_all_labels_and_features(feature_names)
        labeled_vids = self.storagemanager.get_vids_with_labels()
        self.logger.debug(f'Training model using clips from {len(labeled_vids)} vids. Labels: {unique_labels}')
        self.logger.debug(f'Training set size: {len(labels_and_features)}')
        self.logger.debug(f'Labels in table: {set(labels_and_features["labels"].to_pylist())}')
        self.logger.debug(f'Label counts: {self.get_label_counts(feature_names)}')

        if labels is None:
            labels = unique_labels
        return self._train_model_for_features_and_labels(feature_names, labels_and_features, labels, save=save, train_kwargs=train_kwargs)

    def _train_model_for_features_and_labels(self, feature_names: Union[str, List[str]], labels_and_features, unique_labels, model_id=None, save=True, train_kwargs={}):
        trained_model_info = self._train_model_for_features_and_labels_base(feature_names, labels_and_features, unique_labels, model_id, self.storagemanager.get_models_dir(), self.model_type, self.batch_size, self.learningrate, self.epochs, self.device, self.deterministic, self.seed, **train_kwargs)

        # Tell the storage manager where these files are.
        if save:
            self.storagemanager.add_model(**trained_model_info)
        return trained_model_info

    @staticmethod
    def _train_model_for_features_and_labels_base(feature_names: Union[str, List[str]], labels_and_features, unique_labels, model_id, outdir, model_type, batch_size, learningrate, epochs, device, deterministic, seed, tensorboard=False, autolr=False, **train_kwargs):
        try:
            feature_names = core.typecheck.ensure_list(feature_names)
            # Train model.
            creation_time = datetime.datetime.today()
            model_name = str(creation_time).replace(' ', '-').replace(':', '-').replace('.', '-')
            if model_id:
                model_name = f'{model_id}_{model_name}'
            training_info = models.BaseTrainingInfo(model_name, label_col='', train_split='', eval_split='', feature=core.typecheck.ensure_str(feature_names))
            outdir = outdir
            trainer = models.AbstractStrategy(
                outdir=outdir,
                training_info=training_info,
                model=model_type,
                save_models=True,
                batch_size=batch_size,
                learning_rate=learningrate,
                epochs=epochs,
                budgets=[],
                deterministic=deterministic,
                seed=seed,
                tensorboard=tensorboard,
                autolr=autolr,
            )
            X = np.vstack(labels_and_features['feature'].to_numpy())
            y = labels_and_features['labels'].to_numpy()
            training_output = trainer.train_pytorch_model(
                X,
                y,
                unique_labels,
                device,
                **train_kwargs
            )
            return dict(
                model_type = model_type.value,
                feature_name=core.typecheck.ensure_str(feature_names),
                creation_time = creation_time,
                batch_size = batch_size,
                epochs = epochs,
                learningrate = learningrate,
                ntrain = len(labels_and_features),
                labels = training_output.labels.tolist(),
                model_path = training_output.ckpt_outfile,
                labels_path = training_output.classes_file,
                f1_threshold = training_output.f1_threshold,
            )
        except Exception as e:
            logger.exception(f'Failed to train model: {e}')
            return None


    def _predict_model(self, model_info: Union[ModelInfo, Callable[[], ModelInfo]], feature_names: Union[str, List[str]]=None, vids=None, start=None, end=None, ignore_labeled=False, sample_from_validation=-1, priority: Priority=Priority.DEFAULT):
        if vids is not None:
            features = self.featuremanager.get_features(feature_names, np.array(vids), priority=priority)
        else:
            features = self.featuremanager.get_features(feature_names, vids=None, priority=priority)
        labels_and_features = self.storagemanager.get_labels_for_features(features)
        if start is not None or end is not None:
            assert start is not None and end is not None, f'Both start and end must be not-None'
            filtered_features = duckdb.connect().execute("""
                SELECT *
                FROM labels_and_features
                WHERE (? >= start_time AND ? <= end_time)
                    OR (? >= start_time AND ? <= end_time)
                    OR (? <= start_time AND ? >= end_time)
            """, [start, start, end, end, start, end]).arrow()
        else:
            filtered_features = labels_and_features

        if ignore_labeled:
            filtered_features = filtered_features.filter(pc.equal(filtered_features['labels'], 'none'))

        if sample_from_validation > 0:
            self.logger.debug(f'Sampling {sample_from_validation} rows to evaluate on; originally there are {len(filtered_features)} rows.')
            random_idxs = self.random_state.choice(len(filtered_features), sample_from_validation, replace=False)
            filtered_features = filtered_features.take(random_idxs)

        if callable(model_info):
            model_info = model_info()
        if model_info is None:
             self.logger.error('Failed to find a trained model to perform predictions')
             return None
        tosave_feature_name = core.typecheck.ensure_str(feature_names)
        assert model_info.feature_name == tosave_feature_name, f'Error trying to use model trained with feature {model_info.feature_name} when specified feature {tosave_feature_name}'
        self.logger.debug(f'Performing predictions using model ({model_info.mid}) at path {model_info.model_path}, predicts labels {model_info.model_labels}, f1 threshold {model_info.f1_threshold:0.2f}')

        existing_predictions = self.storagemanager.get_predictions(model_info.mid, filtered_features)
        # Create two subsets:
        # One without predictions that is run through the model.
        # One with predictions that we will transform into the expected types.
        # Concat these before returning.
        missing_predictions = pc.is_null(existing_predictions['pred_dict'])
        filtered_features_for_inference = existing_predictions.filter(missing_predictions)
        filtered_features_with_predictions = existing_predictions.filter(pc.invert(missing_predictions))

        if len(filtered_features_for_inference):
            model = models.ModelType(model_info.model_type).get_cls().load_from_checkpoint(model_info.model_path)
            self.logger.debug(f'Loaded model')
            trainer = pl.Trainer(
                accelerator=self.device,
                devices=1,
                max_epochs=-1, # Not used since we're not training, but if not specified a warning is printed.
                enable_progress_bar=False,
                logger=False,
            )
            pt_dataset = data.TensorDataset(
                torch.from_numpy(np.vstack(filtered_features_for_inference['feature'].to_numpy())),
                torch.from_numpy(np.ones((len(filtered_features_for_inference), 1))),
            )
            self.logger.debug(f'Prepared datasets')
            y_pred_probs = torch.stack(trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(pt_dataset, num_workers=0)))
            if len(y_pred_probs.size()) > 2:
                y_pred_probs = y_pred_probs.squeeze()
            self.logger.debug('Got predictions')

            # TODO: read saved predictions and only run the model to get predictions on new videos.
            self.logger.debug(f'Saving predictions for model {model_info.mid}')
            self.storagemanager.add_predictions(model_info.mid, self._probs_to_predictionset(y_pred_probs, model_info.model_labels, filtered_features_for_inference))
        else:
            y_pred_probs = None

        # Concatenate y_pred_probs with filtered_features_with_predictions.
        if len(filtered_features_with_predictions):
            predictions = [json.loads(preds) for preds in filtered_features_with_predictions['pred_dict'].to_pylist()]
            y_pred_probs_existing = torch.stack([torch.Tensor([pred[label] for label in model_info.model_labels]) for pred in predictions])
            if len(y_pred_probs_existing.size()) > 2:
                y_pred_probs_existing = y_pred_probs_existing.squeeze()

            if y_pred_probs is not None:
                y_pred_probs = torch.cat([y_pred_probs, y_pred_probs_existing])
            else:
                y_pred_probs = y_pred_probs_existing
            filtered_features = pa.concat_tables([filtered_features_for_inference, filtered_features_with_predictions])
        else:
            filtered_features = filtered_features_for_inference

        # Now we're returning filtered_features with an extra 'pred_dict' column, but I think all of the callers will just ignore it.
        return y_pred_probs, model_info.model_labels, filtered_features

    def _predict_model_for_feature(self, feature_names: Union[str, List[str]], vids, start, end, ignore_labeled=False, priority: Priority=Priority.DEFAULT) -> Iterable[PredictionSet]:
        if (start is not None and end is None) or (start is None and end is not None):
            self.logger.warn('_predict_model may not correctly handle case where only one of start/end is None')

        # If just a single vid is passed, make it an arrray.
        if vids is not None and not hasattr(vids, '__iter__'):
            vids = [vids]

        # Wait to execute get_model_info until we've loaded features. For background training, it's possible a more up-to-date model
        # will be added between now and then.
        def model_info_lambda():
            feature_name = core.typecheck.ensure_str(feature_names)
            model_info = self.storagemanager.get_model_info(feature_name=feature_name, ignore_labels=self.ignore_labels)
            if model_info:
                self._feature_to_last_prediction_mid[feature_name] = model_info.mid
            return model_info

        return self._predict_model(model_info_lambda, feature_names=feature_names, vids=vids, start=start, end=end, ignore_labeled=ignore_labeled, priority=priority)

    def _model_perf(self, model_info: ModelInfo, vids, ignore_labels=[], groupby_vid=False, sample_from_validation=-1):
        # Returns dictionary with keys:
        #   nclasses, n_pred, pred_mAP, pred_avgprecision_<class>
        # This is weird ... we should probably just be storing a list of features in the model table.
        # If '+' is in the feature name, we assume it's a combination of features. Otherwise, we assume it's either a string or a list of features.
        feature_names = model_info.feature_name.split('+') if '+' in model_info.feature_name else model_info.feature_name
        y_pred_probs, model_labels, labels_and_features = self._predict_model(model_info, feature_names=feature_names, vids=vids, sample_from_validation=sample_from_validation)
        if y_pred_probs.dim() == 1:
            y_pred_probs = y_pred_probs.unsqueeze(dim=1)

        labels = model_info.model_labels
        _, to_multilabel = models.to_multilabel_gen(labels)
        true_labels = labels_and_features['labels'].to_numpy()
        labeled_idxs = np.where(~np.isin(true_labels, ['none', *ignore_labels]))[0]
        y_true = to_multilabel(true_labels[labeled_idxs])
        y_pred_probs = y_pred_probs[labeled_idxs]
        assert y_pred_probs.shape == y_true.shape, f'{y_pred_probs.shape} != {y_true.shape}'
        _warn_for_unlabeled(self.logger, true_labels[labeled_idxs], '_model_perf')

        if groupby_vid:
            # Assert that each vid has one label.
            vids = labels_and_features['vid'].to_numpy()[labeled_idxs]
            groups = pd.Series(true_labels).groupby(vids)
            if groups.nunique().max() > 1:
                self.logger.warn('_model_perf was called with groupby_vid, but some vids have multiple labels. Computing results without grouping.')
            else:
                y_shape = (len(groups.indices), y_true.shape[1])
                y_true_grouped = np.zeros(y_shape)
                y_pred_grouped = torch.zeros(y_shape)
                for i, (vid, indices) in enumerate(groups.indices.items()):
                    y_true_grouped[i] = y_true[indices[0]] # All rows will have the same label, so we can pick one.
                    y_pred_grouped[i] = torch.mean(y_pred_probs[indices], dim=0)
                y_true = y_true_grouped
                y_pred_probs = y_pred_grouped

        results = models.AbstractStrategy._results_for_split('pred', y_true, None, y_pred_probs, labels=labels, multilabel_threshold=model_info.f1_threshold)
        return results

    def _probs_to_predictionset(self, y_pred_probs, model_labels, clipset):
        # Return:
        # <vid, start, end, prediction probabilities>
        # Prediction probs: {<label>: <val>}
        if y_pred_probs.dim() == 1:
            y_pred_probs = y_pred_probs.unsqueeze(dim=1)
        predictions = []
        for i in range(len(clipset)):
            feature_info = clipset.take([i])
            # Each column is an array with one pyarrow-typed element.
            vid = feature_info['vid'][0].as_py()
            start_time = feature_info['start_time'][0].as_py()
            end_time = feature_info['end_time'][0].as_py()
            prediction_probs = {
                l: v.item()
                for l, v in zip(model_labels, y_pred_probs[i])
            }
            predictions.append(PredictionSet(vid, start_time, end_time, prediction_probs))
        return predictions
