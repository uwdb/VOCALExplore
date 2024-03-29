import datetime
import duckdb
from functools import partial
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from sklearn import model_selection
import threading
import torch
from torch.utils import data
from typing import Iterable, Dict, Union, Callable, List

from vfe import core
from vfe import models

from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.scheduler import AbstractScheduler, Priority
from vfe.api.storagemanager import AbstractStorageManager, ModelInfo, LabelInfo, clipinfo_to_clipset, ClipSet, VidType
from .abstract import AbstractModelManager, PredictionSet

logger = logging.getLogger(__name__)

def _warn_for_unlabeled(logger, y, caller):
    n_notlabeled = len(np.where(y == 'none')[0])
    if n_notlabeled:
        logger.warn(f'{caller} found {n_notlabeled} clips that are not labeled')

def probs_to_predictionset(y_pred_probs, model_labels, clipset):
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

class AbstractPytorchModelManager(AbstractModelManager):
    def __init__(self,
        storagemanager: AbstractStorageManager,
        featuremanager: AbstractFeatureManager,
        scheduler: AbstractScheduler = None,
        device=None,
        random_state=None,
        deterministic=False,
        epochs=100,
        batch_size=32,
        learningrate=0.0002,
        predict_on_none=False
    ):
        self.storagemanager = storagemanager
        self.featuremanager = featuremanager
        self.scheduler = scheduler
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
        self.predict_on_none = predict_on_none

        self.ignore_labels = set()

        # For evaluation.
        self._feature_to_last_prediction_mid = {}

        # Async predictions.
        if self.scheduler:
            # We can't pass the queue to the async function unless it comes from a Manager.
            self._predictions_queue = self.scheduler.context().Manager().Queue() # Slow; starts a new process.
            self._prediction_processing_thread = threading.Thread(group=None, target=self._handle_prediction_batch, args=(self._predictions_queue,), name='process-predictions')
            self._prediction_processing_thread.daemon = True
            self._prediction_processing_thread.start()

    def _handle_prediction_batch(self, predictions_queue):
        while True:
            prediction_kwargs = predictions_queue.get()
            self.logger.debug("Read predictions batch")
            self.storagemanager.add_predictions(**prediction_kwargs)

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

        # Split y into multi-labels to handle the case where multiple labels overlap.
        ordered_labels, to_multilabel = models.to_multilabel_gen((unique_labels))
        y_transformed = to_multilabel(y)
        label_counts = np.sum(y_transformed, axis=0)

        # Filter out labels that have fewer than n_splits to avoid 0 f1s.
        sufficient_labels = np.where(label_counts >= n_splits)[0]
        ordered_labels = ordered_labels[sufficient_labels]
        y_transformed_filtered = y_transformed[:, sufficient_labels]
        # Pick rows with at least one label still.
        per_vid_labels = np.sum(y_transformed_filtered, axis=1)
        labeled_videos_idxs = np.where(per_vid_labels > 0)[0]
        if len(labeled_videos_idxs) < n_splits:
            logger.warn('Returning without checking label quality because after eliminating labels there are not enough examples')
            return None

        y_transformed_filtered = y_transformed_filtered[labeled_videos_idxs]
        y = np.array(['_'.join(ordered_labels[np.where(y_row == 1)]) for y_row in y_transformed_filtered])
        X = X[labeled_videos_idxs]

        # Not great, but re-add these checks with the filtered X, y.
        if len(X) <= n_splits \
                or (len(sufficient_labels) <= n_splits and len(sufficient_labels) > 2): # I don't remember why the l(unique)<nsplits came from. Keep it for now and special-case for bears.
            return None

        if min_size > 0 and len(X) < min_size:
            return None

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

        # In multi-label case, we'll pick the first label to stratify on.
        # splits = kf.split(X, y=np.argmax(y_transformed, axis=1), groups=labels_and_features['vid'].to_numpy())
        splits = kf.split(X, y=np.argmax(y_transformed_filtered, axis=1), groups=labels_and_features['vid'].to_numpy()[labeled_videos_idxs])
        if keep_first:
            splits = [list(splits)[0]]
        eval_kwds = dict(
            X=X,
            y=y,
            feature_name=core.typecheck.ensure_str(feature_names),
            unique_labels=ordered_labels,
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
            return [partial(AbstractPytorchModelManager.eval_splits, [split], **eval_kwds) for split in splits]

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

    def _train_model(self, feature_names: Union[str, List[str]], save=True, labels=None, train_kwargs={}, vids=None):
        labels_and_features, unique_labels = self._get_all_labels_and_features(feature_names, vids=vids)
        labeled_vids = self.storagemanager.get_vids_with_labels()
        if vids is not None:
            labeled_vids = np.array([l for l in labeled_vids if l in vids])
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

    class CustomWriter(BasePredictionWriter):
        def __init__(self, prediction_queue=None, features_table=None, model_info=None, write_interval='batch'):
            super().__init__(write_interval)
            self.prediction_queue = prediction_queue
            self.features_table = features_table
            self.model_info = model_info

        def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
            batch_indices = batch[1]
            if len(prediction.size()) > 2:
                prediction = prediction.squeeze()
            prediction_batch = {
                'mid': self.model_info.mid,
                'predictionset_list': probs_to_predictionset(
                    prediction,
                    self.model_info.model_labels,
                    self.features_table.take(batch_indices.cpu().numpy())
                )
            }
            self.prediction_queue.put(prediction_batch)

    @staticmethod
    def _predict_model_async(model_info: ModelInfo, features: pa.Table, accelerator: str, predictions_queue):
        model = models.ModelType(model_info.model_type).get_cls().load_from_checkpoint(model_info.model_path)
        try:
            logger.debug('Loaded model')
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=1,
                max_epochs=-1,
                enable_progress_bar=False,
                logger=False,
                callbacks=[AbstractPytorchModelManager.CustomWriter(prediction_queue=predictions_queue, features_table=features, model_info=model_info)]
            )
            pt_dataset = data.TensorDataset(
                torch.from_numpy(np.vstack(features['feature'].to_numpy())),
                # The second element of each batch will be the index of each prediction.
                torch.from_numpy(np.arange(len(features))),
            )
            logger.debug(f'Prepared datasets')
            trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(pt_dataset, num_workers=0, batch_size=256), return_predictions=False)
            logger.debug('Got predictions')
        except Exception as e:
            logger.exception("Exception: {e}")

    def _predict_model(self, model_info: Union[ModelInfo, Callable[[], ModelInfo]], feature_names: Union[str, List[str]]=None, vids=None, start=None, end=None, ignore_labeled=False, sample_from_validation=-1, priority: Priority=Priority.DEFAULT, run_async=False, cache_predictions=True):
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
        self.logger.debug(f'Performing predictions using model ({model_info.mid}) at path {model_info.model_path}, predicts labels {model_info.model_labels}, f1 threshold {model_info.f1_threshold or -1:0.2f}')

        existing_predictions = self.storagemanager.get_predictions(model_info.mid, filtered_features)
        # Create two subsets:
        # One without predictions that is run through the model.
        # One with predictions that we will transform into the expected types.
        # Concat these before returning.
        missing_vids = set(filtered_features['vid'].to_pylist()) - set(existing_predictions['vid'].to_pylist())
        missing_predictions = pc.is_in(filtered_features['vid'], pa.array(missing_vids))
        # For performing inference, we need the raw features.
        filtered_features_for_inference = filtered_features.filter(missing_predictions)
        # For vids with predictions, we can just use what we got from the storage manager.
        filtered_features_with_predictions = existing_predictions

        if run_async:
            assert self.scheduler, f"Cannot perform asynchronous predictions without a scheduler."
            if not len(filtered_features_for_inference):
                logger.debug(f'Returning before scheduling predict because no predictions are missing for vids {min(vids)}-{max(vids)}')
                return
            logger.debug(f'Scheduling async prediction task for vids {min(vids)}-{max(vids)}')
            self.scheduler.schedule(
                'predict',
                partial(
                    self._predict_model_async,
                    model_info=model_info,
                    features=filtered_features_for_inference,
                    accelerator=self.device,
                    predictions_queue=self._predictions_queue,
                ),
                priority=priority,
            )
            return

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
            y_pred_probs = torch.cat(trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(pt_dataset, num_workers=0, batch_size=256)))
            if len(y_pred_probs.size()) > 2:
                y_pred_probs = y_pred_probs.squeeze()
            self.logger.debug('Got predictions')

            if cache_predictions:
                self.logger.debug(f'Saving predictions for model {model_info.mid}')
                self.storagemanager.add_predictions(model_info.mid, probs_to_predictionset(y_pred_probs, model_info.model_labels, filtered_features_for_inference))
        else:
            y_pred_probs = None

        # Concatenate y_pred_probs with filtered_features_with_predictions.
        common_columns = ["fid", "vid", "start_time", "end_time", "labels", "feature"]
        if len(filtered_features_with_predictions):
            # Do the join first in case things get re-ordered so that the order of
            # y_pred_probs matches.
            # Add labels and fid to filtered_featueres_with_predictions.
            # Otherwise it only has vid, start_time, end_time, and pred_dict.
            filtered_features_with_predictions = duckdb.connect().execute("""
                SELECT p.*, ff.labels, ff.fid, ff.feature
                FROM filtered_features_with_predictions p, filtered_features ff
                WHERE p.vid=ff.vid AND p.start_time=ff.start_time AND p.end_time=ff.end_time
            """).arrow()

            predictions = [json.loads(preds) for preds in filtered_features_with_predictions['pred_dict'].to_pylist()]
            y_pred_probs_existing = torch.stack([torch.Tensor([pred[label] for label in model_info.model_labels]) for pred in predictions])
            if len(y_pred_probs_existing.size()) > 2:
                y_pred_probs_existing = y_pred_probs_existing.squeeze()

            if y_pred_probs is not None:
                y_pred_probs = torch.cat([y_pred_probs, y_pred_probs_existing])
            else:
                y_pred_probs = y_pred_probs_existing

            filtered_features = pa.concat_tables([filtered_features_for_inference.select(common_columns), filtered_features_with_predictions.select(common_columns)])
        else:
            filtered_features = filtered_features_for_inference.select(common_columns)

        # Now we're returning filtered_features with an extra 'pred_dict' column, but I think all of the callers will just ignore it.
        return y_pred_probs, model_info.model_labels, filtered_features

    def _predict_model_for_feature(self, feature_names: Union[str, List[str]], vids, start, end, ignore_labeled=False, priority: Priority=Priority.DEFAULT, run_async=False, cache_predictions=True) -> Union[Iterable[PredictionSet], None]:
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

        return self._predict_model(model_info_lambda, feature_names=feature_names, vids=vids, start=start, end=end, ignore_labeled=ignore_labeled, priority=priority, run_async=run_async, cache_predictions=cache_predictions)

    def _model_perf(self, model_info: ModelInfo, vids, ignore_labels=[], groupby_vid=False, sample_from_validation=-1, include_none=False, cache_predictions=True):
        # Returns dictionary with keys:
        #   nclasses, n_pred, pred_mAP, pred_avgprecision_<class>
        # This is weird ... we should probably just be storing a list of features in the model table.
        # If '+' is in the feature name, we assume it's a combination of features. Otherwise, we assume it's either a string or a list of features.
        feature_names = model_info.feature_name.split('+') if '+' in model_info.feature_name else model_info.feature_name
        y_pred_probs, model_labels, labels_and_features = self._predict_model(model_info, feature_names=feature_names, vids=vids, sample_from_validation=sample_from_validation, cache_predictions=cache_predictions)
        if y_pred_probs.dim() == 1:
            y_pred_probs = y_pred_probs.unsqueeze(dim=1)

        labels = model_info.model_labels
        _, to_multilabel = models.to_multilabel_gen(labels)
        true_labels = labels_and_features['labels'].to_numpy()
        if not include_none and not self.predict_on_none:
            if isinstance(ignore_labels, set):
                ignore_labels.add('none')
            else:
                ignore_labels.append('none')

        labeled_idxs = np.where(~np.isin(true_labels, [*ignore_labels]))[0]
        y_true = to_multilabel(true_labels[labeled_idxs])
        y_pred_probs = y_pred_probs[labeled_idxs]
        true_labels = true_labels[labeled_idxs]
        assert y_pred_probs.shape == y_true.shape, f'{y_pred_probs.shape} != {y_true.shape}'
        _warn_for_unlabeled(self.logger, true_labels[labeled_idxs], '_model_perf')

        if groupby_vid:
            # Assert that each vid has one label.
            vids = labels_and_features['vid'].to_numpy()[labeled_idxs]
            groups = pd.Series(true_labels).groupby(vids)
            if False and groups.nunique().max() > 1:
                self.logger.warn('_model_perf was called with groupby_vid, but some vids have multiple labels. Computing results without grouping.')
            else:
                y_shape = (len(groups.indices), y_true.shape[1])
                y_true_grouped = np.zeros(y_shape)
                y_pred_grouped = torch.zeros(y_shape)
                for i, (vid, indices) in enumerate(groups.indices.items()):
                    # If there are multiple labels for a video, create y_true such that there is a 1 for each one.
                    y_true_grouped[i] = np.max(y_true[indices], axis=0)
                    # y_true[indices[0]] # All rows will have the same label, so we can pick one.
                    y_pred_grouped[i] = torch.max(y_pred_probs[indices], dim=0).values
                y_true = y_true_grouped.astype(int)
                y_pred_probs = y_pred_grouped

        results = models.AbstractStrategy._results_for_split('pred', y_true, None, y_pred_probs, labels=labels, multilabel_threshold=model_info.f1_threshold)
        return results
