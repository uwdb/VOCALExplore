import atexit
from functools import partial
import logging
import threading
import time
import torch
from typing import Iterable, Union, List

from vfe.core.logging import configure_logger
from vfe.api.featuremanager import AbstractAsyncFeatureManager
from vfe.api.storagemanager import LabelInfo
from .abstract import PredictionSet
from .abstractpytorch import AbstractPytorchModelManager, probs_to_predictionset

# BGModelManager could also speed things up by training models on just clips with features
# already extracted, even if they are a subset of all labeled data.
class BackgroundModelManager(AbstractPytorchModelManager):
    def _train_model_async(self, callback=None):
        self.logger.debug(f'getting all features for labeled vids')
        with self._new_features_lock:
            self._outstanding_training_jobs += 1
        labels_and_features, unique_labels = self._get_all_labels_and_features(self.feature_names)
        # Race condition: new labels could be added  between get_labels in
        # _get_all_labels_and_features and resetting the event here.
        # Do nothing for now because eliminating the race condition would require adding a lock,
        # and the downside of the race condition is minor: the current model will have not been
        # trained with the new labels, but we will retrain when more labels come in.
        self._new_features_event.clear()
        self.logger.debug(f'labels {unique_labels}, vids {set(labels_and_features["vid"].to_pylist())}')
        self._training_pool.apply_async(
            self._train_model_for_features_and_labels_base,
            kwds=dict(
                feature_names=self.feature_names,
                labels_and_features=labels_and_features,
                unique_labels=unique_labels,
                model_id=None,
                outdir=self.storagemanager.get_models_dir(),
                model_type=self.model_type,
                batch_size=self.batch_size,
                learningrate=self.learningrate,
                epochs=self.epochs,
                device=self.device,
                deterministic=self.deterministic,
                seed=self.seed,
            ),
            callback=partial(self._handle_trained_model, callback=callback)
        )

    def _handle_new_labels_and_features(self):
        while True:
            self._new_features_event.wait()
            self._pause_event.wait()
            # Now there are new features and labels. Retrain the model.
            self._train_model_async()

    def _handle_trained_model(self, trained_model_info, callback=None):
        self.logger.debug(f'Saving model with info: {trained_model_info}')
        self.storagemanager.add_model(**trained_model_info)
        with self._new_features_lock:
            self._outstanding_training_jobs -= 1
            events = self._finish_training_events
            self._finish_training_events = []
        if callback is not None:
            callback()
        for event in events:
            event.set()

    def __init__(self,
            *args,
            feature_names: List[str]=None,
            num_processes=1,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._asyncfm = isinstance(self.featuremanager, AbstractAsyncFeatureManager)
        self.logger = logging.getLogger(__name__)
        self.feature_names = feature_names
        self.num_processes = num_processes

        # Training pool. Training with pytorch requires 'spawn'.
        self._ctx = torch.multiprocessing.get_context('spawn')
        self._training_pool = self._ctx.Pool(num_processes, initializer=configure_logger)

        self._pause_event = threading.Event()
        self._pause_event.set() # Start unpaused.
        self._new_features_event = threading.Event()
        self._new_features_thread = threading.Thread(group=None, target=self._handle_new_labels_and_features, name='schedule-train')
        self._new_features_thread.daemon = True
        self._new_features_thread.start()

        self._new_features_lock = threading.Lock()
        self._outstanding_training_jobs = 0
        self._finish_training_events = [] # Modify under _new_features_lock.

        atexit.register(self.shutdown)

    def shutdown(self):
        self.logger.info('Terminating training pool')
        if self._training_pool:
            self._training_pool.terminate()

    def pause(self):
        self.logger.info('Paused')
        self._pause_event.clear()

    def resume(self):
        self.logger.info('Resumed')
        self._pause_event.set()

    def add_labels(self, labels: Iterable[LabelInfo]):
        # Pass through to storage manager.
        self.storagemanager.add_labels(labels)

        if self._asyncfm and self.feature_names:
            vids = set([l.vid for l in labels])
            # Proactively retrain model once the new features are extracted.
            self.featuremanager.extract_features_async(self.feature_names, vids, callback=lambda: self._new_features_event.set())

    def get_predictions(self, *, vids=None, start=None, end=None, feature_names: Union[str, List[str]]=None, ignore_labeled=False) -> Iterable[PredictionSet]:
        # len wouldn't work if get_vids_with_labels returned a map rather than a list.
        nlabeled = len(self.storagemanager.get_vids_with_labels())
        if not nlabeled:
            return []

        self.logger.info(f'Getting predictions with feature {feature_names}')

        do_predict = lambda: self._predict_model_for_feature(feature_names, vids, start, end, ignore_labeled=ignore_labeled)

        if self._asyncfm and vids is not None:
            # If possible, start feature extraction process. It can happen in parallel with
            # model training process, if we end up training a new one.
            self.featuremanager.extract_features_async(feature_names, vids)

        # Keep trying to get predictions until we have a model that can predict all known classes (minus ignored ones).
        while True:
            # predictions will be None if there isn't an existing model.
            # If predictions doesn't contain a probability for all known classes, the model was trained before we learned about
            # one or more new classes. In this case, should we wait for a model to be trained that does know about all classes?
            predictions = do_predict()
            if predictions is None:
                self.logger.debug('No saved model. Schedule training if necessary, otherwise wait for running task to finish.')
                self._wait_for_model()
                predictions = do_predict()
                if predictions is None:
                    raise RuntimeError('predictions is None even after waiting')

            model_labels = predictions[1]
            all_known_labels = set(self.storagemanager.get_distinct_labels()) - self.ignore_labels
            if all_known_labels - set(model_labels):
                # Try again. Eventually we'll have trained a model that handles all known labels.
                self.logger.info(f'Trying to get predictions again. Used a model that predicts ({set(model_labels)}), but waiting for one that also predicts ({set(all_known_labels) - set(model_labels)})')
                time.sleep(0.5)
                continue

            return probs_to_predictionset(*predictions)

    def _wait_for_model(self):
        with self._new_features_lock:
            wait_event = threading.Event()
            should_train = self._outstanding_training_jobs == 0
            if not should_train:
                # If we're not scheduling a training task, wait for a currently running one to finish.
                # Otherwise, we'll wait for the event in the callback of train_model_async.
                self._finish_training_events.append(wait_event)
        if should_train:
            # There is a race condition between checking when we first tried do_predict and checking the value
            # of _outstanding_training_jobs. It's possible in between a model was saved and _outstanding_training_jobs
            # was decremented, so it would be unnecessary to retrain now.
            # For simplicity, ignore the race for now and always re-train.
            # It's also possible that an existing training task ignores the specified labels. For simplicity now, always retrain.
            self.logger.debug(f'No outstanding model training tasks. Scheduling model training.')
            # If predictions are still none and there are no outstanding training jobs, then schedule one.
            # This could lead to double-training if new features come in around now.
            # Don't call this while holding _new_features_lock or else we'll deadlock.
            # Wait for the event in the callback.
            self._train_model_async(callback=lambda: wait_event.set())
        else:
            self.logger.debug('Training task is running; wait for it to finish')

        wait_event.wait()
