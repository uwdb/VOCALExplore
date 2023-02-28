from collections import defaultdict
from functools import partial
import logging
import threading
from typing import Iterable, Union, List

from vfe import core
from vfe.api.featuremanager import AbstractAsyncFeatureManager
from vfe.api.storagemanager import LabelInfo
from vfe.api.scheduler import AbstractScheduler, Priority, UserPriority
from .abstract import AbstractAsyncModelManager, PredictionSet
from .abstractpytorch import AbstractPytorchModelManager

# BGModelManager could also speed things up by training models on just clips with features
# already extracted, even if they are a subset of all labeled data.
class BackgroundAsyncModelManager(AbstractPytorchModelManager, AbstractAsyncModelManager):
    def __init__(self,
            *args,
            scheduler: AbstractScheduler = None,
            parallel_kfold = True,
            min_trainsize = 5,
            train_labels: List[str] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._asyncfm = isinstance(self.featuremanager, AbstractAsyncFeatureManager)
        self.logger = logging.getLogger(__name__)
        self.scheduler = scheduler
        self.parallel_kfold = parallel_kfold
        self.min_trainsize = min_trainsize
        self.train_labels = train_labels

        self._new_features_lock = threading.Lock()
        self._outstanding_training_jobs = defaultdict(int)
        self._finish_training_events = [] # Modify under _new_features_lock.

        self._newlabels = False
        self._features_trained_on_newlabels = set()

    def pause(self):
        self.logger.info('Paused')
        self._pause_event.clear()

    def resume(self):
        self.logger.info('Resumed')
        self._pause_event.set()

    # TODO: for this to actually be async, we should extract_features_async first then train the model in the callback.
    def train_model_async(self, feature_names: Union[str, List[str]], callback=None, priority=Priority.DEFAULT, only_already_extracted=False, prepare=None):
        self.logger.debug(f'getting all features for labeled vids')
        with self._new_features_lock:
            self._outstanding_training_jobs[core.typecheck.ensure_str(feature_names)] += 1
        labels_and_features, unique_labels = self._get_all_labels_and_features(feature_names, only_already_extracted=only_already_extracted)

        if self.train_labels:
            # Override and train on the labels that were specified at init.
            unique_labels = self.train_labels

        # nunique_labels = len(set(labels_and_features['labels'].to_pylist()))
        if len(labels_and_features) < self.min_trainsize:
            self.logger.debug(f'Skipping train_model_async because too few labels: {len(labels_and_features)}')
            if prepare:
                # This assumes prepare is cheap.
                prepare()
            if callback:
                # This assumes the callback is cheap.
                callback()
            return
        self.logger.debug(f'labels {unique_labels}, vids {len(set(labels_and_features["vid"].to_pylist()))}')
        self.scheduler.schedule(
            'train_model',
            partial(
                self._train_model_for_features_and_labels_base,
                **dict(
                    feature_names=feature_names,
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
                    f1_val=0.2,
                )
            ),
            callback=partial(self._handle_trained_model, callback=callback),
            priority=priority,
            prepare=prepare,
        )

    def check_label_quality_async(self, feature_names: Union[str, List[str]], n_splits=5, callback=None, priority: Priority = Priority.DEFAULT, **check_quality_kwds):
        self.logger.debug(f'Checking label quality for {feature_names}')

        # Check quality once features are extracted (below). This enables multiple features to be extracted at once.
        def evaluate_quality_callback(feature_names, n_splits, callback, priority, vids=None, **check_quality_kwds):
            labels_and_features, unique_labels = self._get_all_labels_and_features(feature_names, vids=vids)
            self.logger.debug(f'feature {feature_names}, labels {unique_labels}, vids {set(labels_and_features["vid"].to_pylist()) if len(labels_and_features) < 1000 else "(too many to print)"}')

            kwds = dict(
                    feature_names=feature_names,
                    n_splits=n_splits,
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
                    device=self.device,
                    **check_quality_kwds,
                )
            if not self.parallel_kfold:
                self.scheduler.schedule(
                    'check_label_quality',
                    partial(
                        self._check_label_quality_base,
                        **kwds,
                    ),
                    callback=callback,
                    priority=priority
                )
            else:
                tasks = self._check_label_quality_base(**kwds, return_tasks=True)
                if tasks is None:
                    callback(None)
                    return
                wait_events = [threading.Event() for _ in tasks]
                aggregated_results = []
                def wait_callback(result, i):
                    aggregated_results.extend(result)
                    wait_events[i].set()
                    for event in wait_events:
                        if not event.is_set():
                            return
                    callback(self.postprocess_eval_split_results(aggregated_results))
                for i, task in enumerate(tasks):
                    self.scheduler.schedule(
                        f'check_quality_piece_{i}: {feature_names}',
                        task,
                        callback=partial(wait_callback, i=i),
                        priority=priority
                    )

        labeled_vids = self.get_vids_with_labels()
        # Specify that we should evaluate quality on only the vids we are getting features for. Otherwise, there is a
        # potential deadlock. This hapens when new labels are added between now and the callback, but the features are
        # not yet extracted. evaluate_quality_callback would try to get those features synchronously, which blocks. However,
        # because the callback is called on the feature processing thread, the feature extraction task could never finish.
        # An alternative could be to dispatch the callback to a different thread. However, the current solution has the
        # benefit of ensuring the same steps for feature evaluation for both sync and async bandit eval.
        self.featuremanager.extract_features_async(
            feature_names,
            labeled_vids,
            callback=partial(evaluate_quality_callback, feature_names=feature_names, vids=labeled_vids, n_splits=n_splits, callback=callback, priority=priority, **check_quality_kwds)
        )

    def _handle_trained_model(self, trained_model_info, callback=None):
        self.logger.debug(f'Saving model with info: {trained_model_info}')
        self.storagemanager.add_model(**trained_model_info)
        with self._new_features_lock:
            self._outstanding_training_jobs[trained_model_info['feature_name']] -= 1
            events = self._finish_training_events
            self._finish_training_events = []
        if callback is not None:
            callback()
        for event in events:
            event.set()

    def add_labels(self, labels: Iterable[LabelInfo]):
        self._newlabels = True
        self._features_trained_on_newlabels = set()

        # Pass through to storage manager.
        self.storagemanager.add_labels(labels)

    def get_predictions(self, *, vids=None, start=None, end=None, feature_names: Union[str, List[str]]=None, ignore_labeled=False, allow_stale_predictions=False, priority: Priority=Priority.DEFAULT) -> Iterable[PredictionSet]:
        feature_names = core.typecheck.ensure_list(feature_names)
        # len wouldn't work if get_vids_with_labels returned a map rather than a list.
        nlabeled = len(self.storagemanager.get_vids_with_labels())
        if not nlabeled:
            return []

        self.logger.info(f'Getting predictions with feature {feature_names}')

        do_predict = lambda: self._predict_model_for_feature(feature_names, vids, start, end, ignore_labeled=ignore_labeled, priority=priority)

        if self._asyncfm and vids is not None:
            # If possible, start feature extraction process. It can happen in parallel with
            # model training process, if we end up training a new one.
            self.featuremanager.extract_features_async(feature_names, vids, priority=priority)

        feature_names_str = core.typecheck.ensure_str(feature_names)
        if (not allow_stale_predictions and self._newlabels and feature_names_str not in self._features_trained_on_newlabels) \
                or (self.storagemanager.get_model_info(feature_names_str) is None):
            self._wait_for_model(feature_names)
            self._features_trained_on_newlabels.add(feature_names_str)

        predictions = do_predict()
        # Keep trying to get predictions until we have a model that can predict all known classes (minus ignored ones).
        # while True:
        #     # predictions will be None if there isn't an existing model.
        #     # If predictions doesn't contain a probability for all known classes, the model was trained before we learned about
        #     # one or more new classes. In this case, should we wait for a model to be trained that does know about all classes?
        #     predictions = do_predict()
        #     if predictions is None:
        #         self.logger.debug('No saved model. Schedule training if necessary, otherwise wait for running task to finish.')
        #         self._wait_for_model(feature_name)
        #         predictions = do_predict()
        #         if predictions is None:
        #             raise RuntimeError('predictions is None even after waiting')

        #     model_labels = predictions[1]
        #     all_known_labels = set(self.storagemanager.get_distinct_labels()) - self.ignore_labels
        #     if all_known_labels - set(model_labels):
        #         # Try again. Eventually we'll have trained a model that handles all known labels.
        #         self.logger.info(f'Trying to get predictions again. Used a model that predicts ({set(model_labels)}), but waiting for one that also predicts ({set(all_known_labels) - set(model_labels)})')
        #         time.sleep(0.5)
        #         continue

        return self._probs_to_predictionset(*predictions)

    def _wait_for_model(self, feature_names: List[str]):
        with self._new_features_lock:
            wait_event = threading.Event()
            # outstanding_training_jobs needs to be feature-specific.
            should_train = self._outstanding_training_jobs[core.typecheck.ensure_str(feature_names)] == 0
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
            self.train_model_async(feature_names, callback=lambda: wait_event.set(), priority=UserPriority.priority)
        else:
            self.logger.debug('Training task is running; wait for it to finish')

        wait_event.wait()
