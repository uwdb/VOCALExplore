import logging
from typing import Iterable, Union, List

from vfe import models
from vfe.api.featuremanager import AbstractAsyncFeatureManager
from vfe.api.storagemanager import LabelInfo
from .abstract import PredictionSet
from .abstractpytorch import AbstractPytorchModelManager, probs_to_predictionset

class BasicModelManager(AbstractPytorchModelManager):
    def __init__(self,
            *args,
            feature_name=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._asyncfm = isinstance(self.featuremanager, AbstractAsyncFeatureManager)
        self.logger = logging.getLogger(__name__)
        self.feature_name = feature_name

        # Training status.
        self._new_labels = set()
        self._new_ignored = False

    def add_labels(self, labels: Iterable[LabelInfo]):
        # Pass through to storage manager.
        # Don't proactively (re)train model.
        self.storagemanager.add_labels(labels)
        self._new_labels |= set([l.label for l in labels])

        if self._asyncfm and self.feature_name:
            vids = set([l.vid for l in labels])
            self.featuremanager.extract_features_async(self.feature_name, vids)

    def ignore_label_in_predictions(self, label) -> None:
        super().ignore_label_in_predictions(label)
        self._new_ignored = True

    def get_predictions(self, *, vids=None, start=None, end=None, feature_names: Union[str, List[str]]=None, ignore_labeled=False) -> Iterable[PredictionSet]:
        # _has_any_labels isn't accurate if the only labeled clips have labels
        # in ignore_labels. Ignore this edge case for now.
        if not self._has_any_labels:
            self.logger.info('get_predictions called when there are no labeled clips')
            return []

        self.logger.debug(f'Getting predictions with feature {feature_names}')

        if len(self._new_labels - self.ignore_labels) or self._new_ignored:
            # Only train a new model if the new labels aren't in the ignored set.
            self._train_model(feature_names)
            self._new_labels = set()
            self._new_ignored = False

        prediction_results = self._predict_model_for_feature(feature_names, vids, start, end, ignore_labeled=ignore_labeled)
        if prediction_results is None:
            return []

        model_labels = prediction_results[1]
        all_known_labels = set(self.storagemanager.get_distinct_labels()) - self.ignore_labels
        if all_known_labels - set(model_labels):
            raise RuntimeError(f'Model does not predict all known labels (missing ({set(all_known_labels) - set(model_labels)})). This is unexpected since we train a new model whenever new labels come in.')
        return probs_to_predictionset(*prediction_results)

    @property
    def _has_any_labels(self):
        return len(self._new_labels) or len(self.storagemanager.get_vids_with_labels())





# class ParallelBasicModelManager(BasicModelManager):
#     def __init__(self, *args, num_workers=1, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_workers = num_workers
#         # Cannot re-initialize CUDA in forced subprocess. To use CUDA with multiprocessing, you must use 'spawn' start method.
#         self._ctx = torch.multiprocessing.get_context('spawn')
#         self._pool = self._ctx.Pool(num_workers)
#         # atexit.register(self._pool.close)

#     def __del__(self):
#         if self._pool:
#             self._pool.close()

#     @staticmethod
#     def get_results(idxs, trainer, X, y, labels):
#         train_idx, test_idx = idxs
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#         return trainer.train_and_evaluate_pytorch_model(X_train, y_train, X_test, y_test, labels=labels, device='cpu')

#     def check_label_quality(self, feature_name, n_splits=5) -> Dict[str, float]:
#         # The index of the series contains the variables.
#         # The values of the series contains the median value for each variable across the splits.
#         labels_and_features, unique_labels = self._get_all_labels_and_features(feature_name)
#         trainer = models.AbstractStrategy(
#             outdir=None,
#             training_info=models.BaseTrainingInfo(name=None, label_col=None, train_split=None, eval_split=None, feature=feature_name),
#             model=self.model_type,
#             save_models=False,
#             batch_size=self.batch_size,
#             learning_rate=self.learningrate,
#             epochs=self.epochs,
#             budgets=[]
#         )
#         X = np.vstack(labels_and_features['feature'].to_numpy())
#         y = labels_and_features['labels'].to_numpy()
#         self._warn_for_unlabeled(y, 'check_label_quality')
#         kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
#         results_fn = functools.partial(self.get_results, trainer=trainer, X=X, y=y, labels=unique_labels)
#         results = self._pool.map(results_fn, kf.split(X))

#         results_df = pd.DataFrame(results)
#         return results_df.median().to_dict()
