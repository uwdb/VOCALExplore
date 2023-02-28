from collections import namedtuple
from typing import Iterable, Dict, Union, List

from vfe.api.storagemanager import LabelInfo, ClipSet, LabeledClipSet, VidType

PredictionSet = namedtuple('PredictionSet', ['vid', 'start_time', 'end_time', 'predictions'])

class AbstractModelManager:
    def add_labels(self, labels: Iterable[LabelInfo]):
        raise NotImplementedError

    def get_predictions(self, *, vids=None, start=None, end=None, feature_names: Union[str, List[str]]=None, ignore_labeled=False, allow_stale_predictions=False) -> Iterable[PredictionSet]:
        raise NotImplementedError

    def check_label_quality(self, feature_names: Union[str, List[str]], n_splits=5) -> Dict[str, float]:
        # The index of the series contains the variables.
        # The values of the series contains the median value for each variable across the splits.
        raise NotImplementedError

    def get_label_counts(self, feature_names: Union[str, List[str]]) -> Dict[str, int]:
        raise NotImplementedError

    def get_total_label_time(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_labels_for_clips(self, clipset: ClipSet, full_overlap=True) -> LabeledClipSet:
        raise NotImplementedError

    def get_vids_with_labels(self) -> Iterable[VidType]:
        raise NotImplementedError

    def ignore_label_in_predictions(self, label) -> None:
        raise NotImplementedError

    def pause(self):
        pass

    def resume(self):
        pass

class AbstractAsyncModelManager(AbstractModelManager):
    def train_model_async(self, feature_names:  Union[str, List[str]], callback=None, only_already_extracted=False) -> None:
        # Callback may not be called from the main thread.
        raise NotImplementedError

    def check_label_quality_async(self, feature_names:  Union[str, List[str]], n_splits=5, callback=None) -> None:
        # Callback may not be called from the main thread.
        raise NotImplementedError
