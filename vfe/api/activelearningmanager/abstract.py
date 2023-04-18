from typing import Iterable, Tuple, List, NamedTuple

from vfe.api.modelmanager import PredictionSet
from vfe.api.storagemanager import VidType, ClipInfo, ClipInfoWithPath, LabelInfo

class ExploreSet(NamedTuple):
    explore_clips: Iterable[ClipInfo]
    context_clips: Iterable[Iterable[ClipInfoWithPath]]
    explore_predictions: Iterable[Iterable[PredictionSet]]
    context_predictions: Iterable[Iterable[Iterable[PredictionSet]]]
    prediction_feature_names: List[str]

class AbstractActiveLearningManager:
    def add_video(self, path, start_time=None, duration=None) -> VidType:
        raise NotImplementedError

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        raise NotImplementedError

    def add_labels(self, labels: Iterable[LabelInfo]) -> None:
        raise NotImplementedError

    # def check_label_quality(self, n_splits=5) -> Dict[str, float]:
    #     # Dictionary: variable -> median value across 5 splits.
    #     raise NotImplementedError

    def ignore_label_in_predictions(self, label) -> None:
        # Models used to predict labels shouldn't include this label as a possible class.
        raise NotImplementedError

    def explore(self, k, t, label=None, vids=None) -> ExploreSet:
        # Return k video fragments of duration t.
        # Returns: (picked short clips, clips with surrounding context, predictions for each clip)
        # If vids is None, consider all vids.
        # If label == None, fragments are chosen such that labeling them
        #  is expected to improve predictions.
        # If label != None, should fragments be chosen such that they:
        #   (a) have high probability of having that label?
        #   (b) labeling them will likely improve predictions on that label?
        raise NotImplementedError

    # For watch functions: how to distinguish predicted labels vs. user-provided labels?

    def watch_vid(self, vid, start, end) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        # Return video fragments from vid between start and end with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath.
        raise NotImplementedError

    def watch_vids(self, vids) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        # Return video fragments from vids with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath for each vid specified.
        raise NotImplementedError
