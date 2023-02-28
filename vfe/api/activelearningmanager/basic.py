import logging
import math
import numpy as np
from typing import Iterable, Tuple, Dict

from vfe import features
from vfe import core
from vfe.core.timing import logtime
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager, PredictionSet, LabelInfo
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.storagemanager import VidType, ClipInfoWithPath, ClipInfo, clips_overlap
from .abstractexplorer import AbstractExplorer
from .abstract import AbstractActiveLearningManager, ExploreSet
from .common import align_to_feature

class BasicActiveLearningManager(AbstractActiveLearningManager):
    def __init__(self, featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, explorer: AbstractExplorer, feature_name, rng=None):
        self.featuremanager = featuremanager
        self.modelmanager = modelmanager
        self.videomanager = videomanager
        self.explorer = explorer
        self.feature_name = feature_name
        self.rng = np.random.default_rng(rng)
        self.logger = logging.getLogger(__name__)

    @logtime
    def add_video(self, path, start_time=None, duration=None) -> VidType:
        return self.featuremanager.add_video(path, start_time, duration)

    @logtime
    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        return self.featuremanager.add_videos(video_csv_path)

    @logtime
    def add_labels(self, labels: Iterable[LabelInfo]) -> None:
        self.modelmanager.add_labels(labels)

    @logtime
    def check_label_quality(self, n_splits=5) -> Dict[str, float]:
        # Dictionary: variable -> median value across 5 splits.
        results = self.modelmanager.check_label_quality([self.feature_name], n_splits=n_splits)
        self.logger.debug(f'Median performance across {n_splits} splits: {results}')
        return results

    @logtime
    def ignore_label_in_predictions(self, label) -> None:
        self.modelmanager.ignore_label_in_predictions(label)

    def _expand_clip(self, clip: ClipInfo, t):
        clips = self.videomanager.get_physical_clips_for_expanded_clip(clip, t)
        return [align_to_feature([self.feature_name], clip) for clip in clips]

    def _predict_clips(self, clips: Iterable[ClipInfo], partial_overlap=True):
        vids = [clip.vid for clip in clips]
        predictions: Iterable[PredictionSet] = self.modelmanager.get_predictions(vids=vids, feature_names=[self.feature_name])
        clip_predictions = [[] for _ in clips]
        for prediction in predictions:
            for i, clip in enumerate(clips):
                if prediction.vid != clip.vid:
                    continue

                if partial_overlap:
                    # Include if partial overlap or if the prediction is contained within the clip.
                    include = clips_overlap(clip, prediction)
                else:
                    include = prediction.start_time <= clip.start_time and clip.end_time <= prediction.end_time

                if include:
                    clip_predictions[i].append(prediction)
        return clip_predictions

    @logtime
    def explore(self, k, t, label=None, vids=None) -> ExploreSet:
        self.logger.info(f'explore: k={k}, t={t}, label={label}')
        clips = self.explorer.explore([self.feature_name], self.featuremanager, self.modelmanager, self.videomanager, k, t, label=label, vids=vids)
        explore_clips = [self._expand_clip(clip, t) for clip in clips]
        clip_predictions = self._predict_clips(clips)
        flat_clips = [clip for explore_clip in explore_clips for clip in explore_clip]
        explore_predictions_flattened = self._predict_clips(flat_clips)
        explore_predictions = []
        renest_points = [len(c) for c in explore_clips]
        start_idx = 0
        for end_idx in renest_points:
            explore_predictions.append(explore_predictions_flattened[start_idx:start_idx+end_idx])
            start_idx = start_idx + end_idx
        return ExploreSet(clips, explore_clips, clip_predictions, explore_predictions, [self.feature_name])

    @logtime
    def watch_vid(self, vid, start, end) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        self.logger.info(f'watch_vid: vid={vid}, start={start}, end={end}')
        # Return video fragments from vid between start and end with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath.
        predictions: Iterable[PredictionSet] = self.modelmanager.get_predictions(vids=vid, start=start, end=end, feature_names=[self.feature_name])
        vpath = self.videomanager.get_video_paths([vid])
        return (predictions if predictions is not None else [], vpath)

    @logtime
    def watch_vids(self, vids) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        self.logger.info(f'watch_vids: vids={vids}')
        # Return video fragments from vids with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath for each vid specified.
        vpaths = self.videomanager.get_video_paths(vids)
        predictions = self.modelmanager.get_predictions(vids=vids, feature_names=[self.feature_name])
        return (predictions, vpaths)
