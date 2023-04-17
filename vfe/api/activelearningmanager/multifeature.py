from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import logging
import math
import numpy as np
import random
from scipy import stats
import statistics
import threading
import time
from typing import Iterable, Tuple, List, Dict

from vfe.core.timing import logtime
from vfe.api.featuremanager import AbstractAsyncFeatureManager
from vfe.api.modelmanager import AbstractAsyncModelManager, PredictionSet, LabelInfo
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.storagemanager import VidType, ClipInfoWithPath, ClipInfo, clips_overlap
from vfe.api.scheduler import AbstractScheduler, UserPriority, ChorePriority
from .abstractexplorer import AbstractExplorer
from .abstract import AbstractActiveLearningManager, ExploreSet
from .common import align_to_feature
from .risingbandit import BanditTypes
from vfe.api.activelearningmanager import explorers

class TaskTracker:
    def __init__(self):
        self.taskid = 0
        self.task_counts = defaultdict(int)
        self.callbacks = {}
        self.task_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def get_new_taskid(self):
        with self.task_lock:
            taskid = self.taskid
            self.taskid += 1
        return taskid

    def get_last_taskid(self):
        with self.task_lock:
            return self.taskid - 1

    def wait_for_last_task(self):
        taskid = self.get_last_taskid()
        self.wait_for_task(taskid)

    def _modify_task(self, taskid, delta):
        with self.task_lock:
            self.task_counts[taskid] += delta
            if self.task_counts[taskid] == 0:
                self.logger.debug(f'Clearing task {taskid}')
                self.task_counts.pop(taskid)
                event = self.callbacks.pop(taskid, None)
                if event is not None:
                    event.set()

    def add_task(self, taskid):
        self._modify_task(taskid, 1)

    def remove_task(self, taskid):
        self._modify_task(taskid, -1)

    def wait_for_task(self, taskid):
        with self.task_lock:
            if taskid not in self.task_counts:
                return
            wait_event = threading.Event()
            self.callbacks[taskid] = wait_event
        self.logger.debug(f'Waiting for task {taskid} to finish')
        wait_event.wait()

class AbstractFeatureSelectionStrategy:
    def get_best_features(self, alm) -> List[str]:
        raise NotImplementedError

    def get_candidate_features(self, alm) -> List[str]:
        return alm.feature_names

    def step(self, alm) -> None:
        pass

class WaitFeatureSelectionStrategy(AbstractFeatureSelectionStrategy):
    def get_best_features(self, alm) -> List[str]:
        if len(alm.feature_names) == 1:
            return [alm.feature_names[0]]

        if alm._has_new_labels():
            alm.logger.debug(f'Checking feature quality before picking best feature')
            alm._check_feature_quality()

        with alm.feature_to_perf_lock:
            items = list(alm.feature_to_perf.items())
            random.shuffle(items)
            best_feature, best_val = items[0]
            for alt_feature, alt_val in items[1:]:
                best_is_better = self._is_a_greater_than_b_latest(best_val, alt_val)
                if best_is_better == 0:
                    best_feature, best_val = alt_feature, alt_val
        return [best_feature]

    def _is_a_greater_than_b_latest(self, a, b):
        assert len(a) == len(b) # This can actually be false.
        if not a:
            return -1
        return 1 if a[-1] > b[-1] else 0

class ConcatFeatureSelectionStrategy(AbstractFeatureSelectionStrategy):
    def get_best_features(self, alm):
        return alm.feature_names

class RisingBanditFeatureSelectionStrategy(AbstractFeatureSelectionStrategy):
    def __init__(self, bandit_type: BanditTypes, *args, **kwargs):
        self.bandit = bandit_type.get_bandit(*args, **kwargs)

    def get_best_features(self, alm) -> List[str]:
        candidates = self.bandit.candidates()
        if len(candidates) == 1:
            return [candidates[0]]
        else:
            return [random.choice(self.bandit.candidates())]

    def get_candidate_features(self, alm) -> List[str]:
        return self.bandit.candidates()

    def step(self, alm) -> None:
        if len(self.get_candidate_features(alm)) > 1:
            self.bandit.step(alm)

class FeatureEvalStrategy(Enum):
    WAIT = 'wait'
    CONCAT = 'concat'
    RISINGBANDIT = 'risingbandit'

    def get_selector(self, strategy_kwargs):
        if self == FeatureEvalStrategy.WAIT:
            return WaitFeatureSelectionStrategy(**strategy_kwargs)
        elif self == FeatureEvalStrategy.CONCAT:
            return ConcatFeatureSelectionStrategy(**strategy_kwargs)
        elif self == FeatureEvalStrategy.RISINGBANDIT:
            return RisingBanditFeatureSelectionStrategy(**strategy_kwargs)
        else:
            assert False, f'Unknown feature eval strategy {self}'

class MultiFeatureActiveLearningManager(AbstractActiveLearningManager):
    def __init__(self, featuremanager: AbstractAsyncFeatureManager, modelmanager: AbstractAsyncModelManager, videomanager: AbstractVideoManager, explorer: AbstractExplorer, feature_names: Iterable[str], scheduler: AbstractScheduler, rng=None, strategy: FeatureEvalStrategy=None, strategy_kwargs: Dict = {}, eager_feature_extraction_labeled=False, eager_model_training=False, eager_feature_extraction_unlabeled=False, explore_label_threshold=-1, return_predictions=True, thumbnail_dir=None):
        assert len(feature_names), f'Must specify at least one feature; {feature_names}'

        self.featuremanager = featuremanager
        self.modelmanager = modelmanager # Shouldn't be doing background work without direction.
        self.videomanager = videomanager
        self.explorer = explorer
        self.label_explorer = explorers.LabelUncertaintyExplorer(threshold=explore_label_threshold)
        self.feature_names = feature_names
        self.scheduler = scheduler
        self.feature_selector: AbstractFeatureSelectionStrategy = strategy.get_selector(strategy_kwargs)
        self.return_predictions = return_predictions
        self.thumbnail_dir = thumbnail_dir
        self.step = 0

        self.feature_to_perf_lock = threading.Lock()
        self.feature_to_perf = {feature: [] for feature in self.feature_names}
        self._feature_perf_tasks = TaskTracker()

        self.feature_to_newlabels = defaultdict(bool)

        self.rng = np.random.default_rng(rng)
        self.logger = logging.getLogger(__name__)

        # If there are multiple features, always expand to the first one which is
        # used to align the rest. This way if the best feature changes, the labels
        # always align even if the features have different lengths.
        # This fixes a bug where when clip was labeled first then we switched to r3d,
        # the labels on clip were too short and none covered an entire r3d feature vector.
        self._features_to_align_to = [self.feature_names[0]]

        self.eager_feature_extraction_labeled = eager_feature_extraction_labeled
        self.eager_model_training = eager_model_training

        if eager_feature_extraction_unlabeled:
            self._eager_extraction_thread = threading.Thread(group=None, target=self._sample_vids_to_extract, name='eager-feature-extract')
            self._eager_extraction_thread.daemon = True # True because infinite loop never joins.
            self._eager_extraction_thread.start()

        self.in_explore = False

        self._labeled_vids_missing_features = set()

        # Track t_user.
        self._t_user = deque(maxlen=10)
        self._last_user_time = None

        # Track model training time.
        # Unify tracking the total time across features, but track start/end times per-feature
        # to enable easier duration tracking.
        self._t_m_lock = threading.Lock()
        self._t_m = deque(maxlen=10)
        self._start_t_m_by_feature = defaultdict(float)

        # Track the number of outstanding vids that were returned from Explore but not labeled.
        self._outstanding_explore_vids = set()

        # Track the number of outstanding vids where model training should be started.
        self._start_modeltrain_cutoff = -1

    def _sample_vids_to_extract(self):
        # Wait one second after startup to let more important tasks get scheduled if necessary.
        time.sleep(1)
        while True:
            time.sleep(1)

            # Don't potentially slow down a user-facing call.
            if self.in_explore:
                continue

            # Because these tasks will get queued with low priority, it's not the worst thing
            # if we end up scheduling behind other tasks. For now don't schedule too much ahead
            # in case we want flexibility in picking vids in a more intelligent way.
            if self.scheduler.tasks_are_waiting():
                continue

            feature_names = self.feature_selector.get_candidate_features(self)
            # Pick a random feature to use to get vids. Because we schedule all candidates at once,
            # we'll hopefully pick vids that are missing all features.
            vids_with_features, vids_without_features = self.featuremanager.get_extracted_features_info(random.choice(feature_names))
            if not vids_without_features:
                continue

            size = min(10, len(vids_without_features))
            vids_to_extract = self.rng.choice(np.array([*vids_without_features]), size=size, replace=False)
            self.logger.debug(f'Eagerly extracting features from vids {vids_to_extract}')
            for feature_name in feature_names:
                self.featuremanager.extract_features_async(feature_name, vids_to_extract, priority=ChorePriority.priority)

    @logtime
    def add_video(self, path, start_time=None, duration=None) -> VidType:
        return self.featuremanager.add_video(path, start_time, duration, thumbnail_dir=self.thumbnail_dir)

    @logtime
    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        return self.featuremanager.add_videos(video_csv_path)

    @logtime
    def get_videos(self, limit=None, thumbnails=False) -> List[List[str]]:
        return [vid_path[1:] for vid_path in self.videomanager.get_video_paths(vids=None, thumbnails=thumbnails)][:limit]

    @logtime
    def ignore_label_in_predictions(self, label) -> None:
        self.modelmanager.ignore_label_in_predictions(label)

    def _check_feature_quality(self):
        taskid = self._feature_perf_tasks.get_new_taskid()
        for feature_name in self.feature_names:
            self._feature_perf_tasks.add_task(taskid)
            callback = partial(self._handle_check_model_quality, feature_name=feature_name, taskid=taskid)
            self.modelmanager.check_label_quality_async(feature_name, n_splits=3, callback=callback)
        self._feature_perf_tasks.wait_for_task(taskid)

    def _has_new_labels(self):
        for has_new in self.feature_to_newlabels.values():
            if has_new:
                return True
        return False

    def _is_a_greater_than_b_ttest(self, a, b):
        # stat is positive when sample mean of a is greater than the sample mean of b.
        # stat is negative when sample mean of a is less than the sample mean of b.
        stat, pval = stats.ttest(a, b, rng=self.rng)
        if pval < 0.05:
            return 1 if stat > 0 else 0
        else:
            return -1

    def _get_best_features(self) -> List[str]:
        return self.feature_selector.get_best_features(self)

    def _handle_check_model_quality(self, quality, feature_name=None, taskid=None):
        quality_metric = 'test_mAP'
        if quality is not None:
            # May not be called on main thread.
            with self.feature_to_perf_lock:
                if quality_metric in quality:
                    self.feature_to_perf[feature_name].append(quality[quality_metric])
                self.logger.debug(f'Feature to perf: {self.feature_to_perf}')

        self._feature_perf_tasks.remove_task(taskid)

    def _update_userinteraction_stats(self):
        now = time.perf_counter()
        self._t_user.append(now - self._last_user_time)
        self._last_user_time = now
        self.logger.debug(f't_user: {self._t_user}')

    def _update_modeltrain_start(self, feature_name):
        with self._t_m_lock:
            self._start_t_m_by_feature[feature_name] = time.perf_counter()

    def _set_modeltrain_cutoff_info(self, explore_vids):
        self._outstanding_explore_vids = set(explore_vids)
        B = len(explore_vids)

        # If there is no user-interaction information yet, assume model training
        # will be fast because there are no labels. There may even be few enough
        # labels that we don't try to train a model.
        # Set the value to train a model after the second-to-last label.
        if not self._t_user or not self._t_m:
            self.logger.debug('Missing t_user or t_m; assuming that model training is fast and setting cutoff to 1')
            self._start_modeltrain_cutoff = 1
            return

        avg_t_user = statistics.median(self._t_user)
        with self._t_m_lock:
            avg_t_m = statistics.median(self._t_m)

        train_after_label = B - math.ceil(avg_t_m / avg_t_user)
        if train_after_label > 0:
            # Start after cutoff is the reverse of train_after_label because we
            # eliminate vids from the outstanding set.
            self._start_modeltrain_cutoff = B - train_after_label
        else:
            # Flag that we should train over labels collected in the past iteration
            # while the user is labeling the first item of a new iteration.
            self._start_modeltrain_cutoff = -1

        self.logger.debug(f't_m: {avg_t_m}; t_user: {avg_t_user} train_after_label={train_after_label}; self._start_modeltrain_cutoff={self._start_modeltrain_cutoff}')

    def _should_train_after_label_for_vids(self, vids: set):
        self._outstanding_explore_vids -= vids
        self.logger.debug(f'Remaining vids: {len(self._outstanding_explore_vids)}')
        return len(self._outstanding_explore_vids) == self._start_modeltrain_cutoff

    def _start_train_model(self):
        def _model_callback(feature_name):
            # This assumes that there is one model training task per-feature at a time.
            now = time.perf_counter()
            with self._t_m_lock:
                start = self._start_t_m_by_feature.pop(feature_name)
                self._t_m.append(now - start)
                self.logger.debug(f't_m: {self._t_m}')

        def _feature_callback(feature_name, vids):
            self.logger.debug(f'Finished async feature extraction for feature {feature_name}, vids {vids}')
            # Only train over already-extracted vids.
            # Otherwise, there could be a race where something else gets labeled before train_model_async, and
            # then we end up waiting for the new features.
            self.modelmanager.train_model_async(
                feature_name,
                only_already_extracted=True,
                callback=partial(_model_callback, feature_name=feature_name),
            )

        vids = self._labeled_vids_missing_features
        self._labeled_vids_missing_features = set()
        for feature in self.feature_selector.get_candidate_features(self):
            self.logger.debug(f'Queueing feature extraction for {feature}, vids {vids}')
            if self.eager_model_training:
                callback = partial(_feature_callback, feature_name=feature, vids=vids)
            else:
                callback = None

            # We count model training as starting when we get the features needed to train.
            # Otherwise we will underestimate the time to get a model and won't be ready for the next explore call.
            self.featuremanager.extract_features_async(feature, vids, callback, prepare=partial(self._update_modeltrain_start, feature_name=feature),
)

    @logtime
    def add_labels(self, labels: Iterable[LabelInfo]) -> None:
        self._update_userinteraction_stats()

        for feature in self.feature_names:
            self.feature_to_newlabels[feature] = True
        # Model manager doesn't do any work proactively.
        # Once features are extracted, we'll schedule to proactively train/evaluate models.
        self.modelmanager.add_labels(labels)

        if self.eager_feature_extraction_labeled:
            new_vids = set([l.vid for l in labels])
            self._labeled_vids_missing_features |= new_vids
            if self._should_train_after_label_for_vids(new_vids):
                self._start_train_model()

    def _expand_clip(self, feature_names: List[str], clip: ClipInfo, t):
        clips = self.videomanager.get_physical_clips_for_expanded_clip(clip, t)
        return [align_to_feature(self._features_to_align_to, clip) for clip in clips]

    def _predict_clips(self, feature_names: List[str], clips: Iterable[ClipInfo], partial_overlap=True):
        vids = [clip.vid for clip in clips]
        if self.step > 2:
            predictions: Iterable[PredictionSet] = self._do_get_predictions(feature_names, lambda: self.modelmanager.get_predictions(vids=vids, feature_names=feature_names, allow_stale_predictions=self.eager_model_training, priority=UserPriority.priority))
        else:
            predictions = []
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

    def _do_get_predictions(self, feature_names: List[str], do_predict):
        # if self.feature_eval_strategy == FeatureEvalStrategy.WAIT \
        #         and self.feature_to_newlabels[feature_name]:
        #     wait = threading.Event()
        #     self.modelmanager.train_model_async(feature_name, lambda: wait.set(), priority=Priority.USER)
        #     self.feature_to_newlabels[feature_name] = False
        #     wait.wait()
        return do_predict()

    @logtime
    def explore(self, k, t, label=None, vids=None) -> ExploreSet:
        self.in_explore = True
        # This isn't safe because explore could need the results of a low-priority task.
        # In the experiments we'll only actually suspend when we're doing everything eagerly,
        # so hopefully this won't be an issue for now.
        # If the suspend_lowp flag was not set, this will be a noop.
        self.scheduler.suspend_lowp()
        self.step += 1

        feature_names = self._get_best_features()
        self.logger.info(f'explore (feature {feature_names}): k={k}, t={t}, label={label}')
        if label:
            clips = self.label_explorer.explore(feature_names, self.featuremanager, self.modelmanager, self.videomanager, k, t, label=label, vids=vids)
        else:
            clips = self.explorer.explore(feature_names, self.featuremanager, self.modelmanager, self.videomanager, k, t, label=label, vids=vids)
        if self.return_predictions:
            explore_clips = [self._expand_clip(feature_names, clip, t) for clip in clips]
            clip_predictions = self._predict_clips(feature_names, clips)
            flat_clips = [clip for explore_clip in explore_clips for clip in explore_clip]
            explore_predictions_flattened = self._predict_clips(feature_names, flat_clips)
            explore_predictions = []
            renest_points = [len(c) for c in explore_clips]
            start_idx = 0
            for end_idx in renest_points:
                explore_predictions.append(explore_predictions_flattened[start_idx:start_idx+end_idx])
                start_idx = start_idx + end_idx
        else:
            explore_clips = [[clip] for clip in clips]
            clip_predictions = [[] for clip in clips]
            explore_predictions = [[] for clip in clips]

        # Wait to step until end to avoid slowing down any of the time-critical tasks.
        self.feature_selector.step(self)

        # If the suspend_lowp flag was not set, this will be a noop.
        self.scheduler.resume_lowp()
        self.in_explore = False

        self._last_user_time = time.perf_counter()

        # Set up cutoff for when we will train this iteration.
        self._set_modeltrain_cutoff_info([clip.vid for clip in clips])

        # Special case: training takes longer than the expected duration of an iteration.
        # Start training now on labels collected during the past iteration.
        if self._start_modeltrain_cutoff == -1:
            self._start_train_model()

        return ExploreSet(clips, explore_clips, clip_predictions, explore_predictions, feature_names)

    @logtime
    def watch_vid(self, vid, start, end) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        self.logger.info(f'watch_vid: vid={vid}, start={start}, end={end}')
        # Return video fragments from vid between start and end with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath.
        feature_names = self._get_best_features()
        predictions: Iterable[PredictionSet] = self._do_get_predictions(feature_names, lambda: self.modelmanager.get_predictions(vids=vid, start=start, end=end, feature_name=feature_names))
        vpath = self.videomanager.get_video_paths([vid])

        self._last_user_time = time.perf_counter()
        return (predictions if predictions is not None else [], vpath)

    @logtime
    def watch_vids(self, vids) -> Tuple[Iterable[PredictionSet], Iterable[Tuple[VidType, str]]]:
        self.logger.info(f'watch_vids: vids={vids}')
        # Return video fragments from vids with predicted labels.
        # If there are no labeled segments, then predicted labels will be None.
        # First item of return tuple contains predictions.
        # Second item of return tuple contains mapping from vid -> vpath for each vid specified.
        vpaths = self.videomanager.get_video_paths(vids)
        feature_names = self._get_best_features()
        predictions = self._do_get_predictions(feature_names, lambda: self.modelmanager.get_predictions(vids=vids, feature_name=feature_names))

        self._last_user_time = time.perf_counter()
        return (predictions, vpaths)
