from decimal import Decimal
import itertools
import json
import logging
import numpy as np
import time
from typing import Iterable, Union

from vfe import core
from vfe.api.modelmanager.abstractpytorch import AbstractPytorchModelManager

from vfe.api.storagemanager import AbstractStorageManager, ClipInfoWithPath, AnyClipInfo, LabelInfo, ModelInfo
import vfe.api.storagemanager as vsm
from vfe.api.activelearningmanager import AbstractActiveLearningManager

class AbstractLabeler:
    def get_labels(self, clip_infos: Iterable[AnyClipInfo]) -> Iterable[LabelInfo]:
        raise NotImplementedError

class AbstractContextLabeler:
    def get_labels(self, clip_info: AnyClipInfo, context_clip_infos: Iterable[AnyClipInfo]) -> Iterable[LabelInfo]:
        raise NotImplementedError

class OracleLabeler(AbstractLabeler):
    def __init__(self, oracle_storagemanager: AbstractStorageManager):
        self.oracle_storagemanager = oracle_storagemanager

    def get_labels(self, clipinfos: Iterable[AnyClipInfo]) -> Iterable[LabelInfo]:
        clipset = vsm.clipinfo_to_clipset(clipinfos)
        return vsm.labelset_to_labelinfo(
            self.oracle_storagemanager.get_labels_for_clips_nonaggregated_overlapping(clipset)
        )

class OracleContextLabeler(AbstractContextLabeler):
    def __init__(self, oracle_storagemanager: AbstractStorageManager):
        self.oracle_storagemanager = oracle_storagemanager

    def get_labels(self, clip_info: AnyClipInfo, context_clip_infos: Iterable[AnyClipInfo]) -> Iterable[LabelInfo]:
        # Take whatever labels appear in the specified clip, and assign these labels to the entirety of the context clips.
        clip_labels = vsm.labelset_to_labelinfo(
            self.oracle_storagemanager.get_labels_for_clips_nonaggregated_overlapping(
                vsm.clipinfo_to_clipset([clip_info])
            )
        )
        all_labels = []
        for clip in context_clip_infos:
            for label in clip_labels:
                all_labels.append(LabelInfo(clip.vid, clip.start_time, clip.end_time, label.label))
        return all_labels

class AbstractUser:
    def __init__(self, *, status_file=None):
        self.status_file = status_file

    def peform_action(self, alm: AbstractActiveLearningManager):
        raise NotImplementedError

    def print_status(self, *args):
        if self.status_file is None:
            return
        with open(self.status_file, 'a+') as f:
            print('Status:', *args, file=f)

    def print_clips(self, clips: Iterable[ClipInfoWithPath]):
        if self.status_file is None:
            return
        with open(self.status_file, 'a+') as f:
            print('Clips start', file=f)
            for clip in clips:
                print(*clip, sep='|', file=f)
            print('Clips end', file=f)

    def print_predictions(self, predictions):
        if self.status_file is None:
            return
        with open(self.status_file, 'a+') as f:
            print('Predictions start', file=f)
            for vid_predictions in predictions:
                for prediction in vid_predictions:
                    print(*prediction, sep='|', file=f)
            print('Predictions end', file=f)

    def print_labels(self, labels: Iterable[LabelInfo]):
        if self.status_file is None:
            return
        with open(self.status_file, 'a+') as f:
            print('Labels start', file=f)
            for label in labels:
                print(*label, sep='|', file=f)
            print('Labels end', file=f)

    @staticmethod
    def evaluate_labels(base_sm, base_fm, eval_sm, eval_fm, feature_names, test_vids, ignore_labels=[], groupby_vid=False, sample_from_validation=-1, eval_on_trained=False, trained_mid=None):
        logging.info('*** Evaluating labels.')
        if not eval_on_trained:
            # Train a model with all labels on base.
            train_mm = AbstractPytorchModelManager(base_sm, base_fm)
            for label in ignore_labels:
                train_mm.ignore_label_in_predictions(label)
            train_labels = set(eval_sm.get_distinct_labels()) - set(ignore_labels)
            trained_model_info = train_mm._train_model(feature_names, save=False, labels=train_labels, train_kwargs=dict(f1_val=0.2))
            train_mm = None
            model_info = ModelInfo(trained_model_info['model_type'], trained_model_info['model_path'], trained_model_info['labels'], core.typecheck.ensure_str(feature_names), trained_model_info['f1_threshold'])
        else:
            if trained_mid is None:
                logging.info('*** Performance: 0')
                return {}
            model_info = base_sm.get_model_info_for_mid(trained_mid)

        eval_mm = AbstractPytorchModelManager(eval_sm, eval_fm)
        performance = eval_mm._model_perf(model_info, test_vids, ignore_labels=ignore_labels, groupby_vid=groupby_vid, sample_from_validation=sample_from_validation)
        logging.info(f'*** Performance {performance}')
        return performance

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

class ExploreUser(AbstractUser):
    def __init__(self, k, label_t, watch_t, labeler: Union[AbstractLabeler, AbstractContextLabeler], playback_speed=1, ignore_labels=[], status_file=None):
        super().__init__(status_file=status_file)
        self.k = k
        self.label_t = label_t
        self.watch_t = watch_t
        self.labeler = labeler
        self.playback_speed = playback_speed
        self.sleep_time = self.watch_t / self.playback_speed
        self.ignore_labels = ignore_labels
        self.logger = logging.getLogger(__name__)

    def _explore_step(self, alm: AbstractActiveLearningManager):
        self.print_status('Explore with label None')
        explore_set = alm.explore(self.k, self.label_t, label=None, vids=None)
        # Because we passed in label_t as the context duration, the clips we should label are under
        # the context* parameters rather than the explore* parameters of the explore_set.
        for center_clip, clips, predictions in zip(explore_set.explore_clips, explore_set.context_clips, explore_set.context_predictions):
            self.logger.info(f'Sleeping for {self.sleep_time} while watching clips')
            time.sleep(self.sleep_time)
            if isinstance(self.labeler, AbstractLabeler):
                labels = self.labeler.get_labels(clips)
            elif isinstance(self.labeler, AbstractContextLabeler):
                labels = self.labeler.get_labels(center_clip, clips)
            else:
                raise RuntimeError(f'Unrecognized labeler type: {self.labeler}')

            # Print information about labels.
            self.print_clips(clips)
            self.print_predictions(predictions)
            self.print_labels(labels)
            self.logger.debug('Labeled clips:')
            for clip in clips:
                self.logger.debug(f'({clip.vid}, {clip.start_time:.3f}-{clip.end_time:.3f})')
            for label in labels:
                self.logger.debug(f'Label ({json.dumps(label._asdict(), cls=DecimalEncoder)})')
                if label.label in self.ignore_labels:
                    alm.ignore_label_in_predictions(label.label)
            for prediction in itertools.chain(*predictions):
                self.logger.debug(f'Predictions: {json.dumps(prediction._asdict(), cls=DecimalEncoder)}')

            # Actually add labels.
            alm.add_labels(labels)

        return explore_set.prediction_feature_names

    def perform_action(self, alm: AbstractActiveLearningManager):
        return self._explore_step(alm)

class ExploreLabelUser(AbstractUser):
    # TODO: get rid of all of the duplicated code with ExploreUser (everything except explore_label).
    def __init__(self, explore_label, k, label_t, watch_t, labeler: Union[AbstractLabeler, AbstractContextLabeler], playback_speed=1, ignore_labels=[], status_file=None):
        super().__init__(status_file=status_file)
        self.explore_label = explore_label
        self.k = k
        self.label_t = label_t
        self.watch_t = watch_t
        self.labeler = labeler
        self.playback_speed = playback_speed
        self.sleep_time = self.watch_t / self.playback_speed
        self.ignore_labels = ignore_labels
        self.logger = logging.getLogger(__name__)

        self.step = 0

    def _explore_step(self, alm: AbstractActiveLearningManager):
        if self.step < 10:
            self.print_status('Explore with label None')
            explore_label = None
        else:
            self.print_status(f'Explore with label {self.explore_label}')
            explore_label = self.explore_label
        explore_set = alm.explore(self.k, self.label_t, label=explore_label, vids=None)
        # Because we passed in label_t as the context duration, the clips we should label are under
        # the context* parameters rather than the explore* parameters of the explore_set.
        for center_clip, clips, predictions in zip(explore_set.explore_clips, explore_set.context_clips, explore_set.context_predictions):
            self.logger.info(f'Sleeping for {self.sleep_time} while watching clips')
            time.sleep(self.sleep_time)
            if isinstance(self.labeler, AbstractLabeler):
                labels = self.labeler.get_labels(clips)
            elif isinstance(self.labeler, AbstractContextLabeler):
                labels = self.labeler.get_labels(center_clip, clips)
            else:
                raise RuntimeError(f'Unrecognized labeler type: {self.labeler}')

            # Print information about labels.
            self.print_clips(clips)
            self.print_predictions(predictions)
            self.print_labels(labels)
            self.logger.debug('Labeled clips:')
            for clip in clips:
                self.logger.debug(f'({clip.vid}, {clip.start_time:.3f}-{clip.end_time:.3f})')
            for label in labels:
                self.logger.debug(f'Label ({json.dumps(label._asdict(), cls=DecimalEncoder)})')
                if label.label in self.ignore_labels:
                    alm.ignore_label_in_predictions(label.label)
            for prediction in itertools.chain(*predictions):
                self.logger.debug(f'Predictions: {json.dumps(prediction._asdict(), cls=DecimalEncoder)}')

            # Actually add labels.
            alm.add_labels(labels)

        return explore_set.prediction_feature_names

    def perform_action(self, alm: AbstractActiveLearningManager):
        self.step += 1
        return self._explore_step(alm)
