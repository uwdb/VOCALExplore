import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
from pathlib import Path
import rpyc
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer
import uuid
from typing import List, Tuple

from vfe.api.activelearningmanager import ExploreSet
from vfe.api.storagemanager import LabelInfo
from vfe import core
from vfe.utils.create_managers import get_alm

logger = logging.getLogger(__name__)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@rpyc.service
class VOCALExploreService(rpyc.Service):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alm = get_alm(config_path)
        self._cached_predictions = defaultdict(dict)
        self._cached_clip_info = defaultdict(dict)

    def on_connect(self, conn):
        logger.info("Connected")

    @rpyc.exposed
    def add_video(self, video_path, start_time):
        logger.info('Adding video')
        self.alm.add_video(video_path, start_time)

    @rpyc.exposed
    def get_videos(self, limit=None):
        logger.info('Getting videos')
        vpaths_and_thumbnails = self.alm.get_videos(limit=limit, thumbnails=True)
        return vpaths_and_thumbnails

    def _prune_context_clips(self, clips: ExploreSet):
        for i in range(len(clips.explore_clips)):
            vid_to_keep = clips.explore_clips[i].vid
            context_clips_to_keep = [context_clip for context_clip in clips.context_clips[i] if context_clip.vid == vid_to_keep]
            clips.context_clips[i] = context_clips_to_keep
        return clips

    def _postprocess_clips(self, clips: ExploreSet):
        # Until we figure out how to expire keys, only keep around 1.
        self._cached_predictions.clear()
        self._cached_clip_info.clear()

        clips = self._prune_context_clips(clips)
        explore_id = str(uuid.uuid4())
        vid_info = {
            clips.explore_clips[i].vid: {
                # Assumes that we have one context clip per explore clip,
                # which is true as long as we don't return context across videos.
                'context_predictions': clips.context_predictions[i][0],
            }
            for i in range(len(clips.explore_clips))
        }
        self._cached_predictions[explore_id] = vid_info

        clip_info = {
            clips.explore_clips[i].vid: {
                # Assumes that we have one context clip per explore clip,
                # which is true as long as we don't return context across videos.
                "thumbpath": clips.context_clips[i][0].thumbpath,
                "vpath": clips.context_clips[i][0].vpath,
            }
            for i in range(len(clips.explore_clips))
        }
        self._cached_clip_info[explore_id] = clip_info

        assert len(self._cached_predictions[explore_id]) == len(self._cached_clip_info[explore_id])

        logger.debug(f'Caching predictions for vids {list(self._cached_predictions.keys())}')
        return {
            'explore_id': explore_id,
            'prediction_feature_names': clips.prediction_feature_names,
        }

    @rpyc.exposed
    def explore(self, B, t, label=None):
        logger.info('Explore')
        clips: ExploreSet = self.alm.explore(B, t, label=label)
        return self._postprocess_clips((clips))

    @rpyc.exposed
    def get_vids(self, explore_id, start, end) -> Tuple[List[int], int]:
        """Returns vids between start and end, along with the total number of vids."""
        all_vids = sorted(list(self._cached_clip_info[explore_id].keys()))
        return all_vids[start:end], len(all_vids)

    @rpyc.exposed
    def get_predictions(self, explore_id, vid):
        if vid not in self._cached_predictions[explore_id]:
            logger.exception(f'Error: expected to find vid {vid} in cached predictions, but only have {list(self._cached_predictions[explore_id].keys())}')
            return {}

        return [
            ps._asdict() for ps in
            self._cached_predictions[explore_id][vid]['context_predictions']
        ]

    @rpyc.exposed
    def get_clip_info(self, explore_id, vids):
        return json.dumps([
            {
                'vid': vid,
                **self._cached_clip_info[explore_id][vid]
            }
            for vid in vids
        ], cls=CustomEncoder)

    @rpyc.exposed
    def get_labels(self, vid):
        logger.info("Getting labels")
        labels = list(self.alm.get_labels([vid]))

        logger.debug([label._asdict() for label in labels])
        return {
            'labels': [label._asdict() for label in labels],
        }

    @rpyc.exposed
    def get_all_labels(self):
        logger.info("Getting all labels")
        return self.alm.get_unique_labels()

    @rpyc.exposed
    def add_label(self, label_dict):
        logger.info("Adding label")
        label_info = LabelInfo(
            lid=None,
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label']
        )
        self.alm.add_labels([label_info])

    @rpyc.exposed
    def update_label(self, label_dict):
        logger.info("Updating label")
        label_info = LabelInfo(
            lid=label_dict['lid'],
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label'],
        )
        self.alm.update_labels([label_info])

    @rpyc.exposed
    def delete_label(self, label_dict):
        logger.info(f"Deleting label {label_dict['lid']}")
        label_info = LabelInfo(
            lid=label_dict['lid'],
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label'],
        )
        self.alm.delete_labels([label_info])

    @rpyc.exposed
    def search_videos(self, date_range=None, labels=None, predictions=None, prediction_confidence=None):
        logger.info(f"Searching for videos with filters: date_range={date_range}, label={labels}, prediction_confidence={prediction_confidence}")
        if predictions:
            assert prediction_confidence is not None

        clips = self.alm.search_videos(date_range=date_range, labels=labels, predictions=predictions, prediction_confidence=prediction_confidence)
        return self._postprocess_clips((clips))

    @rpyc.exposed
    def debug_reset_annotations(self):
        logger.info("Resetting annotations")
        self.alm.videomanager.reset_annotations()


if __name__ == "__main__":
    core.logging.configure_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config-path', default='/home/maureen/VOCALExplore/server/configs/r3d.yaml')
    args = ap.parse_args()

    config_path = args.config_path
    # service = classpartial(VOCALExploreService, config_path=config_path)
    service = VOCALExploreService(config_path=config_path)
    t = ThreadedServer(
        service,
        port=os.environ.get('SERVER_PORT', 8890),
        protocol_config={
            "allow_public_attrs": True,
        }
    )
    t.start()
