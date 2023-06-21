import argparse
from collections import defaultdict
import datetime
from decimal import Decimal
from functools import partial
import json
import logging
import os
from pathlib import Path
import queue
import rpyc
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer
import uuid
import threading
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
        if isinstance(obj, Decimal):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


@rpyc.service
class VOCALExploreService(rpyc.Service):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alm = get_alm(config_path)
        self._cached_predictions = defaultdict(dict)
        self._cached_clip_info = defaultdict(dict)
        self._cached_prediction_features = defaultdict(list)
        self._cache = defaultdict(dict)

        # Thread for tasks that shouldn't block returning from a function.
        self._lowp_task_queue = queue.SimpleQueue()
        self._lowp_thread = threading.Thread(group=None, target=self._poll_for_tasks, name='low-priority-tasks')
        self._lowp_thread.daemon = True
        self._lowp_thread.start()

    def _poll_for_tasks(self):
        while True:
            name, fn = self._lowp_task_queue.get()
            logger.info(f"Executing low-p task: {name}")
            fn()

    def on_connect(self, conn):
        logger.info("Connected")

    @rpyc.exposed
    def add_video(self, video_path, start_time):
        logger.info('Adding video')
        self.alm.add_video(video_path, start_time)

    def _progress_prepare_fn(self, *args, lock=None, key=None):
        with lock:
            if "total" not in self._cache[key]:
                self._cache[key]["total"] = 0
            self._cache[key]["total"] += 1

    def _progress_callback_fn(self, *args, lock=None, key=None):
        with lock:
            if "done" not in self._cache[key]:
                self._cache[key]["done"] = 0
            self._cache[key]["done"] += 1

    @rpyc.exposed
    def add_videos(self, base_dir):
        load_key = str(uuid.uuid4())
        lock = threading.Lock()
        self._cache[load_key]["lock"] = lock

        self._lowp_task_queue.put((
            "add_videos",
            partial(
                self.alm.featuremanager.add_videos_from_dir,
                base_dir=base_dir,
                thumbnail_dir=self.alm.thumbnail_dir,
                prepare_fn=partial(self._progress_prepare_fn, lock=lock, key=load_key),
                callback_fn=partial(self._progress_callback_fn, lock=lock, key=load_key),
            )
        ))
        return json.dumps({"key": load_key})

    @rpyc.exposed
    def get_videos(self, limit=None):
        logger.info('Getting videos')
        vpaths_and_thumbnails = self.alm.get_videos(limit=limit, thumbnails=True)
        return vpaths_and_thumbnails

    def _prune_context_clips(self, clips: ExploreSet):
        if not clips.context_clips:
            return clips
        for i in range(len(clips.explore_clips)):
            vid_to_keep = clips.explore_clips[i].vid
            context_clips_to_keep = [context_clip for context_clip in clips.context_clips[i] if context_clip.vid == vid_to_keep]
            clips.context_clips[i] = context_clips_to_keep
        return clips

    def _postprocess_clips(self, clips: ExploreSet, cache_predictions=True):
        # Until we figure out how to expire keys, only keep around 1.
        self._cached_predictions.clear()
        self._cached_clip_info.clear()
        self._cached_prediction_features.clear()

        clips = self._prune_context_clips(clips)
        explore_id = str(uuid.uuid4())
        if cache_predictions:
            vid_info = {
                clips.explore_clips[i].vid: {
                    # Assumes that we have one context clip per explore clip,
                    # which is true as long as we don't return context across videos.
                    'context_predictions': clips.context_predictions[i][0] if clips.context_predictions else [],
                }
                for i in range(len(clips.explore_clips))
            }
            self._cached_predictions[explore_id] = vid_info
        else:
            self._cached_predictions[explore_id] = {}

        if clips.context_clips:
            clip_info = {
                clips.explore_clips[i].vid: {
                    # Assumes that we have one context clip per explore clip,
                    # which is true as long as we don't return context across videos.
                    "vid": clips.context_clips[i][0].vid,
                    "thumbpath": clips.context_clips[i][0].thumbpath,
                    "vpath": clips.context_clips[i][0].vpath,
                }
                for i in range(len(clips.explore_clips))
            }
        else:
            clip_info = {
                clip.vid: {}
                for clip in clips.explore_clips
            }
        self._cached_clip_info[explore_id] = clip_info

        self._cached_prediction_features[explore_id] = clips.prediction_feature_names

        if cache_predictions:
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
        return json.dumps(all_vids[start:end]), len(all_vids)

    @rpyc.exposed
    @core.timing.logtime
    def get_predictions(self, explore_id, vid):
        if vid not in self._cached_predictions[explore_id]:
            logger.info(f'Did not find vid {vid} in cached predictions')
            predictions = self.alm.modelmanager.get_predictions(vids=[vid], feature_names=self._cached_prediction_features[explore_id], allow_stale_predictions=True)
        else:
            predictions = self._cached_predictions[explore_id][vid]['context_predictions']

        return json.dumps([
            ps._asdict() for ps in
            predictions
        ])

    @rpyc.exposed
    def get_clip_info(self, explore_id, vids):
        if not vids:
            return json.dumps([])

        # Check if we have to fetch clip info with path for the vids.
        if self._cached_clip_info[explore_id][vids[0]]:
            clips = list(self._cached_clip_info[explore_id].values())
        else:
            clips = [c._asdict() for c in self.alm.videomanager.get_clipinfo_with_path(vids)]

        return json.dumps([
            {
                'vid': clip['vid'],
                'thumbpath': clip['thumbpath'],
                'vpath': clip['vpath'],
            }
            for clip in clips
        ], cls=CustomEncoder)

    @rpyc.exposed
    def get_labels(self, vid):
        logger.info("Getting labels")
        labels = list(self.alm.get_labels([vid]))

        logger.debug([label._asdict() for label in labels])
        return json.dumps([label._asdict() for label in labels], cls=CustomEncoder)

    @rpyc.exposed
    def get_all_labels(self):
        logger.info("Getting all labels")
        return json.dumps(self.alm.get_unique_labels())

    @rpyc.exposed
    def get_all_prediction_labels(self):
        logger.info("Getting all prediction labels")
        # Just get the labels from the most recent model.
        model_info = self.alm.videomanager.storagemanager.get_model_info(None)
        return json.dumps(model_info.model_labels) if model_info else self.get_all_labels()

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
        return self._postprocess_clips(clips, cache_predictions=False)

    @rpyc.exposed
    def transcode_videos(self, base_dir, output_dir, target_extension):
        transcode_key = str(uuid.uuid4())
        lock = threading.Lock()

        self._cache[transcode_key]["lock"] = lock

        self.alm.videomanager.transcode_videos(
            base_dir,
            output_dir,
            target_extension,
            self.alm.scheduler,
            prepare_fn=partial(self._progress_prepare_fn, lock=lock, key=transcode_key),
            callback_fn=partial(self._progress_callback_fn, lock=lock, key=transcode_key),
        )
        return json.dumps({"key": transcode_key})

    @rpyc.exposed
    def get_progress(self, progress_key):
        if progress_key not in self._cache:
            return json.dumps({})

        with self._cache[progress_key]["lock"]:
            total = self._cache[progress_key].get("total", 0)
            done = self._cache[progress_key].get("done", 0)

        return json.dumps({
            "total": total,
            "done": done,
        })

    @rpyc.exposed
    def debug_reset_annotations(self):
        logger.info("Resetting annotations")
        self.alm.videomanager.reset_annotations()

    @rpyc.exposed
    def debug_add_audio(self):
        logger.info("Adding audio")
        self.alm.videomanager.add_silent_audio()
        return True


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
