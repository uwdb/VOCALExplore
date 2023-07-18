import atexit
from collections import defaultdict
import functools
import logging
import numpy as np
import os
import pandas as pd
import queue
import re
import shutil
from tempfile import NamedTemporaryFile
import time
import threading
import torch
from torch import multiprocessing
from typing import Iterable, Tuple, Union, List

from vfe import core
from vfe.core.logging import configure_logger
from vfe.core.timing import logtime
from vfe import datasets
from vfe import features
from vfe.features.modelcoordinator import ModelFeatureExtractorCoordinator

from vfe.api.storagemanager import AbstractStorageManager, ClipSet, VidType
from vfe.api.scheduler import AbstractScheduler, Priority
from .abstract import AbstractAsyncFeatureManager, FeatureSet

def copy_to_ssd(vids_and_vpaths):
    updated = []
    prefixes = ['/gscratch/balazinska/', '/data/']
    for vpath, info_dict in vids_and_vpaths:
        for prefix in prefixes:
            if vpath.startswith(prefix):
                updated_vpath = vpath.replace(prefix, '/scr/')
                break
        core.filesystem.create_dir(os.path.dirname(updated_vpath))
        shutil.copy(vpath, updated_vpath)
        updated.append((updated_vpath, info_dict))
    return updated

class BackgroundAsyncFeatureManager(AbstractAsyncFeatureManager):
    @staticmethod
    def _extract(pause_event, features_queue, feature_name, vids_and_vpaths, callid, batch_size, num_workers, checkpoint, device, use_dali, vid_ssd):
        pause_event.wait()

        logger = logging.getLogger(__name__)
        logger.debug(f'Extracting features for callid {callid}: feature {feature_name} using {num_workers} threads on device {device}/gpu')
        extractor = features.utils.get_extractor(feature_name, features_dir=None, device=device)

        if use_dali:
            dl_kwargs = {
                'type': datasets.DatasetType.DALI,
                'device': 'gpu',
                **extractor.dali_kwargs(feature_name),
            }
        else:
            fps = re.findall(r'(\d+)fps', feature_name)
            fps = int(fps[0]) if fps else None
            fstride = re.findall(r'(\d+)fstride', feature_name)
            fstride = int(fstride[0]) if fstride else None
            dataset_type = features.utils.get_extractor_type(extractor)
            dl_kwargs = {
                'type': dataset_type,
                'transform': extractor.transform(),
                'done_idxs': set(),
            }
            if dataset_type == datasets.DatasetType.FRAME:
                dl_kwargs['stride'] = datasets.frame.CreateStride(fps=fps, fstride=fstride)
            elif dataset_type == datasets.DatasetType.CLIP:
                dl_kwargs['clip_sampler_fn'] = extractor.clip_sampler_fn

        coordinator = ModelFeatureExtractorCoordinator(
            models=[extractor],
            ignore_done=True,
            device=device,
        )
        if vid_ssd:
            vids_and_vpaths = copy_to_ssd(vids_and_vpaths)

        coordinator.extract_features_from_vids_and_vpaths(
            vids_and_vpaths=vids_and_vpaths,
            featurequeue=features_queue,
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint=checkpoint
        )

        if vid_ssd:
            for vpath, info_dict in vids_and_vpaths:
                if vpath.startswith('/scr'):
                    os.remove(vpath)

        logger.debug(f'Finished extracting features for callid {callid}')
        features_queue.put(('Done', callid))

    def _handle_feature_batch(self, features_queue):
        while not self._processing_event.is_set():
            try:
                args = features_queue.get(True, timeout=1)
                if args[0] == 'Done':
                    with self._callbacks_lock:
                        vids_and_callbacks = self._callbacks.pop(args[1])
                    self.logger.debug(f'Processing feature extraction done callback for callid {args[1]}')
                    for callback in vids_and_callbacks['callbacks']:
                        if callback:
                            callback()
                else:
                    self.logger.debug('Adding feature batch')
                    self.storagemanager.add_feature_batch(*args)
            except Exception as e: # queue.Empty, or queue handle is closed.
                if not isinstance(e, queue.Empty):
                    self.logger.warn(f'Exception while handling feature batch: {e}')
                continue

    # def _sample_vids_to_extract(self):
    #     # Currently expects one feature_name.
    #     # Wait a second after startup to let more important tasks get scheduled.
    #     time.sleep(1)
    #     consecutive_free_checks = 0
    #     target_consecutive_free_checks = 2
    #     while True:
    #         time.sleep(1)

    #         # This check can be imprecise, so don't worry about locking.
    #         if not self._callbacks:
    #             consecutive_free_checks += 1
    #         else:
    #             consecutive_free_checks = 0

    #         if consecutive_free_checks >= target_consecutive_free_checks:
    #             # Sample 10 vids to extract features from.
    #             all_vids = set(self.storagemanager.get_all_vids())
    #             # This code block may read from self._done_or_inprogress_vids while another thread is modifying it.
    #             # As long as it doesn't cause a crash, it's not an issue. _extract_features_async will remove any
    #             # vids that were just added to _done_or_inprogress_vids[feature_name].
    #             feature_names = list(self._done_or_inprogress_vids.keys())
    #             if not feature_names:
    #                 continue
    #             feature_name = feature_names[0]
    #             vids_without_features = all_vids - self._done_or_inprogress_vids[feature_name]
    #             if not vids_without_features:
    #                 continue
    #             size = min(10, len(vids_without_features))
    #             vids_to_extract = self.rng.choice(np.array([*vids_without_features]), size=size, replace=False)
    #             self.logger.debug(f'Extracting features from vids {vids_to_extract}')
    #             # _extract_features_async is threadsafe, so we can call it from this background thread
    #             # and from the main thread.
    #             self._extract_features_async(feature_name, vids_to_extract)

    def __init__(self, storagemanager: AbstractStorageManager, scheduler: AbstractScheduler, num_workers=0, batch_size=1, device=None, checkpoint=500, rng=None, dali_preprocess=True, async_batch_size=-1, vid_ssd=False):
        self.logger = logging.getLogger(__name__)
        self.dali_preprocess = dali_preprocess
        self.storagemanager = storagemanager
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.rng = np.random.default_rng(rng)
        self.device = device if device is not None else \
                'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu' and self.dali_preprocess:
            self.logger.warn('Overriding dali_preprocess to False because no GPU is available')
            self.dali_preprocess = False

        if self.dali_preprocess and self.num_workers == 0:
            self.logger.warn('Overriding num_workers from 0 to 1 because Dali requires non-zero threads')
            self.num_workers = 1

        self.async_batch_size = async_batch_size
        self.vid_ssd = vid_ssd

        self._callid = 0
        # callid -> {feature_name:, vids:, callbacks:}
        self._callbacks = {}
        # base_callid -> [count, base_callback]
        self._base_callbacks = {}
        self._callbacks_lock = threading.Lock()

        self._done_or_inprogress_vids = defaultdict(set)
        for feature_name in self.storagemanager.get_feature_names():
            self._done_or_inprogress_vids[feature_name] = set(self.storagemanager.get_stored_feature_vids(feature_name))

        self._mp_manager = self.scheduler.context().Manager() # Slow; starts a new process.
        self._pause_event = self._mp_manager.Event()
        self._pause_event.set()
        self._features_queue = self._mp_manager.Queue()

        self._processing_event = threading.Event()
        self._feature_processing_thread = threading.Thread(group=None, target=self._handle_feature_batch, args=(self._features_queue,), name='process-features')
        self._feature_processing_thread.daemon = True # True because infinite loop never joins.
        self._feature_processing_thread.start()

        atexit.register(self.shutdown)

    def shutdown(self):
        self.logger.info('Shutting down')
        self._processing_event.set()
        self._mp_manager.shutdown()

    def pause(self):
        self.logger.info('Paused')
        self._pause_event.clear()

    def resume(self):
        self.logger.info('Resumed')
        self._pause_event.set()

    def add_video(self, path, start_time=None, duration=None, thumbnail_dir=None) -> VidType:
        # Don't proactively extract features.
        if duration is None:
            duration = core.video.get_video_duration(path)
        thumbnail_path = None if not thumbnail_dir else core.video.save_thumbnail(path, thumbnail_dir)
        try:
            return self.storagemanager.add_video(path, start_time, duration, thumbnail_path=thumbnail_path)
        except Exception as e:
            self.logger.warn(f'Failed to add video at path {path} with exception {e}')
            return None

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        return self.storagemanager.add_videos(video_csv_path)

    @staticmethod
    def _add_videos(vpaths, thumbnail_dir):
        for vpath in vpaths:
            core.video.save_thumbnail(vpath, thumbnail_dir)

    def add_videos_from_dir(self, base_dir, thumbnail_dir, prepare_fn, callback_fn) -> None:
        vpaths = []
        vstart = None

        for dirpath, dirnames, filenames in os.walk(base_dir):
            for file in filenames:
                if not file.endswith(".mp4"):
                    continue

                vpath = os.path.join(dirpath, file)
                vpaths.append(vpath)
                # Prepare twice: once for the thumbnail, and once for the video metadata.
                prepare_fn()
                prepare_fn()

        # Schedule all of the feature extraction tasks.
        for i in range(0, len(vpaths), 50):
            batch = vpaths[i : i+50]
            def wrapped_callback(*args, callback_fn=None, n=None):
                for _ in range(n):
                    callback_fn(*args)

            self.scheduler.schedule(
                'extract-thumbnail',
                functools.partial(
                    self._add_videos,
                    vpaths=batch,
                    thumbnail_dir=thumbnail_dir,
                ),
                callback=functools.partial(wrapped_callback, callback_fn=callback_fn, n=len(batch))
            )

        # Add all of the metadata to the database.
        # Add a prepare/callback for each batch so we are done once everything
        # is in the database.
        with NamedTemporaryFile(mode="w+", suffix=".csv") as video_csv:
            rows = []
            for vpath in vpaths:
                duration = core.video.get_video_duration(vpath)
                thumbnail_path = core.video.get_thumbnail_path(vpath, thumbnail_dir)
                # Callback now that we have the duration. It's not actually done since
                # we still have to add it to the database, but it's close.
                callback_fn()

                # Keep track of data for bulk insert.
                rows.append((vpath, vstart, duration, thumbnail_path))
                if len(rows) > 2000:
                    df = pd.DataFrame.from_records(rows, columns=["path", "start", "duration", "thumbpath"])
                    df.to_csv(video_csv.name)
                    self.storagemanager.add_videos(video_csv.name, include_thumbpath=True)
                    rows = []

            if rows:
                df = pd.DataFrame.from_records(rows, columns=["path", "start", "duration", "thumbpath"])
                df.to_csv(video_csv.name)
                self.storagemanager.add_videos(video_csv.name, include_thumbpath=True)


    @logtime
    def get_features(self, feature_names: Union[str, List[str]], vids, priority: Priority=Priority.DEFAULT) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)

        if vids is None:
            self.logger.debug(f'vids is None; returning all stored features')
            return self.storagemanager.get_features(feature_names=feature_names, vids=None)

        if not isinstance(vids, np.ndarray):
            vids = np.array(vids)

        self.logger.debug(f'Requested features for {len(vids)} vids: {vids if len(vids) < 100 else "(too many to print)"}')
        # _extract_features_async automatically filters out already-done vids.
        for feature_name in feature_names:
            self._extract_features_async(feature_name, vids, priority=priority)

        # Only wait once we've queued all extraction tasks.
        for feature_name in feature_names:
            self._wait_for_vids(feature_name, vids)

        return self.storagemanager.get_features(feature_names=feature_names, vids=vids)

    def extract_features_async(self, feature_names: Union[str, List[str]], vids, callback=None, priority: Priority=Priority.DEFAULT, prepare=None) -> None:
        feature_names = core.typecheck.ensure_list(feature_names)
        if len(feature_names) == 1:
            self._extract_features_async(feature_names[0], vids, callback, priority=priority, prepare=prepare)
        else:
            # TODO: create an abstraction to wait on multiple sub-tasks and call the callback once all are done.
            # This logic duplicates the logic in background.py, and a few other places do similar things.
            n_remaining = len(vids)
            decrement_lock = threading.Lock()
            def wrapped_callback():
                nonlocal n_remaining
                with decrement_lock:
                    n_remaining -= 1
                    do_callback = n_remaining == 0
                if do_callback and callback is not None:
                    callback()
            for feature_name in feature_names:
                self._extract_features_async(feature_name, vids, wrapped_callback, priority=priority, prepare=prepare)

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet, only_already_extracted=False) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)

        if not only_already_extracted:
            # Assumes that all of the features for a given vid have been materialized, or none of them have.
            # Also assumes that for a given model, either all layers have been materialized or none of them have.
            # Figure out what vids have already been materialized.
            cliptable = clipset.to_table()

            vids = cliptable['vid'].to_pylist()
            # _extract_features_async automatically filters out already-done vids.
            for feature_name in feature_names:
                self._extract_features_async(feature_name, vids)

            # If any vids in the clipset are in the feature extraction pipeline in the background,
            # wait for them to finish.
            # Only wait once we've queued all extraction tasks.
            for feature_name in feature_names:
                self._wait_for_vids(feature_name, vids)

        # Return the complete feature set.
        # At this point, we know that the features for all vids have been extracted and therefore will be in the feature store.
        return self.storagemanager.get_features_for_clips(feature_names=feature_names, clipset=clipset)

    def get_extracted_features_info(self, feature_name) -> Tuple[Iterable[VidType], Iterable[VidType]]:
        # Return (vids _with_ features extracted, vids _without_ features extracted)
        vids_with_features = self.storagemanager.get_stored_feature_vids(feature_name)
        vids_without_features = set(self.storagemanager.get_all_vids()) - set(vids_with_features)
        return (vids_with_features, vids_without_features)

    def _extract_features_async(self, feature_name, vids, callback=None, priority: Priority=Priority.DEFAULT, prepare=None):
        args = []
        vids = set(vids)
        ntasks_remaining = 0
        with self._callbacks_lock:
            missing_vids = vids - self._done_or_inprogress_vids[feature_name]

            def wrapped_callback_fn(call_id):
                callback = None
                with self._callbacks_lock:
                    self._base_callbacks[call_id][0] -= 1
                    # self.logger.debug(f'Remaining tasks for callback {call_id}: {self._base_callbacks[call_id][0]}')
                    if self._base_callbacks[call_id][0] == 0:
                        callback = self._base_callbacks[call_id][1]
                        self._base_callbacks.pop(call_id)
                if callback:
                    callback()

            base_callid = self._callid
            # self.logger.debug(f'Counting remaining tasks for callback {base_callid}')
            self._callid += 1
            wrapped_callback = functools.partial(wrapped_callback_fn, call_id=base_callid)

            # Check for any in progress vids. Do this before missing vids so that we don't double
            # wait on new tasks.
            for callid, fname_and_vids_and_callbacks in self._callbacks.items():
                if fname_and_vids_and_callbacks['feature_name'] != feature_name:
                    continue
                if len(vids & fname_and_vids_and_callbacks['vids']):
                    # self.logger.debug(f'Adding a callback event for callid {callid}')
                    fname_and_vids_and_callbacks['callbacks'].append(wrapped_callback)
                    # Increment the number of tasks we're waiting for before calling the specified callback.
                    ntasks_remaining += 1

            if len(missing_vids) > 0:
                vids_and_vpaths = datasets.utils.unlabeled_video_paths_for_vids_from_storagemanager(self.storagemanager, missing_vids, mp4=True)
                self._done_or_inprogress_vids[feature_name] |= missing_vids

                nvids = len(vids_and_vpaths)
                step = self.async_batch_size if self.async_batch_size > 0 \
                    else nvids
                for start in range(0, len(vids_and_vpaths), step):
                    stop = min(start+step, nvids)
                    vids = set([v[1]['vid'] for v in vids_and_vpaths[start:stop]])
                    callid = self._callid
                    self._callid += 1
                    self._callbacks[callid] = {'feature_name': feature_name, 'vids': vids, 'callbacks': [wrapped_callback]}
                    ntasks_remaining += 1
                    self.logger.debug(f'Queueing feature extraction for callid {callid}: feature {feature_name}, vids {vids}, callback {base_callid}')
                    args.append((self._pause_event, self._features_queue, feature_name, vids_and_vpaths[start:stop], callid, self.batch_size, self.num_workers, self.checkpoint, self.device, self.dali_preprocess, self.vid_ssd))

            # Use a list so we can decrement ntasks_remaining directly.
            # self.logger.debug(f'Remaining tasks for callback {base_callid}: {ntasks_remaining}')
            self._base_callbacks[base_callid] = [ntasks_remaining, callback]

        if ntasks_remaining:
            for argset in args:
                self.scheduler.schedule('extract_features', functools.partial(self._extract, *argset), priority=priority, prepare=prepare)
        else:
            if prepare:
                prepare()
            if callback:
                callback()

    def _wait_for_vids(self, feature_name, vids):
        # This doesn't have to wait on entire extraction tasks.
        # There can be more fine-grained callbacks per-vid handled in the feature thread.
        vids = set(vids)
        events = []
        with self._callbacks_lock:
            for callid, fname_and_vids_and_callbacks in self._callbacks.items():
                if fname_and_vids_and_callbacks['feature_name'] != feature_name:
                    continue
                if len(vids & fname_and_vids_and_callbacks['vids']):
                    # self.logger.debug(f'Adding a callback event for callid {callid}')
                    event = threading.Event()
                    def handle_event(event=None):
                        event.set()
                    fname_and_vids_and_callbacks['callbacks'].append(functools.partial(handle_event, event=event))
                    events.append(event)
        self.logger.debug(f'Waiting for vids {vids if len(vids) < 50 else "#" + str(len(vids))}; {len(events)} remaining feature extraction tasks')
        for event in events:
            event.wait()
