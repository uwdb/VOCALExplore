import atexit
from collections import defaultdict
import functools
import logging
import numpy as np
import os
import re
import shutil
import time
import threading
import torch
from torch import multiprocessing
from typing import Iterable, Tuple, Union, List

from vfe import core
from vfe.core.logging import configure_logger
from vfe import datasets
from vfe import features
from vfe.features.modelcoordinator import ModelFeatureExtractorCoordinator

from vfe.api.storagemanager import AbstractStorageManager, ClipSet, VidType
from .abstract import AbstractAsyncFeatureManager, FeatureSet

class BackgroundFeatureManager(AbstractAsyncFeatureManager):
    _s_features_queue = None
    _s_pause_event = None

    @staticmethod
    def _initialize_pool(features_queue, pause_event):
        configure_logger()
        BackgroundFeatureManager._s_features_queue = features_queue
        BackgroundFeatureManager._s_pause_event = pause_event

    @staticmethod
    def _wait_for_extract(features_queue, pause_event, args_queue):
        BackgroundFeatureManager._initialize_pool(features_queue, pause_event)
        while True:
            next_args = args_queue.get()
            BackgroundFeatureManager._extract(*next_args)

    @staticmethod
    def _copy_to_ssd(vids_and_vpaths):
        updated = []
        prefixes = ['/gscratch/balazinska/', '/data/']
        for vpath, info_dict in vids_and_vpaths:
            for prefix in prefixes:
                if vpath.startswith(prefix):
                    updated_vpath = vpath.replace(prefix, '/scr/')
                    break
            core.filesystem.create_dir(os.path.dirname(updated_vpath))
            if not os.path.exists(updated_vpath):
                shutil.copy(vpath, updated_vpath)
            updated.append((updated_vpath, info_dict))
        return updated

    @staticmethod
    def _extract(feature_name, vids_and_vpaths, callid, batch_size, num_workers, checkpoint, device, use_dali, vid_ssd):
        BackgroundFeatureManager._s_pause_event.wait()

        features_queue = BackgroundFeatureManager._s_features_queue
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
            ignore_done=True
        )
        if vid_ssd:
            vids_and_vpaths = BackgroundFeatureManager._copy_to_ssd(vids_and_vpaths)

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
                            self.logger.debug(f'Processing additional callback for callid {args[1]}')
                            callback()
                elif not self._no_store:
                    self.storagemanager.add_feature_batch(*args)
            except: # queue.Empty, or queue handle is closed.
                continue

    def _sample_vids_to_extract(self):
        # Currently expects one feature_name.
        # Wait a second after startup to let more important tasks get scheduled.
        time.sleep(1)
        consecutive_free_checks = 0
        target_consecutive_free_checks = 2
        while True:
            time.sleep(1)

            # This check can be imprecise, so don't worry about locking.
            if not self._callbacks:
                consecutive_free_checks += 1
            else:
                consecutive_free_checks = 0

            if consecutive_free_checks >= target_consecutive_free_checks:
                # Sample 10 vids to extract features from.
                all_vids = set(self.storagemanager.get_all_vids())
                # This code block may read from self._done_or_inprogress_vids while another thread is modifying it.
                # As long as it doesn't cause a crash, it's not an issue. _extract_features_async will remove any
                # vids that were just added to _done_or_inprogress_vids[feature_name].
                feature_names = list(self._done_or_inprogress_vids.keys())
                if not feature_names:
                    continue
                feature_name = feature_names[0]
                vids_without_features = all_vids - self._done_or_inprogress_vids[feature_name]
                if not vids_without_features:
                    continue
                size = min(10, len(vids_without_features))
                vids_to_extract = self.rng.choice(np.array([*vids_without_features]), size=size, replace=False)
                self.logger.debug(f'Extracting features from vids {vids_to_extract}')
                # _extract_features_async is threadsafe, so we can call it from this background thread
                # and from the main thread.
                self._extract_features_async(feature_name, vids_to_extract)

    def __init__(self, storagemanager: AbstractStorageManager, num_workers=1, batch_size=1, device=None, checkpoint=500, rng=None, num_processes=1, dali_preprocess=True, async_batch_size=-1, quiettime_async=True, vid_ssd=False, no_store=False):
        self.dali_preprocess = dali_preprocess
        self.storagemanager = storagemanager
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.rng = np.random.default_rng(rng)
        self.device = device if device is not None else \
                'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_processes = num_processes
        self.async_batch_size = async_batch_size
        self.vid_ssd = vid_ssd
        self.no_store = no_store
        self.logger = logging.getLogger(__name__)

        self._callid = 0
        self._callbacks = {}
        self._callbacks_lock = threading.Lock()

        self._done_or_inprogress_vids = defaultdict(set)
        for feature_name in self.storagemanager.get_feature_names():
            self._done_or_inprogress_vids[feature_name] = set(self.storagemanager.get_stored_feature_vids(feature_name))

        self._ctx = torch.multiprocessing.get_context('spawn')
        # We start out unpaused.
        self._pause_event = self._ctx.Event()
        self._pause_event.set()
        self._features_queue = self._ctx.Queue()
        if self.dali_preprocess:
            self._extraction_pool = self._ctx.Pool(self.num_processes, initializer=self._initialize_pool, initargs=(self._features_queue, self._pause_event))
            self._extract_queue = None
        else:
            self._extraction_pool = None
            self._extract_queue = self._ctx.Queue()
            self._extraction_process = self._ctx.Process(target=self._wait_for_extract, args=(self._features_queue, self._pause_event, self._extract_queue))
            self._extraction_process.start()

        self._processing_event = threading.Event()
        self._feature_processing_thread = threading.Thread(group=None, target=self._handle_feature_batch, args=(self._features_queue,), name='process-features')
        self._feature_processing_thread.daemon = True # True because infinite loop never joins.
        self._feature_processing_thread.start()

        if quiettime_async:
            self._quiettime_extraction_thread = threading.Thread(group=None, target=self._sample_vids_to_extract, name='quiettime-features')
            self._quiettime_extraction_thread.daemon = True # True because infinite loop never joins.
            self._quiettime_extraction_thread.start()

        atexit.register(self.shutdown)

    def shutdown(self):
        self.logger.info('Terminating extraction pool')
        self._processing_event.set()
        self._features_queue.close()
        if self._extraction_pool:
            self._extraction_pool.terminate()
        else:
            self._extraction_process.terminate()
            self._extract_queue.close()

    def pause(self):
        self.logger.info('Paused')
        self._pause_event.clear()

    def resume(self):
        self.logger.info('Resumed')
        self._pause_event.set()

    def add_video(self, path, start_time=None, duration=None) -> VidType:
        # Don't proactively extract features.
        if duration is None:
            duration = core.video.get_video_duration(path)
        try:
            return self.storagemanager.add_video(path, start_time, duration)
        except Exception as e:
            self.logger.warn(f'Failed to add video at path {path} with exception {e}')
            return None

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        return self.storagemanager.add_videos(video_csv_path)

    def get_features(self, feature_names: Union[str, List[str]], vids) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)

        if vids is None:
            self.logger.debug(f'vids is None; returning all stored features')
            return self.storagemanager.get_features(feature_names=feature_names, vids=None)

        if not isinstance(vids, np.ndarray):
            vids = np.array(vids)

        self.logger.debug(f'Requested features for {len(vids)} vids: {vids if len(vids) < 500 else "(too many to print)"}')
        # _extract_features_async automatically filters out already-done vids.
        for feature_name in feature_names:
            self._extract_features_async(feature_name, vids)

        # Queue all extraction tasks before waiting.
        for feature_name in feature_names:
            self._wait_for_vids(feature_name, vids)

        return self.storagemanager.get_features(feature_names=feature_names, vids=vids)

    def extract_features_async(self, feature_names: Union[str, List[str]], vids, callback=None) -> None:
        feature_names = core.typecheck.ensure_list(feature_names)
        if len(feature_names) == 1:
            self._extract_features_async(feature_names[0], vids, callback)
        else:
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
                self._extract_features_async(feature_name, vids, wrapped_callback)

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet) -> FeatureSet:
        feature_names = core.typecheck.ensure_list(feature_names)

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
        # Only wait after all extraction tasks have been queued.
        # The order we wait doesn't matter because we'll always have to wait for the slowest task.
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

    def _extract_features_async(self, feature_name, vids, callback=None):
        with self._callbacks_lock:
            missing_vids = set(vids) - self._done_or_inprogress_vids[feature_name]
            if len(missing_vids) == 0:
                if callback:
                    callback()
                return

            vids_and_vpaths = datasets.utils.unlabeled_video_paths_for_vids_from_storagemanager(self.storagemanager, missing_vids, mp4=True)
            self._done_or_inprogress_vids[feature_name] |= missing_vids

            args = []
            nvids = len(vids_and_vpaths)
            step = self.async_batch_size if self.async_batch_size > 0 \
                else nvids
            for start in range(0, len(vids_and_vpaths), step):
                stop = min(start+step, nvids)
                vids = set([v[1]['vid'] for v in vids_and_vpaths[start:stop]])
                callid = self._callid
                self._callid += 1
                self._callbacks[callid] = {'feature_name': feature_name, 'vids': vids, 'callbacks': [callback]}
                self.logger.debug(f'Queueing feature extraction for callid {callid}: feature {feature_name}, vids {vids}')
                args.append((feature_name, vids_and_vpaths[start:stop], callid, self.batch_size, self.num_workers, self.checkpoint, self.device, self.dali_preprocess, self.vid_ssd))

        for argset in args:
            if self._extraction_pool is not None:
                self._extraction_pool.apply_async(self._extract, args=argset)
            else:
                self._extract_queue.put(argset)

    def _wait_for_vids(self, feature_name, vids):
        # This doesn't have to wait on entire extraction tasks.
        # There can be more fine-grained callbacks per-vid handled in the feature thread.
        # Because we take the lock before adding all of the callbacks, there shouldn't be a
        # race condition where we add a callback for something that finishes in the background,
        # so the callback is never called.
        vids = set(vids)
        events = []
        with self._callbacks_lock:
            for callid, fname_and_vids_and_callbacks in self._callbacks.items():
                if fname_and_vids_and_callbacks['feature_name'] != feature_name:
                    continue
                if len(vids & fname_and_vids_and_callbacks['vids']):
                    event = threading.Event()
                    def handle_event(callid=None, event=None):
                        logging.debug(f'Handling event for callid {callid}')
                        event.set()
                    fname_and_vids_and_callbacks['callbacks'].append(functools.partial(handle_event, callid=callid, event=event))
                    events.append(event)
        self.logger.debug(f'Waiting for vids {vids if len(vids) < 500 else "#" + str(len(vids))}; {len(events)} remaining feature extraction tasks')
        for i, event in enumerate(events):
            self.logger.debug(f'Waiting on event {i} of {len(events)}')
            event.wait()
