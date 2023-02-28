import multiprocessing
import numpy as np
import time
from torch.utils.data import DataLoader
from typing import List

from vfe import datasets
from vfe import core
from vfe.features.abstract import AbstractSegmentProcessor, WrappedIterator


class FeatureExtractorCoordinator:
    def __init__(self,
            extractors: List[AbstractSegmentProcessor],
            done_file_path,
            log_file_path,
            checkpoint_path,
            ignore_done = False):
        self._extractors = extractors
        self._done_videos = set() if ignore_done else self._load_done_videos(done_file_path)
        self._done_video_path = done_file_path
        self._log_file_path = log_file_path
        self._checkpoint_path = checkpoint_path
        self._pool = None
        self._pool_processes = None

    @staticmethod
    def _load_done_videos(done_file_path):
        with open(done_file_path, 'r') as f:
            done_videos = set()
            for video in f:
                done_videos.add(video.strip())
        return done_videos

    @classmethod
    def _is_video_done(cls, video_path):
        return video_path in cls._done_videos

    def _mark_video_done(self, video_paths):
        with open(self._done_video_path, 'a') as f:
            for video_path in video_paths:
                print(video_path, file=f)

    def _log_timing_info(self, video_paths, video_idxs, timing_infos):
        with open(self._log_file_path, 'a') as f:
            for video_path, video_idx, timing_info in zip(video_paths, video_idxs, timing_infos):
                timing_strs = [f'{k},{v}' for k, v in timing_info.items()]
                print(video_path, video_idx, *timing_strs, sep='\t', file=f)

    def _log_checkpoint(self, n, checkpoint_time, checkpoint_time_p):
        with open(self._checkpoint_path, 'a') as f:
            print(n, checkpoint_time, checkpoint_time_p, file=f, sep='\t')

    @staticmethod
    def _load_video(video_path):
        return core.videoframe.extract_raw_frames(video_path)

    @classmethod
    def process_video(cls, batch):
        video_path, _, idx = batch
        assert len(video_path) == 1, f'Got more videos than expected: {len(video_path)}'
        video_path = video_path[0]
        idx = idx.item()
        if cls._is_video_done(video_path):
            # print(f'Skipping already-done video {idx}')
            return None
        start_load = time.perf_counter()
        start_load_p = time.process_time()
        video_info = cls._load_video(video_path)
        timing_info = {
            'load_video': time.perf_counter() - start_load,
            'load_video_p': time.process_time() - start_load_p
        }
        results_info = {}
        if video_info is not None:
            for extractor in cls._processors:
                extractor_id = extractor.filename()
                start = time.perf_counter()
                start_p = time.process_time()
                results_info[extractor_id] = extractor.process_video(video_info, idx)
                timing_info[extractor_id] = time.perf_counter() - start
                timing_info[extractor_id + '_p'] = time.process_time() - start_p
        return (video_path, idx, timing_info, results_info)

    @classmethod
    def _initialize_processors(cls, cls_kwargs, done_videos):
        cls._done_videos = done_videos
        cls._processors = []
        for cls_name, (cls_type, kwargs) in cls_kwargs.items():
            cls._processors.append(cls_type(**kwargs))

    @staticmethod
    def _extractors_to_cls_kwargs(extractors: List[AbstractSegmentProcessor]):
        cls_kwargs = {}
        for extractor in extractors:
            cls_kwargs[extractor.filename()] = (extractor.__class__, {k:v for k, v in extractor.__dict__.items() if not k.startswith('_')})
        return cls_kwargs

    def _get_pool(self, processes):
        if self._pool:
            return self._pool
        self._pool = multiprocessing.Pool(
            processes,
            initializer=self._initialize_processors,
            initargs=(self._extractors_to_cls_kwargs(self._extractors), self._done_videos))
        self._pool_processes = processes
        return self._pool

    def _process_checkpoint(self, video_paths, idxs, timing_infos, results_infos):
            # self._log_timing_info(video_path, idx, timing_info)
            # self._mark_video_done(video_path)
        # For each item in results_info, have the extractor process the results.
        start_checkpoint = time.perf_counter()
        start_checkpoint_p = time.process_time()
        for extractor in self._extractors:
            # Specify dtype=object because some per-frame features may have variable lengths.
            extractor.add_results(np.array(idxs), np.array([results[extractor.filename()] for results in results_infos], dtype=object))
        self._log_timing_info(video_paths, idxs, timing_infos)
        self._mark_video_done(video_paths)
        checkpoint_time = time.perf_counter() - start_checkpoint
        checkpoint_time_p = time.process_time() - start_checkpoint_p
        self._log_checkpoint(len(idxs), checkpoint_time, checkpoint_time_p)

    def _wait_for_imap(self, checkpoint, results):
        video_paths, idxs, timing_infos, results_infos = [], [], [], []
        for video_info in results:
            if video_info is None:
                continue
            (video_path, idx, timing_info, results_info) = video_info
            if not len(results_info):
                # If results_info is empty, we failed to load the video.
                self._mark_video_done([video_path])
                continue
            video_paths.append(video_path)
            idxs.append(idx)
            timing_infos.append(timing_info)
            results_infos.append(results_info)
            if len(video_paths) >= checkpoint:
                self._process_checkpoint(video_paths, idxs, timing_infos, results_infos)
                video_paths, idxs, timing_infos, results_infos = [], [], [], []
        if len(video_paths):
            self._process_checkpoint(video_paths, idxs, timing_infos, results_infos)

    def extract_features(self, *,
            dataset: datasets.VFEDataset = None,
            split: str = None,
            processes = None,
            checkpoint = 500):
        # Open arrays for writing so we can add results.
        for extractor in self._extractors:
            extractor.initialize_array(dataset.name(), split)
        video_iterator = iter(DataLoader(dataset.get_dataset(split), batch_size=1, shuffle=False))
        self._wait_for_imap(checkpoint, self._get_pool(processes).imap_unordered(
            self.process_video,
            video_iterator
        ))
