import cv2
from enum import Enum
from functools import partial
import glob
import joblib
import logging
import math
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from torch.utils.data import DataLoader

from vfe.core import consts, videoframe
from vfe.core import dataframe
from vfe import datasets
from vfe.features.pooledts.pot import PoT
from vfe.features.abstract import *

# Python OpenCV samples: https://github.com/opencv/opencv/blob/4.x/samples/python

class IntermediateFileWriter:
    def __init__(self, processor, feature_dim, lock, chunk_size=4096, dtype=np.float32):
        self.features_dir = processor.features_dir
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.dtype = dtype
        self._processor_name = processor.filename()
        self._intermediate_filename = processor.filename() + '_intermediate'
        self._intermediate_idx_filename = processor.filename() + '_indices'
        self._opened_intermediate_array = None
        self._opened_intermediate_idx_array = None

    def add_results(self, idxs, results):
        order = np.argsort(idxs)
        idxs = idxs[order]
        results = results[order]

        h, _ = self._opened_intermediate_array.shape
        heights = [len(result) for result in results]
        cumulative_heights = np.cumsum(heights)
        ends = h + cumulative_heights
        starts = np.append(np.array(h), ends[:-1])
        self._opened_intermediate_idx_array[idxs, 0] = starts
        self._opened_intermediate_idx_array[idxs, 1] = ends
        self._opened_intermediate_idx_array.flush()
        self._opened_intermediate_array.append(np.vstack(results))

    def initialize_array(self, dataset_name, split):
        # Open one array to hold intermediate results.
        self._opened_intermediate_array = get_zarr_array(self.features_dir, self._intermediate_filename, dataset_name, split, sync=False)

        # Open another array that holds (start, end) values.
        self._opened_intermediate_idx_array = get_npy_array(self.features_dir, self._intermediate_idx_filename, dataset_name, split)

    def ensure_array_exists(self, dataset: datasets.VFEDataset, split):
        for arr_path in [
            zarr_arr_path(self.features_dir, self._intermediate_filename, dataset.name(), split)[0],
            npy_arr_path(self.features_dir, self._intermediate_idx_filename, dataset.name(), split)
        ]:
            if not os.path.exists(arr_path):
                self.reset_array(dataset, split)

    def reset_array(self, dataset: datasets.VFEDataset, split):
        # Create an empty array of undetermined size to hold intermediate results.
        create_zarr_array(self.features_dir, self._intermediate_filename, dataset.name(), split, shape=(0, self.feature_dim), chunk_size=self.chunk_size, dtype=self.dtype, sync=False)

        # Create a 2d array to hold start/end indices into the intermediate array.
        n_records = len(dataset.get_labels(split))
        create_npy_array(self.features_dir, self._intermediate_idx_filename, dataset.name(), split, shape=(n_records, 2), dtype=np.int64)

    def verify_array(self, dataset: datasets.VFEDataset, split):
        for arr_name in [self._intermediate_filename, self._intermediate_idx_filename]:
            arr = get_npy_array(self.features_dir, arr_name, dataset.name(), split)
            print('verifying', arr_name)
            for idx, row in enumerate(arr):
                if np.all(row == 0):
                    print(f'({arr_name}) Row {idx} not set')

    def _file_prefix(self, dataset, split):
        return f'{self._processor_name}_{dataset.name()}_{split}'

    def get_intermediate_results(self, dataset: datasets.VFEDataset, split):
        return [
            zarr_arr_path(self.features_dir, self._intermediate_filename, dataset.name(), split)[0],
            npy_arr_path(self.features_dir, self._intermediate_idx_filename, dataset.name(), split),
        ]


class PerFrameAbstractSegmentProcessor(AbstractSegmentProcessor):
    def __init__(self, levels=4, lock=None, chunk_size=4096, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        self.lock = lock
        self._pot = PoT(levels=self.levels)
        self._intermediate_file_writer = IntermediateFileWriter(self, self.base_feature_dim(), self.lock, chunk_size=chunk_size)
        self._opened_arrays = {}

    def initialize_array(self, dataset_name, split):
        # self._intermediate_file_writer.initialize_array(dataset_name, split)
        for agg in self.AggregationType:
            self._opened_arrays[agg] = get_npy_array(self.features_dir, self.filename(agg), dataset_name, split)

    def ensure_array_exists(self, dataset: datasets.VFEDataset, split):
        # self._intermediate_file_writer.ensure_array_exists(dataset, split)
        for agg in self.AggregationType:
            arr_path = npy_arr_path(self.features_dir, self.filename(agg), dataset.name(), split)
            if not os.path.exists(arr_path):
                self._reset_array(dataset, split, agg)

    def reset_array(self, dataset: datasets.VFEDataset, split):
        # self._intermediate_file_writer.reset_array(dataset, split)
        for agg in self.AggregationType:
            self._reset_array(dataset, split, agg)

    def _reset_array(self, dataset: datasets.VFEDataset, split, agg):
        n_records = len(dataset.get_labels(split))
        create_npy_array(self.features_dir, self.filename(agg), dataset.name(), split, shape=(n_records, self.featuredim(agg)))

    def verify_array(self, dataset: datasets.VFEDataset, split):
        self._intermediate_file_writer.verify_array(dataset, split)

    def add_results(self, idxs, results):
        # self._intermediate_file_writer.add_results(idxs, results)
        order = np.argsort(idxs)
        idxs = idxs[order]
        results = results[order]
        for agg in self.AggregationType:
            arr = self._opened_arrays[agg]
            arr[idxs] = [result[agg.value] for result in results]
            arr.flush()

    def write_intermediate_results(self, idx, np_result):
        self._intermediate_file_writer.write_intermediate_results(idx, np_result)

    def base_feature_dim(self):
        raise NotImplementedError

    def _pot_dim(self):
        n_tws = len(self._pot.get_temporal_windows(self.levels))
        n_features = self.base_feature_dim()
        # 4 values: sum, max, positive gradients, negative gradients.
        return n_tws * n_features * 4

    class AggregationType(Enum):
        POT = 'pot'
        MEAN = 'mean'
        MAX = 'max'

    def filename(self, agg: str = None):
        if agg is None:
            return self._filename(None)
        return self._filename(self.AggregationType(agg))

    def _filename(self, agg: AggregationType):
        raise NotImplementedError

    def featuredim(self, agg: str):
        agg = self.AggregationType(agg)
        if agg in (self.AggregationType.MEAN, self.AggregationType.MAX):
            return self.base_feature_dim()
        elif agg == self.AggregationType.POT:
            return self._pot_dim()
        else:
            assert False, 'Unrecognized aggregation type: %r' % agg

    @classmethod
    def _extract_mean(cls, np_array):
        return np.mean(np_array, axis=0)

    @classmethod
    def _extract_max(cls, np_array):
        return np.max(np_array, axis=0)

    @classmethod
    def _extract_pot(cls, np_array):
        return cls._processor._pot.extract_features(preprocessing.normalize(np_array, norm='l1'))

    @classmethod
    def _extract_aggregated_features(cls, wrapped_it, feature_fn):
        start = wrapped_it['value']
        end = start + wrapped_it['step']
        index_array = np.load(wrapped_it['index_path'], mmap_mode='r')
        min_intermediate = np.min(index_array[start:end, 0])
        max_intermediate = np.max(index_array[start:end, 1])
        intermediates_array = zarr.open(wrapped_it['intermediates_path'], mode='r')
        # Read the range we need into memory once.
        needed_intermediates = intermediates_array[min_intermediate:max_intermediate]
        max_idx = len(index_array)
        idxs = []
        update_vals = []
        for idx in range(start, end):
            if idx >= max_idx:
                # For the last range, we may go past the end.
                break
            idxs.append(idx)
            int_start, int_end = index_array[idx]
            # start/end are offset by min_intermediate.
            int_start = int(int_start - min_intermediate)
            int_end = int(int_end - min_intermediate)
            update_vals.append(feature_fn(needed_intermediates[int_start:int_end]))
        arr = open_npy_array(wrapped_it['arr_path'])
        arr[idxs] = update_vals
        arr.flush()

    def _extract_combined(self, dataset, split, processes, fn, agg: AggregationType):
        intermediates_path, index_path = self._intermediate_file_writer.get_intermediate_results(dataset, split)
        n_records = len(dataset.get_labels(split))
        shape = (n_records, self.featuredim(agg))
        arr_path = create_npy_array(self.features_dir, self._filename(agg), dataset.name(), split, shape)
        step = math.ceil(n_records / processes)
        self._wait_for_imap(self._get_pool(processes).imap_unordered(
            partial(self._extract_aggregated_features, feature_fn=fn),
            WrappedIterator(iter(range(0, n_records, step)), step=step, arr_path=arr_path, intermediates_path=intermediates_path, index_path=index_path)
        ))

    def extract_pot(self, dataset, split, processes=multiprocessing.cpu_count() - 1):
        self._extract_combined(dataset, split, processes, self._extract_pot, self.AggregationType.POT)

    def extract_mean(self, dataset, split, processes=multiprocessing.cpu_count() - 1):
        self._extract_combined(dataset, split, processes, self._extract_mean, self.AggregationType.MEAN)

    def extract_max(self, dataset, split, processes=multiprocessing.cpu_count() - 1):
        self._extract_combined(dataset, split, processes, self._extract_max, self.AggregationType.MAX)


class BagOfWordsAbstractSegmentProcessor(AbstractSegmentProcessor):
    def __init__(self, feature_dim=-1, n_clusters=100, lock=None, chunk_size=4096, dtype=np.float32, **kwargs):
        super().__init__(**kwargs)
        # self.batch_size = 2048
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.lock = lock
        self._intermediate_file_writer = IntermediateFileWriter(self, self.feature_dim, self.lock, chunk_size, dtype=dtype)

    def initialize_array(self, dataset_name, split):
        self._intermediate_file_writer.initialize_array(dataset_name, split)

    def ensure_array_exists(self, dataset: datasets.VFEDataset, split):
        self._intermediate_file_writer.ensure_array_exists(dataset, split)

    def reset_array(self, dataset: datasets.VFEDataset, split):
        self._intermediate_file_writer.reset_array(dataset, split)

    def verify_array(self, dataset: datasets.VFEDataset, split):
        self._intermediate_file_writer.verify_array(dataset, split)

    def add_results(self, idxs, results):
        self._intermediate_file_writer.add_results(idxs, results)

    def write_intermediate_results(self, idx, np_result):
        self._intermediate_file_writer.write_intermediate_results(idx, np_result)

    def process(self, video_path, idx):
        # Returns a list of (idx, feature_vector) tuples
        raise NotImplementedError

    @classmethod
    def _process_segment_features(cls, batch):
        video_paths, _, idxs = batch
        assert len(idxs) == 1, f'len(idxs) = {len(idxs)}'
        return cls._processor.process(video_paths[0], idxs[0].item())

    def process_dataset(self, dataset: datasets.VFEDataset, split, processes=multiprocessing.cpu_count() - 1):
        current_records = []
        segments_iterator = iter(DataLoader(dataset.get_dataset(split), batch_size=1, shuffle=False))
        for res in self._get_pool(processes).imap_unordered(self._process_segment_features, segments_iterator):
            current_records.extend(res)
            if len(current_records) > self.batch_size:
                self._intermediate_file_writer.write_intermediate_file(current_records, dataset, split)
                current_records = []
        if len(current_records):
            self._intermediate_file_writer.write_intermediate_file(current_records, dataset, split)

    @staticmethod
    def _objects_combined(object_ids):
        return int("".join(map(str, sorted(object_ids))))

    def _get_model_file(self, dataset, split):
        model_filename = f'{self._intermediate_file_writer._file_prefix(dataset, split)}.joblib'
        return os.path.abspath(os.path.join(get_data_dir(self._intermediate_file_writer.features_dir, dataset.name(), split), model_filename))

    def _reshape_X(self, X):
        if self.feature_dim != -1:
            if len(X.shape) == 1:
                # Each sub-array is of a different length, so we have to reshape each one and vstack them.
                X = np.vstack([np.array(f).reshape((-1, self.feature_dim)) for f in X])
            else:
                X = X.reshape(-1, self.feature_dim)
        return X

    def train_bag_of_words(self, dataset: datasets.VFEDataset):
        # Get data files from metadata table for object ids.
        split = 'train'

        # Initialize KMeansMiniBatch.
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                random_state=0)

        # Read each file and perform a training batch on model.
        intermediate_results_path = self._intermediate_file_writer.get_intermediate_results(dataset, split)[0]
        X_train = zarr.open(intermediate_results_path, mode='r')
        print(f'Training MiniBatchKMeans on X with shape {X_train.shape}')
        batch_size = 4096
        chunk_size = self._intermediate_file_writer.chunk_size
        # Make sure what we read into memory is an even multiple of the training batch size.
        chunk_size = int((chunk_size // batch_size) * chunk_size)
        start_mem = 0
        start_batch = 0
        X_mem = X_train[start_mem:start_mem + chunk_size]
        while start_batch < len(X_train):
            if start_batch >= start_mem + chunk_size:
                start_mem = start_mem + chunk_size
                X_mem = X_train[start_mem:start_mem + chunk_size]
            mem_start = start_batch - start_mem
            mem_end = start_batch + batch_size - start_mem
            kmeans.partial_fit(X_mem[mem_start:mem_end])
            start_batch += batch_size

        # Save model to disk.
        model_path = self._get_model_file(dataset, split)
        print(f'Saving to {model_path}')
        joblib.dump(kmeans, model_path)

    def extract_bow_features(self, dataset, extract_split, processes=multiprocessing.cpu_count() - 1):
        model_split = 'train'
        model_path = self._get_model_file(dataset, model_split)
        n_records = len(dataset.get_labels(extract_split))
        shape = (n_records, self.featuredim())
        arr_path = create_npy_array(self.features_dir, self.filename(), dataset.name(), extract_split, shape)
        intermediates_path, index_path = self._intermediate_file_writer.get_intermediate_results(dataset, extract_split)
        step = math.ceil(n_records / processes)
        self._wait_for_imap(self._get_pool(processes).imap_unordered(
            self._process_batch,
            WrappedIterator(iter(range(0, n_records, step)), step=step, arr_path=arr_path, intermediates_path=intermediates_path, index_path=index_path, model_path=model_path)
        ))

    @classmethod
    def _process_batch(cls, wrapped_it):
        start = wrapped_it['value']
        end = start + wrapped_it['step']
        index_array = np.load(wrapped_it['index_path'], mmap_mode='r')
        max_idx = len(index_array)
        min_intermediate = int(np.min(index_array[start:end, 0]))
        max_intermediate = int(np.max(index_array[start:end, 1]))
        intermediates_array = zarr.open(wrapped_it['intermediates_path'], mode='r')
        # Read the range we need into memory once.
        needed_intermediates = intermediates_array[min_intermediate:max_intermediate]
        model = joblib.load(wrapped_it['model_path'])

        idxs = []
        features = []
        for idx in range(start, end):
            if idx >= max_idx:
                break
            idxs.append(idx)
            int_start, int_end = index_array[idx]
            # start/end are offset by min_intermediate.
            int_start = int(int_start - min_intermediate)
            int_end = int(int_end - min_intermediate)
            # Get the cluster prediction for each feature.
            clusters = model.predict(needed_intermediates[int_start:int_end])
            # agg_by_row = lambda row: np.bincount(row, minlength=cls._processor.n_clusters)
            # bow_features = np.apply_along_axis(agg_by_row, 1, clusters)
            # We don't have frame information. See how just aggregating over the entire video works for now.
            features.append(np.bincount(clusters, minlength=cls._processor.n_clusters))
        arr = open_npy_array(wrapped_it['arr_path'])
        arr[idxs] = features
        arr.flush()


class ColorSegmentProcessor(AbstractSegmentProcessor):
    def filename(self):
        return 'avg_lab'

    def featuredim(self):
        return 3

    def process(self, video_path) -> np.ndarray:
        if video_path is None:
            return
        img = videoframe.extract_median_frame(video_path)
        L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        med_L, med_a, med_b = np.median(L), np.median(a), np.median(b)
        return np.array([med_L, med_a, med_b])

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        median_frame = np.median(video_info.frames, axis=0).astype(np.uint8)
        L, a, b = cv2.split(cv2.cvtColor(median_frame, cv2.COLOR_BGR2LAB))
        med_L, med_a, med_b = np.median(L), np.median(a), np.median(b)
        return np.array([med_L, med_a, med_b])


class SpectrumHistogramProcessor(AbstractSegmentProcessor):
    def filename(self):
        return 'spectrum_histogram'

    def featuredim(self):
        return 400

    def process(self, video_path):
        img = cv2.cvtColor(videoframe.extract_median_frame(video_path), cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(img, (20, 20))
        A = np.fft.fft2(img_small)
        return np.abs(A).ravel(order='C')

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        img = cv2.cvtColor(np.median(video_info.frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(img, (20, 20))
        A = np.fft.fft2(img_small)
        return np.abs(A).ravel(order='C')


class LuminancePatternsProcessor(AbstractSegmentProcessor):
    def filename(self):
        return 'luminance_aggregates'

    def featuredim(self):
        return 2 * 45

    def _extract_luminance_deltas_from_path(self, video_path):
        video_info = videoframe.extract_raw_frames(video_path)
        return self._extract_luminance_deltas(video_info)

    def _extract_luminance_deltas(self, video_info: videoframe.VideoInfo):
        width = video_info.width
        height = video_info.height
        n_frames = video_info.nframes
        try:
            # Split the frame into a 3x3 grid.
            block_w = width // 3
            block_h = height // 3
            avg_L = np.zeros((3, 3, n_frames))
            ranks = np.zeros((9, n_frames))

            i = 0
            for frame in video_info.frames:
                L = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for y in range(3):
                    for x in range(3):
                        avg_L[y, x, i] = np.mean(
                            L[
                                y*block_h:(height if y == 2 else (y+1)*block_h),
                                x*block_w:(width if x == 2 else (x+1)*block_w)
                            ]
                        )
                # First argsort: index of grid cells in order of L value.
                # Second argsort: rank of each grid cell in terms of L value.
                ranks[:, i] = avg_L[:, :, i].flatten().argsort().argsort()
                i += 1
            delta_L = np.diff(avg_L)
            # Order: all 0th percentiles, then 25ths, …
            aggregated_deltas = np.percentile(delta_L, [0, 25, 50, 75, 100], axis=-1).flatten()
            # Order : all 0th percentiles, then 25ths, …
            aggregated_ranks = np.percentile(ranks, [0, 25, 50, 75, 100], axis=-1).flatten()
            return np.concatenate((aggregated_deltas, aggregated_ranks))
        except (cv2.error, IndexError):
            logging.error(f'(extract_luminance_deltas) Failed to read frames from video')
            return None

    def process(self, video_path):
        if video_path is None:
            return None
        return self._extract_luminance_deltas_from_path(video_path)

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        return self._extract_luminance_deltas(video_info)

class DenseOpticalFlowProcessor(AbstractSegmentProcessor):
    def __init__(self, *, stride_kw=None, **kwargs):
        super().__init__(**kwargs)
        self.stride_kw = stride_kw
        self._stridedesc = ''.join([f'{k}{v}' for k, v in self.stride_kw.items()])
        self._stride = datasets.frame.CreateStride(**self.stride_kw)

    def filename(self):
        return 'agg_optical_flow_' + self._stridedesc

    def featuredim(self):
        return 10

    def process(self, video_path) -> np.ndarray:
        if video_path is None:
            return None
        try:
            cap = videoframe.extract_fps(video_path, self._stride)
            frame1 = next(cap)
            prvs = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (consts.RESIZE_WIDTH, consts.RESIZE_HEIGHT))
            dxdy = np.zeros((consts.RESIZE_HEIGHT, consts.RESIZE_WIDTH, 2), dtype=np.float32)
            for frame2 in cap:
                next_frame = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (consts.RESIZE_WIDTH, consts.RESIZE_HEIGHT))
                flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                np.add(dxdy[..., 0], flow[..., 0], out=dxdy[..., 0], dtype=np.float32)
                np.add(dxdy[..., 1], flow[..., 1], out=dxdy[..., 1], dtype=np.float32)
                prvs = next_frame

            qxs = np.percentile(dxdy[..., 0], [0, 25, 50, 75, 100])
            qys = np.percentile(dxdy[..., 1], [0, 25, 50, 75, 100])
            return np.concatenate((qxs, qys))

        except Exception as e:
            logging.error(f'Failed to extract motion information from video at {video_path}: {e}')
            return None

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        step = self._stride.step(video_info.fps, video_info.nframes)
        for frame_idx in range(0, video_info.max_frame, step):
            if frame_idx == 0:
                prvs = cv2.cvtColor(video_info.frames[frame_idx], cv2.COLOR_BGR2GRAY)
                dxdy = np.zeros((video_info.height, video_info.width, 2), dtype=np.float32)
                continue
            next_frame = cv2.cvtColor(video_info.frames[frame_idx], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            np.add(dxdy[..., 0], flow[..., 0], out=dxdy[..., 0], dtype=np.float32)
            np.add(dxdy[..., 1], flow[..., 1], out=dxdy[..., 1], dtype=np.float32)
            prvs = next_frame

        qxs = np.percentile(dxdy[..., 0], [0, 25, 50, 75, 100])
        qys = np.percentile(dxdy[..., 1], [0, 25, 50, 75, 100])
        return np.concatenate((qxs, qys))


class SiftProcessor(BagOfWordsAbstractSegmentProcessor):
    @classmethod
    def create(cls, **kwargs):
        return SiftProcessor(feature_dim=32, chunk_size=8e6, dtype=np.uint8, **kwargs)

    def __init__(self, *, stride_kw=None, **kwargs):
        self.stride_kw = stride_kw
        self._stridedesc = ''.join([f'{k}{v}' for k, v in self.stride_kw.items()])
        self._stride = datasets.frame.CreateStride(**self.stride_kw)
        super().__init__(**kwargs)

    def filename(self):
        return 'orb_descriptors_' + self._stridedesc

    def featuredim(self):
        return self.n_clusters

    def process(self, video_path, idx):
        results = []
        for frame in videoframe.extract_fps(video_path, stride=self._stride):
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (consts.RESIZE_WIDTH, consts.RESIZE_HEIGHT))
            sift = cv2.SIFT_create()
            _, desc = sift.detectAndCompute(gray, None)
            assert (len(desc.ravel(order='C')) % self.feature_dim) == 0, f'{len(desc.ravel(order="C"))} % {self.feature_dim} != 0'
            results.append((idx, desc.ravel(order='C')))
        return results

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        good_descriptors = []
        step = self._stride.step(video_info.fps, video_info.nframes)
        # prv = None
        try:
        # From https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
            # bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING)
            for frame_idx in range(0, video_info.max_frame, step):
                gray = cv2.cvtColor(video_info.frames[frame_idx], cv2.COLOR_BGR2GRAY)
                orb = cv2.ORB_create()
                _, cur = orb.detectAndCompute(gray, None)
                if cur is not None:
                    good_descriptors.append(cur)
            return np.vstack(good_descriptors)
        except Exception as e:
            print('Error:', e)
            return np.zeros((1, self.feature_dim))


class HogProcessor(BagOfWordsAbstractSegmentProcessor):
    @classmethod
    def create(cls, **kwargs):
        # previously: feature_dim = 36
        return HogProcessor(feature_dim=3780, chunk_size=2e4, **kwargs)

    def __init__(self, *, stride_kw=None, **kwargs):
        self.stride_kw = stride_kw
        self._stridedesc = ''.join([f'{k}{v}' for k, v in self.stride_kw.items()])
        self._stride = datasets.frame.CreateStride(**self.stride_kw)
        super().__init__(**kwargs)

    def filename(self):
        return 'hog_bow_descriptors_' + self._stridedesc

    def featuredim(self):
        return self.n_clusters

    def process(self, video_path, idx):
        frames = videoframe.extract_fps(video_path, self._stride)
        # Descriptor shape: 3780 when size is 64x128.
        results = []
        for img in frames:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog = cv2.HOGDescriptor()
            hog_desc = hog.compute(cv2.resize(gray, (64, 128)))
            results.append((idx, hog_desc.ravel(order='C')))
        return results

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        step = self._stride.step(video_info.fps, video_info.nframes)
        results = []
        for frame_idx in range(0, video_info.max_frame, step):
            gray = cv2.cvtColor(video_info.frames[frame_idx], cv2.COLOR_BGR2GRAY)
            hog = cv2.HOGDescriptor()
            hog_desc = hog.compute(cv2.resize(gray, (64, 128)))
            results.append(hog_desc.ravel(order='C'))
        return np.vstack(results)

class PoTHogProcessor(PerFrameAbstractSegmentProcessor):
    @classmethod
    def create(cls, **kwargs):
        # batch_size=256, filesize ~ 85MB
        return PoTHogProcessor(chunk_size=5e4, **kwargs)

    def __init__(self, *, stride_kw=None, **kwargs):
        # When we initialize with super(), it will call _filename(), so we need to initialize _stridedesc.
        self.stride_kw = stride_kw
        self._stridedesc = ''.join([f'{k}{v}' for k, v in self.stride_kw.items()])
        self._stride = datasets.frame.CreateStride(**self.stride_kw)
        super().__init__(**kwargs)

    def base_feature_dim(self):
        # ((128-32)/16 + 1)**2 * 4 cells * 9 hist buckets
        return 1764

    def _filename(self, agg: PerFrameAbstractSegmentProcessor.AggregationType):
        base = 'hog_pf_' + self._stridedesc
        if agg == PerFrameAbstractSegmentProcessor.AggregationType.MEAN:
            return base + '_mean'
        if agg == PerFrameAbstractSegmentProcessor.AggregationType.MAX:
            return base + '_max'
        elif agg == PerFrameAbstractSegmentProcessor.AggregationType.POT:
            return base + '_pot'
        else:
            return base

    def _center_square_crop(self, image):
        xy_dim = image.shape[:2]
        min_dim = min(xy_dim) // 2
        cy, cx = [d // 2 for d in xy_dim]
        return image[cy-min_dim:cy+min_dim, cx-min_dim:cx+min_dim]

    def process(self, video_path, idx):
        hog_descs = []
        for frame in videoframe.extract_fps(video_path, self._stride):
            gray = cv2.cvtColor(self._center_square_crop(frame), cv2.COLOR_BGR2GRAY)
            hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
            hog_desc = hog.compute(cv2.resize(gray, (128, 128)))
            hog_descs.append(hog_desc)
        if len(hog_descs):
            return (idx, np.vstack(hog_descs).ravel(order='C').tolist(), len(hog_descs))
        else:
            return []

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        hog_descs = []
        step = self._stride.step(video_info.fps, video_info.nframes)
        for frame_idx in range(0, video_info.max_frame, step):
            gray = cv2.cvtColor(self._center_square_crop(video_info.frames[frame_idx]), cv2.COLOR_BGR2GRAY)
            hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
            hog_desc = hog.compute(cv2.resize(gray, (128, 128)))
            hog_descs.append(hog_desc)
        raw = np.vstack(hog_descs)
        return {
            self.AggregationType.MEAN.value: np.mean(raw, axis=0),
            self.AggregationType.MAX.value: np.max(raw, axis=0),
            self.AggregationType.POT.value: self._pot.extract_features(preprocessing.normalize(raw, norm='l1')),
        }

class PoTOpticalFlowProcessor(PerFrameAbstractSegmentProcessor):
    @classmethod
    def create(cls, **kwargs):
        # batch_size=28, file size ~ 10 mb
        return PoTOpticalFlowProcessor(w_d=5, h_d=5, o_d=8, chunk_size=5e4, **kwargs)

    def __init__(self, w_d, h_d, o_d, *, stride_kw=None, **kwargs):
        self.stride_kw = stride_kw
        self._stridedesc = ''.join([f'{k}{v}' for k, v in self.stride_kw.items()])
        self._stride = datasets.frame.CreateStride(**self.stride_kw)
        self.w_d = w_d
        self.h_d = h_d
        self.o_d = o_d
        super().__init__(**kwargs)

    def base_feature_dim(self):
        return self.w_d * self.h_d * self.o_d

    def _filename(self, agg: PerFrameAbstractSegmentProcessor.AggregationType):
        base = 'opt_pf_' + self._stridedesc
        if agg == PerFrameAbstractSegmentProcessor.AggregationType.MEAN:
            return base + '_mean'
        elif agg == PerFrameAbstractSegmentProcessor.AggregationType.MAX:
            return base + '_max'
        elif agg == PerFrameAbstractSegmentProcessor.AggregationType.POT:
            return base + '_pot'
        else:
            return base

    def process(self, video_path, idx):
        optical_flow_hists = self._pot.get_optical_histograms(video_path, w_d=self.w_d, h_d=self.h_d, o_d=self.o_d, stride=self._stride)
        if not len(optical_flow_hists):
            logging.debug(f'Error extracting optical flow hists from {idx}')
            return []
        h = np.vstack([h.ravel(order='C') for h in optical_flow_hists])
        return (idx, h.ravel(order='C').tolist(), len(optical_flow_hists))

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        optical_flow_hists = self._pot.get_optical_histograms_from_info(video_info, w_d=self.w_d, h_d=self.h_d, o_d=self.o_d, stride=self._stride)
        if not len(optical_flow_hists):
            logging.debug(f'Error extracting optical flow hists from {idx}')
            return []
        raw = np.vstack([h.ravel(order='C') for h in optical_flow_hists])
        return {
            self.AggregationType.MEAN.value: np.mean(raw, axis=0),
            self.AggregationType.MAX.value: np.max(raw, axis=0),
            self.AggregationType.POT.value: self._pot.extract_features(preprocessing.normalize(raw, norm='l1')),
        }

class TimeOfDayProcessor(AbstractSegmentProcessor):
    def filename(self):
        return 'timeofday'

    def featuredim(self):
        return 1

class TimeOfDayProcessor(AbstractSegmentProcessor):
    def filename(self):
        return 'datetime'

    def featuredim(self):
        return 1
