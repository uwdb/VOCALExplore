import math
import multiprocessing
import numpy as np
import os
from torch.utils.data import DataLoader
from typing import List, Iterable

from vfe import datasets
from vfe.core import filesystem, videoframe


def get_data_dir(features_dir, dataset_name, split):
    return filesystem.create_dir(features_dir, dataset_name, split)

def get_array_path(data_dir, filename, type='npy'):
    if type == 'npy':
        ext = '.npy'
    elif type == 'zarr':
        ext = '.zarr'
    else:
        ext = '.sync'
    return os.path.join(data_dir, filename + ext)

def zarr_arr_path(features_dir, filename, dataset_name, split):
    data_dir = get_data_dir(features_dir, dataset_name, split)
    arr_path = get_array_path(data_dir, filename, type='zarr')
    sync_path = get_array_path(data_dir, filename, type='sync')
    return arr_path, sync_path

def npy_arr_path(features_dir, filename, dataset_name, split):
    return get_array_path(get_data_dir(features_dir, dataset_name, split), filename, type='npy')

# def create_zarr_array(features_dir, filename, dataset_name, split, shape, chunk_size, dtype=np.float32, sync=True):
    # arr_path, sync_path = zarr_arr_path(features_dir, filename, dataset_name, split)
    # synchronizer = zarr.ProcessSynchronizer(sync_path) if sync else None
    # arr = zarr.open(arr_path, mode='w', shape=shape, chunks=(chunk_size, None), dtype=dtype, synchronizer=synchronizer)
    # return arr_path, sync_path

def create_npy_array(features_dir, filename, dataset_name, split, shape, dtype=np.float32):
    arr_path = npy_arr_path(features_dir, filename, dataset_name, split)
    np.save(arr_path, np.zeros(shape=shape, dtype=dtype))
    return arr_path

# def open_zarr_array(arr_path, sync_path):
#     synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path else None
#     return zarr.open(arr_path, mode='r+', synchronizer=synchronizer)

def open_npy_array(arr_path):
    return np.load(arr_path, mmap_mode='r+')

def get_zarr_array(features_dir, filename, dataset_name, split, sync=True):
    arr_path, sync_path = zarr_arr_path(features_dir, filename, dataset_name, split)
    return open_zarr_array(arr_path, sync_path if sync else None)

def get_npy_array(features_dir, filename, dataset_name, split):
    return open_npy_array(npy_arr_path(features_dir, filename, dataset_name, split))

class WrappedIterator(Iterable):
    def __init__(self, iterator, **kwargs):
        self.iterator = iterator
        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        return {
            'value': next(self.iterator),
            **self.kwargs
        }

class AbstractSegmentProcessor:
    def __init__(self, *, features_dir=None, chunk_size=4096):
        self.features_dir = features_dir
        self._pool = None
        self._pool_processes = None
        self.chunk_size = chunk_size

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def filename(self):
        raise NotImplementedError

    def featuredim(self):
        raise NotImplementedError

    def initialize_array(self, dataset_name, split):
        self._opened_array = get_npy_array(self.features_dir, self.filename(), dataset_name, split)

    def ensure_array_exists(self, dataset: datasets.VFEDataset, split):
        arr_path = npy_arr_path(self.features_dir, self.filename(), dataset.name(), split)
        if not os.path.exists(arr_path):
            self.reset_array(dataset, split)

    def reset_array(self, dataset: datasets.VFEDataset, split):
        n_records = len(dataset.get_labels(split))
        create_npy_array(self.features_dir, self.filename(), dataset.name(), split, shape=(n_records, self.featuredim()))

    def verify_array(self, dataset: datasets.VFEDataset, split):
        arr = get_npy_array(self.features_dir, self.filename(), dataset.name(), split)
        for idx, row in enumerate(arr):
            assert not np.all(row == 0), f'({self.filename()}) Row {idx} not set'

    def process_video(self, video_info: videoframe.VideoInfo, idx):
        raise NotImplementedError

    def add_results(self, idxs, results):
        idxs_filtered = [idx for i, idx in enumerate(idxs) if results[i] is not None]
        if len(idxs_filtered) != len(idxs):
            # Re-create results array so that it has the correct dimensions instead of being an object array.
            idxs = np.array(idxs_filtered)
            results = np.array([r for r in results if r is not None])
        order = np.argsort(idxs)
        self._opened_array[idxs[order]] = results[order]
        self._opened_array.flush()

    def process(self, video_path) -> np.ndarray:
        raise NotImplementedError

    def load_data(self, dataset: datasets.VFEDataset, split):
        X_path = os.path.join(self.features_dir, f'{self.filename()}_{split}.npy')
        labels = dataset.get_labels(split)
        return np.memmap(X_path, mode='r', shape=(len(labels), self.featuredim())), labels

    @staticmethod
    def _wait_for_imap(result):
        for res in result:
            continue

    @classmethod
    def _initialize_processor(cls, kwargs={}):
        cls._processor = cls(**kwargs)

    @classmethod
    def _process_segment_features(cls, wrapped_it):
        batch = wrapped_it['value']
        arr = open_npy_array(wrapped_it['arr_path'])
        video_paths, _, idxs = batch
        for path, idx in zip(video_paths, idxs):
            idx = idx.item()
            result = cls._processor.process(path)
            arr[idx] = result if result is not None else -1
        arr.flush()

    def _get_pool(self, processes):
        if self._pool and processes == self._pool_processes:
            return self._pool
        self._pool = multiprocessing.Pool(processes, initializer=self._initialize_processor, initargs=({k:v for k, v in self.__dict__.items() if not k.startswith('_')},))
        self._pool_processes = processes
        return self._pool

    def process_dataset(self, dataset: datasets.VFEDataset, split, processes=multiprocessing.cpu_count() - 1):
        # Create subdirectory under features directory for the dataset.
        n_records = len(dataset.get_labels(split))
        arr_path = create_npy_array(self.features_dir, self.filename(), dataset.name(), split, (n_records, self.featuredim()))
        segments_iterator = iter(DataLoader(dataset.get_dataset(split), batch_size=1, shuffle=False))
        self._wait_for_imap(self._get_pool(processes).imap_unordered(
            self._process_segment_features,
            WrappedIterator(segments_iterator, arr_path=arr_path),
        ))
