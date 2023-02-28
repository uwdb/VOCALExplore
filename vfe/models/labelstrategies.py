import math
import random

import numpy as np
from scipy.spatial import distance
from sklearn_extra import cluster

from .abstractlabeler import AbstractLabeler, Dataset


def budget_to_int(dataset: Dataset, budget_fraction):
    return math.ceil(budget_fraction * len(dataset.X))

class RandomLabeler(AbstractLabeler):
    @classmethod
    def name(cls):
        return 'random'

    def label_progress(self, dataset: Dataset, train_size):
        return [np.random.choice(dataset.fids, budget_to_int(dataset, train_size), replace=False)]

class Random30SecondLabeler(AbstractLabeler):
    @classmethod
    def name(cls):
        return 'random30sec'

    def label_progress(self, dataset: Dataset, train_size):
        train_size = budget_to_int(dataset, train_size)
        labeled_idxs = set()
        not_labeled_idxs = set(range(len(dataset.fids)))
        while len(labeled_idxs) < train_size:
            new_idx = random.sample(not_labeled_idxs, 1)[0] # Sample returns an array.
            # Add new_idx plus enough fids afterward so the total duration is 30 seconds.
            start_time = dataset.start_times[new_idx]
            end_time = start_time + np.timedelta64(30, 's')
            # Add all fids whose start time is >= start_time and end time is <= end_time.
            og_not_labeled_idxs = not_labeled_idxs.copy()
            for i in og_not_labeled_idxs:
                if dataset.start_times[i] >= start_time and dataset.end_times[i] <= end_time:
                    labeled_idxs.add(i)
                    not_labeled_idxs.remove(i)
        return [dataset.fids[list(labeled_idxs)]]

class TemporalConsecutiveLabeler(AbstractLabeler):
    @classmethod
    def name(cls):
        return 'temporalconsecutive'

    def label_progress(self, dataset: Dataset, train_size):
        sort_idx = np.argsort(dataset.start_times)
        train_size = budget_to_int(dataset, train_size)
        return [dataset.fids[sort_idx[:train_size]]]

class KMedoidsLabeler(AbstractLabeler):
    @classmethod
    def name(cls):
        return 'kmedoids'

    def label_progress(self, dataset: Dataset, train_size):
        n_clusters = budget_to_int(dataset, train_size)
        clusterer = cluster.KMedoids(n_clusters=n_clusters, init='k-medoids++', metric='euclidean', method='alternate').fit(dataset.X)
        return [dataset.fids[clusterer.medoid_indices_]]

class KMedoids30SecLabeler(AbstractLabeler):
    @classmethod
    def name(cls):
        return 'kmedoids30sec'

    def _try_cluster(self, dataset: Dataset, n_clusters):
        clusterer = cluster.KMedoids(n_clusters=n_clusters, init='k-medoids++', metric='euclidean', method='alternate').fit(dataset.X)
        labeled_idxs = set()
        not_labeled_idxs = set(range(len(dataset.fids)))
        for medoid_idx in clusterer.medoid_indices_:
            labeled_idxs.add(medoid_idx)
            start_time = dataset.start_times[medoid_idx]
            end_time = start_time + np.timedelta64(30, 's')
            # Add all fids whose start time is >= start_time and end time is <= end_time.
            og_not_labeled_idxs = not_labeled_idxs.copy()
            for i in og_not_labeled_idxs:
                if dataset.start_times[i] >= start_time and dataset.end_times[i] <= end_time:
                    labeled_idxs.add(i)
                    not_labeled_idxs.remove(i)
        return [dataset.fids[list(labeled_idxs)]]

    def label_progress(self, dataset: Dataset, train_size):
        train_size = budget_to_int(dataset, train_size)
        # Binary search to find the best number of clusters between train_size and 1.
        min_clusters = 1
        max_clusters = train_size
        n_current = -1
        deltas = []
        fids = []
        while n_current != train_size and abs(max_clusters - min_clusters) > 1:
            n_clusters = min_clusters + (max_clusters - min_clusters) // 2
            current_fids = self._try_cluster(dataset, n_clusters)
            n_current = len(current_fids[0])

            fids.append(current_fids)
            delta = abs(n_current - train_size) / train_size
            deltas.append(delta)

            if n_current > train_size:
                # If there are more fids than desired, create fewer clusters.
                max_clusters = n_clusters - 1
            else:
                # If there are fewer fids than desired, create more clusters.
                min_clusters = n_clusters + 1

        return fids[np.argmin(deltas)]

class CoresetsLabeler(AbstractLabeler):
    # Assumes that label_progress() will be called for each budget for a
    # single experiment at a time before reset() is called.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labeled_idxs = np.array([])
        self._distance_matrix = None

    def reset_dataset(self, dataset: Dataset):
        self._distance_matrix = distance.squareform(distance.pdist(dataset.X))

    def reset_experiment(self):
        self._labeled_idxs = np.array([])
        self._min_distance = None

    @classmethod
    def name(cls):
        return 'greedycoresets'

    def label_progress(self, dataset: Dataset, train_size):
        coreset_size = budget_to_int(dataset, train_size)
        self._labeled_idxs = self._greedycoreset(self._labeled_idxs, coreset_size)
        assert self._labeled_idxs.size == coreset_size, f'{self._labeled_idxs.size} != {coreset_size}'
        return [dataset.fids[self._labeled_idxs]]

    def _greedycoreset(self, labeled_idxs, return_size):
        if labeled_idxs.size == 0:
            labeled_idxs = np.array([np.random.choice(len(self._distance_matrix))])
            self._min_distance = self._distance_matrix[:, labeled_idxs].flatten()
        while len(labeled_idxs) < return_size:
            new_idx = np.argmax(self._min_distance)
            self._min_distance = np.minimum(self._min_distance, self._distance_matrix[:, new_idx])
            labeled_idxs = np.append(labeled_idxs, new_idx)
        return labeled_idxs

def get_labeler(strategy, *args, **kwargs):
    for sample_cls in [
        RandomLabeler,
        Random30SecondLabeler,
        TemporalConsecutiveLabeler,
        KMedoidsLabeler,
        KMedoids30SecLabeler,
        CoresetsLabeler,
    ]:
        if strategy == sample_cls.name():
            return sample_cls(*args, **kwargs)
    assert False, f'Unrecognized labeler: {strategy}'
