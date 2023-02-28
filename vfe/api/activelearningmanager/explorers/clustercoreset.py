import logging
import numpy as np
from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer
from . import cluster, coresets

class ClusterCoresetsExplorer(AbstractExplorer):
    def __init__(self, rng=None, cluster_rounds=3):
        self.cluster_rounds = cluster_rounds
        self.nrounds = 0
        self.rng = np.random.default_rng(rng)
        self.logger = logging.getLogger(__name__)

        self._clusterer = cluster.ClusterExplorer(rng=rng)
        self._coresets = coresets.CoresetsExplorer(rng=rng)
        self._cluster_clips = None

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        self.nrounds += 1
        if self.nrounds > self.cluster_rounds:
            self.logger.debug(f'Coresets explore (features {feature_names}) (nrounds={self.nrounds})')
            return self._coresets.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids)

        # Else, we're in the rounds where we want to use clustering. Since we know how many rounds
        # we'll cluster, try assuming k will be the same for each of these rounds and cache the
        # cluster centers if we create cluster_rounds * k clusters.
        # If k decreases, we'll use a subset of these. If k increases, we'll re-cluster.
        if self._cluster_clips is None or len(self._cluster_clips) < k:
            nclusters = (self.cluster_rounds - self.nrounds + 1) * k
            self.logger.debug(f'Cluster explore (features {feature_names}) with nclusters={nclusters}')
            self._cluster_clips = self._clusterer.explore(feature_names, featuremanager, modelmanager, videomanager, nclusters, t, label=label, vids=vids)

        assert len(self._cluster_clips) >= k, f'{len(self._cluster_clips)} < {k}'
        self.logger.debug(f'Cluster explore (features {feature_names}) returning the first {k} cached clips')
        return_clips = self._cluster_clips[:k]
        self._cluster_clips = self._cluster_clips[k:]
        return return_clips
