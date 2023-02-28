import logging
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Iterable

from vfe import core
from vfe.api.storagemanager import ClipInfo, clipinfo_to_clipset
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

class DBScanExplorer(AbstractExplorer):
    def __init__(self, feature_name, rng=None):
        self.feature_name = feature_name
        self.random_state = np.random.RandomState(rng)
        self.logger = logging.getLogger(__name__)

    def explore(self, featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        self.logger.info(f'DBScan explore')
        return []
