from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager

class AbstractExplorer:
    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        raise NotImplementedError
