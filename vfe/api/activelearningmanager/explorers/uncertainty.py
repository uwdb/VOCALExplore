from typing import Iterable

from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer
from vfe.api.activelearningmanager.explorers import LabelUncertaintyExplorer

class UncertaintyExplorer(AbstractExplorer):
    def __init__(self, nolabel_explorer, feature_name, rng=None):
        self.nolabel = nolabel_explorer
        self.label = LabelUncertaintyExplorer(feature_name, rng)

    def explore(self, featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        explorer = self.nolabel if label is None else self.label
        return explorer.explore(featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids)
