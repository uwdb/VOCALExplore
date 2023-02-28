import logging
import math
import numpy as np
import re
from scipy import stats
from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo, clipinfo_to_clipset
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

from .randomexp import RandomExplorer
from .coresets import CoresetsExplorer

class AbstractRandomIfUniformExplorer(AbstractExplorer):
    def __init__(self, rng=None, missing_vids_X=-1, **kwargs):
        self.rng = np.random.default_rng(rng)
        self.random_explorer = RandomExplorer(rng=rng, **kwargs)
        self.coresets_explorer = CoresetsExplorer(rng=rng, missing_vids_X=missing_vids_X)
        self.logger = logging.getLogger(__name__)
        self._uniform_labels = True

    def _labels_are_uniform(self, label_to_time):
        raise NotImplementedError

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        if self._labels_are_uniform(modelmanager.get_total_label_time()):
            self.logger.debug('Random exploration because labels are uniform')
            return self.random_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids)
        else:
            self.logger.debug('Coresets exploration because labels are not uniform')
            return self.coresets_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=None, vids=None)

class AKSRandomIfUniformExplorer(AbstractRandomIfUniformExplorer):
    def __init__(self, pval=0.001, rng=None, **kwargs):
        super().__init__(rng=rng, **kwargs)
        self.pval = pval

    def _labels_are_uniform(self, label_to_time):
        label_counts = [math.ceil(seconds) for seconds in label_to_time.values()]
        nclasses = len(label_counts)
        if nclasses <= 1:
            return True
        samples_from_counts = np.concatenate([np.array([i] * c) for i, c in enumerate(label_counts)])
        stat_value = stats.anderson_ksamp([
            samples_from_counts,
            self.rng.integers(0, nclasses, size=max(nclasses, 200))
        ])[-1]
        self.logger.debug(f'Uniformity stat: {stat_value}')
        return stat_value > self.pval

class CVMRandomIfUniformExplorer(AbstractRandomIfUniformExplorer):
    def __init__(self, pval=0.001, rng=None):
        super().__init__(rng=rng)
        self.pval = pval

    def _labels_are_uniform(self, label_to_time):
        label_counts = [math.ceil(seconds) for seconds in label_to_time.values()]
        nclasses = len(label_counts)
        if nclasses <= 1:
            return True
        samples_from_counts = np.concatenate([np.array([i] * c) for i, c in enumerate(label_counts)])
        stat_value = stats.cramervonmises(
            samples_from_counts,
            'randint',
            args=(0, nclasses)
        ).pvalue
        self.logger.debug(f'Uniformity stat: {stat_value}')
        return stat_value > self.pval

def RandomIfUniformExplorerFromName(cls, name, **kwargs):
    try:
        pval = float(re.search(r'pval([\d\.]+)', name)[1])
        return cls(pval=pval, **kwargs)
    except:
        logging.warn('Creating explorer with default pval because regex failed')
        # Use default pvalue if we can't match on the name.
        return cls(**kwargs)
