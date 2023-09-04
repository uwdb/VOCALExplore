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
from .clustermargin import ClusterMarginExplorer

logger = logging.getLogger(__name__)

class AbstractUniformTester:
    def __init__(self, pval=0.001, m=None, rng=None):
        self.pval = pval
        self.m = m
        self.rng = np.random.default_rng(rng)

    def labels_are_uniform(self, label_to_time):
        raise NotImplementedError

class RandomIfUniformExplorer(AbstractExplorer):
    def __init__(self, uniform_tester: AbstractUniformTester, rng=None, missing_vids_X=-1, **kwargs):
        self.uniform_tester = uniform_tester
        self.rng = np.random.default_rng(rng)
        self.random_explorer = RandomExplorer(rng=rng, **kwargs)
        self.coresets_explorer = CoresetsExplorer(rng=rng, missing_vids_X=missing_vids_X)
        self._uniform_labels = True

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None, step=None) -> Iterable[ClipInfo]:
        if self.uniform_tester.labels_are_uniform(modelmanager.get_total_label_time()):
            logger.debug('Random exploration because labels are uniform')
            return self.random_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids, step=step)
        else:
            logger.debug('Coresets exploration because labels are not uniform')
            return self.coresets_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=None, vids=None, step=step)

class RandomIfUniformCMExplorer(AbstractExplorer):
    def __init__(self, uniform_tester: AbstractUniformTester, rng=None, missing_vids_X=-1, **kwargs):
        self.uniform_tester = uniform_tester
        self.rng = np.random.default_rng(rng)
        self.random_explorer = RandomExplorer(rng=rng, **kwargs)
        self.clustermargin_explorer = ClusterMarginExplorer(rng=rng, missing_vids_X=missing_vids_X, **kwargs)
        self._uniform_labels = True

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None, step=None) -> Iterable[ClipInfo]:
        if self.uniform_tester.labels_are_uniform(modelmanager.get_total_label_time()):
            logger.debug('Random exploration because labels are uniform')
            return self.random_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids, step=step)
        else:
            logger.debug('Cluster-Margin exploration because labels are not uniform')
            return self.clustermargin_explorer.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=None, vids=None, step=step)

class AKSUniformTester(AbstractUniformTester):
    def labels_are_uniform(self, label_to_time):
        label_counts = [math.ceil(seconds) for seconds in label_to_time.values()]
        nclasses = len(label_counts)
        if nclasses <= 1:
            return True
        samples_from_counts = np.concatenate([np.array([i] * c) for i, c in enumerate(label_counts)])
        stat_value = stats.anderson_ksamp([
            samples_from_counts,
            self.rng.integers(0, nclasses, size=max(nclasses, 200))
        ])[-1]
        logger.debug(f'Uniformity stat: {stat_value}')
        return stat_value > self.pval

class CVMUniformTester(AbstractUniformTester):
    def labels_are_uniform(self, label_to_time):
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
        logger.debug(f'Uniformity stat: {stat_value}')
        return stat_value > self.pval

class BinomialUniformTester(AbstractUniformTester):
    def labels_are_uniform(self, label_to_time):
        # Multiply by 1000 for frame-level labels that don't accumulate to more than one second.
        label_counts = [math.ceil(seconds * 1000) for seconds in label_to_time.values()]
        nclasses = len(label_counts)
        if nclasses <= 1:
            return True

        n = np.sum(label_counts)
        min_c = np.min(label_counts)
        k = len(label_counts)
        stat_value = k * stats.binom.cdf(min_c, n, 1/(k * self.m))
        logger.debug(f'Uniformity stat: {stat_value}')
        return stat_value > self.pval


def RandomIfUniformExplorerFromName(cls, name, **kwargs):
    if name.startswith('cvm'):
        uniform_tester_cls = CVMUniformTester
    elif name.startswith('binom'):
        uniform_tester_cls = BinomialUniformTester
    else:
        uniform_tester_cls = AKSUniformTester
    rng = kwargs.get('rng', None)

    if uniform_tester_cls == BinomialUniformTester:
        # The name must specify m and pval.
        pval = float(re.search(r'pval([\d\.]+)', name)[1])
        m = float(re.search(r'_m([\d\.]+)', name)[1])
        logger.debug(f'Creating binomial tester with pval={pval}, m={m}')
        uniform_tester = uniform_tester_cls(pval=pval, m=m, rng=rng)
    else:
        try:
            pval = float(re.search(r'pval([\d\.]+)', name)[1])
            uniform_tester = uniform_tester_cls(pval=pval, rng=rng)
        except:
            logger.warn('Creating explorer with default pval because regex failed')
            uniform_tester = uniform_tester_cls(rng=rng)
    return cls(uniform_tester, **kwargs)
