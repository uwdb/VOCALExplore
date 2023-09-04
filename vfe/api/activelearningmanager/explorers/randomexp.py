import logging
import numpy as np
import pyarrow.compute as pc
from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo, clipinfo_to_clipset
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

class RandomExplorer(AbstractExplorer):
    def __init__(self, rng=None, limit_to_extracted=False):
        self.rng = np.random.default_rng(rng)
        self.limit_to_extracted = limit_to_extracted
        self.logger = logging.getLogger(__name__)

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None, step=None) -> Iterable[ClipInfo]:
        if vids is None:
            vids_with_features, vids_without_features = featuremanager.get_extracted_features_info(feature_names)
            labeled_vids = set(modelmanager.get_vids_with_labels())
            unlabeled_vids_with_features = set(vids_with_features) - labeled_vids
            self.logger.debug(f'{len(unlabeled_vids_with_features)} unlabeled vids with features')
            if self.limit_to_extracted and len(unlabeled_vids_with_features) >= k:
                self.logger.debug(f'Random explore (features {feature_names}) over unlabeled vids with features already extracted')
                vids = unlabeled_vids_with_features
            else:
                vids = videomanager.get_all_vids()
        else:
            self.logger.debug(f'Random explore (features {feature_names}) over vids {vids}')

        self.logger.debug(f'Got candidate vids')
        clip_duration = 1
        # vids must be converted from a ndarray to a tuple of ints so that it is hashable.
        clips: Iterable[ClipInfo] = list(videomanager.get_clip_splits(tuple(int(vid) for vid in vids), clip_duration=clip_duration))
        self.logger.debug('Got clips')
        labeled_clips = modelmanager.get_labels_for_clips(clipinfo_to_clipset(clips))
        # Filter to rows without labels.
        unlabeled_clips = labeled_clips.filter(pc.equal(labeled_clips['labels'], 'none'))
        self.logger.debug('Filtered to unlabeled clips')

        sampled_idxs = self.rng.choice(len(unlabeled_clips), size=k, replace=False)
        # For now don't worry about overlapping physical clips. If the dataset size is large relative
        #   to the number of labeled segments, overlap should be rare.
        sampled = unlabeled_clips.take(sampled_idxs).to_pylist()
        for c in sampled:
            self.logger.debug(f'Adding clip ({c["vid"]}, {c["start_time"]:.2f}-{c["end_time"]:.2f})')
        sampled_videos =  [
            ClipInfo(vid=c['vid'], vstart=None, start_time=c['start_time'], end_time=c['end_time'])
            for c in sampled
        ]
        return sampled_videos
