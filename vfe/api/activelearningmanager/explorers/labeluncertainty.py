from collections import defaultdict
import logging
import numpy as np
from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo, clips_overlap
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager, PredictionSet
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

class LabelUncertaintyExplorer(AbstractExplorer):
    def __init__(self, rng=None, threshold=-1):
        self.rng = np.random.default_rng(rng)
        self.logger = logging.getLogger(__name__)
        self._shown_fids = defaultdict(set) # vid -> start times shown
        self.threshold = threshold

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        self.logger.info('LabelUncertaintyExplorer explore (feature {feature_name})')
        if label is None:
            raise RuntimeError('LabelUncertaintyExplorer requires label to be specified')

        # Make sure there are enough features extracted.
        vids_with_features, vids_without_features = featuremanager.get_extracted_features_info(feature_names)
        labeled_vids = set(modelmanager.get_vids_with_labels())
        # Find how many vids with features are unlabeled. These are the ones we're most interested in for labeling.
        unlabeled_vids_with_features = set(vids_with_features) - labeled_vids
        self.logger.debug(f'_model_topk: {len(vids_with_features)} vids with features extracted, {len(unlabeled_vids_with_features)} unlabeled vids with features extracted, {len(vids_without_features)} vids without features extracted')
        if len(unlabeled_vids_with_features) < k:
            candidate_vids = set(vids_without_features) - set(labeled_vids)
            vids_to_extract = self.rng.choice(np.array([*candidate_vids]), size=min(k, len(candidate_vids)), replace=False)
            self.logger.info(f'Before picking model topk, extracting features from {len(vids_to_extract)} new vids')
            featuremanager.get_features(feature_names, vids_to_extract)

        # Get model predictions over all extracted features that aren't already labeled.
        # Make sure we use a model that can predict the specified label.
        # If vids is None, we'll get predictions over all stored features.
        predictions: Iterable[PredictionSet] = modelmanager.get_predictions(vids=vids, start=None, end=None, feature_names=feature_names, ignore_labeled=True)

        label_counts = modelmanager.get_label_counts(feature_names)
        self.logger.debug(f'Label counts: {label_counts}')
        npos = label_counts[label] if label in label_counts else 0
        nneg = sum(v for k, v in label_counts.items() if k != label)
        if npos < nneg:
            sort_by = 'most_certain'
        else:
            sort_by = 'most_uncertain'

        if self.threshold > 0 and npos > self.threshold:
            sort_by = 'most_uncertain'

        # Use sort_by to pick the ones to return.
        nlabels = len(predictions[0].predictions)
        if sort_by == 'most_certain':
            baseline_prob = 1.0
        elif sort_by == 'most_uncertain':
            baseline_prob = 1.0 / nlabels
        else:
            assert False, f'_model_topk: unrecognized sort_by {sort_by}'

        self.logger.debug(f'Picking {k} clips to label ordered by {sort_by}')
        sorted_idxs = np.argsort([abs(baseline_prob - ps.predictions[label]) for ps in predictions])
        topk_clips = []
        topk_expanded = []
        for idx in sorted_idxs:
            candidate_clip = ClipInfo(predictions[idx].vid, None, predictions[idx].start_time, predictions[idx].end_time)
            overlaps_existing = False
            expanded_clip = videomanager.get_physical_clips_for_expanded_clip(candidate_clip, t)
            # If expanded_clip overlaps any existing clip, don't use it.
            for i in range(len(topk_clips)):
                if i >= len(topk_expanded):
                    topk_expanded.append(videomanager.get_physical_clips_for_expanded_clip(topk_clips[i], t))
                topk_clip_expanded = topk_expanded[i]
                overlaps_existing = any([any([clips_overlap(topk_clip, clip) for clip in expanded_clip]) for topk_clip in topk_clip_expanded])
                if overlaps_existing:
                    break
            if overlaps_existing:
                continue

            vid = predictions[idx].vid
            start_time = f"{predictions[idx].start_time:0.3f}"
            if start_time in self._shown_fids[vid]:
                continue

            self._shown_fids[vid].add(start_time)

            self.logger.debug(f'Adding clip ({candidate_clip.vid}, {candidate_clip.start_time:.2f}-{candidate_clip.end_time:.2f}) with p({label})={predictions[idx].predictions[label]:.3f}')
            # Predictions don't include vstart. Specify None, and then storage manager will look it up based on vid.
            topk_clips.append(candidate_clip)

            if len(topk_clips) == k:
                break
        return topk_clips
