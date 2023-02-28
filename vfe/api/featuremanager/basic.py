import logging
import numpy as np
import os
import re
import torch
from typing import Iterable, Tuple, Union, List

from vfe import core
from vfe import datasets
from vfe import features
from vfe.features.modelcoordinator import ModelFeatureExtractorCoordinator

from vfe.api.storagemanager import AbstractStorageManager, ClipSet, VidType
from .abstract import AbstractFeatureManager, FeatureSet

class BasicFeatureManager(AbstractFeatureManager):
    def __init__(self, storagemanager: AbstractStorageManager, tmp_dir=None, num_workers=1, batch_size=1, device=None, checkpoint=500, dali_preprocess=False):
        self.logger = logging.getLogger(__name__)
        self.storagemanager = storagemanager
        self.tmp_dir = tmp_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.device = device if device is not None else \
            'cuda' if torch.cuda.is_available() else 'cpu'
        self.dali_preprocess = dali_preprocess
        if self.dali_preprocess and self.device == 'cpu':
            self.logger.warn('Overriding dali_preprocess to False because no gpu is available')
            self.dali_preprocess = False

    def add_video(self, path, start_time=None, duration=None) -> VidType:
        # Don't proactively extract features.
        if duration is None:
            duration = core.video.get_video_duration(path)
        try:
            return self.storagemanager.add_video(path, start_time, duration)
        except Exception as e:
            self.logger.warn(f'Failed to add video at path {path} with exception {e}')
            return None

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        return self.storagemanager.add_videos(video_csv_path)

    def _extract_features_dali(self, feature_name, vids):
        assert self.device == 'cuda', f'Dali requires gpu'
        ignore_done = True # Don't trust the done file to filter out done vids.
        batch_size = self.batch_size
        num_workers = self.num_workers
        checkpoint = self.checkpoint

        extractor = features.utils.get_extractor(feature_name, features_dir=None, device=self.device)
        dl_kwargs = {
            'type': datasets.DatasetType.DALI,
            'device': 'gpu',
            **extractor.dali_kwargs(feature_name),
        }

        self.logger.debug(f'Extracting features with {extractor._filename()} using {num_workers} workers on device {self.device}')
        coordinator = ModelFeatureExtractorCoordinator(
            models=[extractor],
            ignore_done=ignore_done,
        )
        coordinator.extract_features_from_storagemanager_dali(
            storagemanager=self.storagemanager,
            vids=vids,
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint=checkpoint,
        )

    def _extract_features(self, feature_name, vids):
        if self.dali_preprocess:
            return self._extract_features_dali(feature_name, vids)

        features_dir = None # Not used.
        progress_dir = self.tmp_dir
        progress_suffix = f'{feature_name}_{min(vids)}_{max(vids)}_{len(vids)}'

        fps = re.findall(r'(\d+)fps', feature_name)
        fps = int(fps[0]) if fps else None
        fstride = re.findall(r'(\d+)fstride', feature_name)
        fstride = int(fstride[0]) if fstride else None
        # feature_extractor_name = re.findall(r'(.*?)(_\d+fps|_\d+fstride|$)', feature_name)[0][0]

        ignore_done = True # Don't trust the done file to filter out done vids.
        batch_size = self.batch_size
        num_workers = self.num_workers
        checkpoint = self.checkpoint
        device = self.device # Need to check why both extractor and coordinator need device param.

        extractor = features.utils.get_extractor(feature_name, features_dir=features_dir, device=device)
        self.logger.debug(f'Extracting features with {extractor._filename()} using {num_workers} workers on device {device}')
        transform = extractor.transform()
        dataset_type = features.utils.get_extractor_type(extractor)
        dl_kwargs = {
            'type': dataset_type
        }
        if dataset_type == datasets.DatasetType.FRAME:
            dl_kwargs['stride'] = datasets.frame.CreateStride(fps=fps, fstride=fstride)
        elif dataset_type == datasets.DatasetType.CLIP:
            dl_kwargs['clip_sampler_fn'] = extractor.clip_sampler_fn

        if progress_dir:
            done_path = os.path.join(progress_dir, f'done-{progress_suffix}.txt') # This file is ignored.
            time_path = os.path.join(progress_dir, f'time-{progress_suffix}.txt')
            checkpoint_time_path = os.path.join(progress_dir, f'checkpoint-time-{progress_suffix}.txt')
            for path in [done_path, time_path, checkpoint_time_path]:
                core.filesystem.ensure_exists(path)
        else:
            done_path, time_path, checkpoint_time_path = None, None, None

        coordinator = ModelFeatureExtractorCoordinator(
            models=[extractor],
            transform=transform,
            done_file_path=done_path,
            log_file_path=time_path,
            checkpoint_path=checkpoint_time_path,
            ignore_done=ignore_done,
            device=device
        )
        coordinator.extract_features_from_storagemanager(
            storagemanager=self.storagemanager,
            vids=vids,
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint=checkpoint
        )

    def get_features(self, feature_names: Union[str, List[str]], vids, priority=None) -> FeatureSet:
        # Ignore priority.

        feature_names = core.typecheck.ensure_list(feature_names)

        if vids is None:
            self.logger.debug(f'get_features: vids is None; returning all stored features')
            return self.storagemanager.get_features(feature_names=feature_names, vids=None)

        if not isinstance(vids, np.ndarray):
            vids = np.array(vids)

        self.logger.debug(f'Requested features for {len(vids)}')

        # Assumes that all of the features for a given vid have been materialized, or none of them have.
        # Also assumes that for a given model, either all layers have been materialized or none of them have.
        # Figure out what vids have already been materialized.
        for feature_name in feature_names:
            vids_with_saved_features = self.storagemanager.get_stored_feature_vids(feature_name)

            # Compute the difference between saved_features['vid'] and vids.
            missing_vids = set(vids) - set(vids_with_saved_features)

            # Extract the features for the missing vids.
            if len(missing_vids):
                self.logger.debug(f'Extracting features for {len(missing_vids)} vids: {missing_vids}')
                self._extract_features(feature_name, missing_vids)

        # Return the complete feature set.
        # At this point, we know that the features for all vids have been extracted and therefore will be in the feature store.
        return self.storagemanager.get_features(feature_names=feature_names, vids=vids)

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet, only_already_extracted=False) -> FeatureSet:
        # Ignore only_already_extracted.

        feature_names = core.typecheck.ensure_list(feature_names)

        # Assumes that all of the features for a given vid have been materialized, or none of them have.
        # Also assumes that for a given model, either all layers have been materialized or none of them have.
        # Figure out what vids have already been materialized.
        cliptable = clipset.to_table()

        for feature_name in feature_names:
            vids_with_saved_features = self.storagemanager.get_stored_feature_vids(feature_name)
            missing_vids = set(cliptable['vid'].to_pylist()) - set(vids_with_saved_features)

            if len(missing_vids):
                self.logger.debug(f'Extracting features for vids {missing_vids}')
                self._extract_features(feature_name, missing_vids)

        # Return the complete feature set.
        # At this point, we know that the features for all vids have been extracted and therefore will be in the feature store.
        return self.storagemanager.get_features_for_clips(feature_names=feature_names, clipset=clipset)

    def get_extracted_features_info(self, feature_names: Union[str, List[str]]) -> Tuple[Iterable[VidType], Iterable[VidType]]:
        # Return (vids _with_ features extracted, vids _without_ features extracted)
        vids_with_features = self.storagemanager.get_stored_feature_vids(feature_names)
        vids_without_features = set(self.storagemanager.get_all_vids()) - set(vids_with_features)
        return (vids_with_features, vids_without_features)
