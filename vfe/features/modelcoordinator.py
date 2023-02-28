import logging
import numpy as np
import os
import time
import torch
from typing import List, Dict

from nvidia.dali.plugin import pytorch as dali_pytorch

from vfe.api.storagemanager import AbstractStorageManager
from vfe import datasets
from vfe.features.pretrained.models import PretrainedModelExtractor
from vfe.featurestore.parquet import ParquetFeatureStore as FeatureStore

class ModelFeatureExtractorCoordinator:
    def __init__(self,
            *,
            models: List[PretrainedModelExtractor] = None,
            transform = None, # Specify the transform with the models because they must all share the same transform.
            done_file_path = None,
            log_file_path = None,
            checkpoint_path = None,
            device = 'cuda',
            ignore_done = False):
        self._extractors = models
        self._transform = transform
        self._done_videos = set() if ignore_done else self._load_done_videos(done_file_path)
        self._done_video_path = done_file_path
        self._log_file_path = log_file_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._pool = None
        self._pool_processes = None

        self._featurestore = None
        self._storagemanager = None
        self._featurequeue = None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _load_done_videos(done_file_path):
        with open(done_file_path, 'r') as f:
            done_videos = set()
            for video in f:
                done_videos.add(int(video.strip()))
        return done_videos

    def _are_all_videos_done(self, idxs):
        for idx in idxs:
            if idx not in self._done_videos:
                return False
        return True

    def _process_batch(self, batch):
        # Process a batch at a time from the dataloader.
        # If all videos have been processed, we can skip this batch.
        # If some of the videos have been processed and some haven't,
        #   we'll just re-do some work and re-do the already done ones.
        if isinstance(batch, list):
            batch = batch[0]

        vids = list(set(v.item() for v in batch['vid'].cpu()))
        if self._are_all_videos_done(vids):
            return

        self.logger.debug(f'Processing batch with vids {vids}')

        # batch['frames']: shape [B, C, F, H, W]
        # batch['vid]: shape [B/10, 1]
        # batch['frame_secs']: [B/10, 10]
        start_coalesce_frames = time.perf_counter()
        start_coalesce_frames_p = time.process_time()
        coalesced_frames = batch['frames'].to(self._device)
        timing_info = {
            'coalesce_frames': time.perf_counter() - start_coalesce_frames,
            'coalesce_frames_p': time.process_time() - start_coalesce_frames_p,
            'batch_size_videos': len(vids),
            'batch_size_frames': coalesced_frames.shape[0],
        }
        results_info = {}
        with torch.inference_mode():
            for extractor in self._extractors:
                extractor_id = extractor._filename()

                # Preds has an entry for each layer in the model, and a row for each frame.
                start_predict = time.perf_counter()
                start_predict_p = time.process_time()
                preds = extractor.model(coalesced_frames)
                preds = extractor.coalesce(preds)
                timing_info[extractor_id + '_predict'] = time.perf_counter() - start_predict
                timing_info[extractor_id + '_predict' + '_p'] = time.process_time() - start_predict_p

                # Coalesced is a dictionary with an entry for each layer in the model.
                # Each layer is a list with one entry per video.
                start_coalesce = time.perf_counter()
                start_coalesce_p = time.process_time()
                results_info[extractor_id] = preds
                timing_info[extractor_id + '_coalesce'] = time.perf_counter() - start_coalesce
                timing_info[extractor_id + '_coalesce' + '_p'] = time.process_time() - start_coalesce_p

        if len(batch['vid']) < len(batch['frames']):
            results_info['vids'] = torch.repeat_interleave(batch['vid'], batch['size'])
        else:
            results_info['vids'] = batch['vid'].flatten().cpu()

        if 'frame_timestamp' in batch:
            # batch['frame_timestamp'].shape = [B x F]. Take the first and last of each clip of F frames.
            if 'num_frames_in_window' in batch and torch.all(batch['num_frames_in_window'] > 0).item():
                # Duration is last frame start + (duration / # frames)
                results_info['frame_starts'] = batch['frame_timestamp'][:, -1].flatten().cpu()
                results_info['frame_ends'] = results_info['frame_starts'] + (batch['frame_timestamp'][:, -1] - batch['frame_timestamp'][:, 0]).flatten().cpu() / batch['num_frames_in_window'].flatten().cpu()
            else:
                results_info['frame_starts'] = batch['frame_timestamp'][:, 0].flatten().cpu()
                results_info['frame_ends'] = batch['frame_timestamp'][:, -1].flatten().cpu()
        else:
            results_info['frame_starts'] = batch['frame_secs'].flatten()
            if 'frame_ends' in batch:
                results_info['frame_ends'] = batch['frame_ends'].flatten()
            else:
                results_info['frame_ends'] = results_info['frame_starts'] + torch.repeat_interleave(batch['frame_dur'], batch['size'])

        return (vids, timing_info, results_info)

    def _log_timing_info(self, timing_infos):
        if not self._log_file_path:
            return
        with open(self._log_file_path, 'a') as f:
            for timing_info in timing_infos:
                timing_strs = [f'{k},{v}' for k, v in timing_info.items()]
                print(*timing_strs, sep='\t', file=f)

    def _mark_videos_done(self, video_idxs):
        if not self._done_video_path:
            return
        with open(self._done_video_path, 'a') as f:
            print(*set(np.concatenate(video_idxs)), sep='\n', file=f)

    def _log_checkpoint(self, n, checkpoint_time, checkpoint_time_p):
        if not self._checkpoint_path:
            return
        with open(self._checkpoint_path, 'a') as f:
            print(n, checkpoint_time, checkpoint_time_p, sep='\t', file=f)

    def _process_checkpoint(self, video_idxs, timing_infos, results_infos):
        start_checkpoint = time.perf_counter()
        start_checkpoint_p = time.process_time()
        combined_idxs = set(np.concatenate(video_idxs))
        vids = np.concatenate([ri['vids'] for ri in results_infos]).astype(np.uint32)
        starts = np.concatenate([ri['frame_starts'] for ri in results_infos]).astype(np.float64)
        ends = np.concatenate([ri['frame_ends'] for ri in results_infos]).astype(np.float64)
        for extractor in self._extractors:
            extractor_features = [results[extractor._filename()] for results in results_infos]
            for layer in extractor.layers:
                layer_features = torch.vstack([f[layer].flatten(start_dim=1) for f in extractor_features]).to(torch.float32).cpu()
                if self._featurestore is not None:
                    self._featurestore.insert_batch(extractor.filename(layer), vids, starts, ends, list(layer_features.numpy()))
                elif self._storagemanager is not None:
                    self._storagemanager.add_feature_batch(extractor.filename(layer), vids, starts, ends, list(layer_features.numpy()))
                elif self._featurequeue is not None:
                    self._featurequeue.put((extractor.filename(layer), vids, starts, ends, list(layer_features.numpy())))
        self._log_timing_info(timing_infos)
        self._mark_videos_done(video_idxs)
        checkpoint_time = time.perf_counter() - start_checkpoint
        checkpoint_time_p = time.process_time() - start_checkpoint_p
        self._log_checkpoint(len(combined_idxs), checkpoint_time, checkpoint_time_p)

    def measure_extraction(self, video_iterator, checkpoint):
        video_idxs, timing_infos, results_infos = [], [], []
        total_size = 0

        start_wait_for_batch = time.perf_counter()
        start_wait_for_batch_p = time.process_time()
        for batch in video_iterator:
            batch_time_info = {}
            batch_time_info['wait_for_batch'] = time.perf_counter() - start_wait_for_batch
            batch_time_info['wait_for_time_p'] = time.process_time() - start_wait_for_batch_p

            video_info = self._process_batch(batch)
            if video_info is None:
                start_wait_for_batch = time.perf_counter()
                start_wait_for_batch_p = time.process_time()
                continue

            (video_idx_list, timing_info, results_info) = video_info
            # Add information about how long it took to load the batch to the timing info.
            timing_info = { **timing_info, **batch_time_info }

            video_idxs.append(video_idx_list)
            timing_infos.append(timing_info)
            results_infos.append(results_info)
            total_size += len(video_idx_list)
            if total_size >= checkpoint:
                self._process_checkpoint(video_idxs, timing_infos, results_infos)
                video_idxs, timing_infos, results_infos = [], [], []
                total_size = 0

            start_wait_for_batch = time.perf_counter()
            start_wait_for_batch_p = time.process_time()

        if len(video_idxs):
            self._process_checkpoint(video_idxs, timing_infos, results_infos)

    @staticmethod
    def _check_dl_kwargs(dl_kwargs):
        assert 'type' in dl_kwargs
        if dl_kwargs['type'] == datasets.DatasetType.FRAME:
            assert 'stride' in dl_kwargs
        elif dl_kwargs['type'] == datasets.DatasetType.CLIP:
            assert 'clip_sampler_fn' in dl_kwargs
        else:
            logging.warn(f'_check_dl_kwargs: Unhandled type {dl_kwargs["type"]}')

    def _extract_features_base(self, *,
        featurestore,
        get_dataset_fn,
        get_dataset_kwargs,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500,
    ):
        self._check_dl_kwargs(dl_kwargs)

        # This seems weird?
        self._featurestore = featurestore

        video_iterator = iter(torch.utils.data.DataLoader(
            get_dataset_fn(
                **get_dataset_kwargs,
                done_idxs=self._done_videos,
                transform=self._transform,
                **dl_kwargs
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=datasets.frame.videoFrameDatasetCollateFn,
        ))

        self.measure_extraction(video_iterator, checkpoint)

    def extract_features(self, *,
        dataset: datasets.VFEDataset = None,
        split: str = None,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500,
    ):
        featurestore = FeatureStore.create(
                base_dir=self._extractors[0].features_dir,
                dataset=dataset.name(),
            )

        self._extract_features_base(
            featurestore=featurestore,
            get_dataset_fn=dataset.get_dataset,
            get_dataset_kwargs={},
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint=checkpoint
        )

    def extract_features_dali(self, *,
        dataset: datasets.VFEDataset = None,
        split: str = None,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500,
    ):
        self._check_dl_kwargs(dl_kwargs)

        self._featurestore = FeatureStore.create(
            base_dir=self._extractors[0].features_dir,
            dataset=dataset.name()
        )
        pipeline = dataset.get_dataset(
                mp4=True,
                batch_size=batch_size,
                num_threads=num_workers,
                **dl_kwargs
            )
        if pipeline is None:
            return
        video_iterator = dali_pytorch.DALIGenericIterator(
            pipeline,
            ['frames', 'vid', 'frame_timestamp', 'num_frames_in_window'],
            last_batch_policy=dali_pytorch.LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader',
        )

        self.measure_extraction(video_iterator, checkpoint)

    def extract_features_from_vids(self, *,
        featurestore,
        dbcon,
        vids,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500,
    ):
        self._extract_features_base(
            featurestore=featurestore,
            get_dataset_fn=datasets.VFEDBDataset.get_dataset_from_vids,
            get_dataset_kwargs=dict(vids=vids, dbcon=dbcon),
            dl_kwargs=dl_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint=checkpoint
        )

    def extract_features_from_storagemanager_dali(self, *,
        storagemanager: AbstractStorageManager = None,
        vids=None,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500
    ):
        self._check_dl_kwargs(dl_kwargs)
        self._storagemanager = storagemanager

        pipeline = datasets.VFEDBDataset.get_dataset_from_storagemanager(
                vids=vids,
                storagemanager=self._storagemanager,
                mp4=True,
                batch_size=batch_size,
                num_threads=num_workers,
                **dl_kwargs
            )
        if pipeline is None:
            return
        video_iterator = dali_pytorch.DALIGenericIterator(
            pipeline,
            ['frames', 'vid', 'frame_timestamp', 'num_frames_in_window'],
            last_batch_policy=dali_pytorch.LastBatchPolicy.PARTIAL,
            # Required or iterator loops indefinitely (https://github.com/NVIDIA/DALI/issues/2873)
            # reader_name must match name in frame::VideoFrameDaliDataloader::create_pipeline.
            reader_name='reader',
        )
        self.measure_extraction(video_iterator, checkpoint)

    def extract_features_from_vids_and_vpaths(self, *,
        vids_and_vpaths=None,
        featurequeue=None,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500
    ):
        self._check_dl_kwargs(dl_kwargs)
        self._featurequeue = featurequeue

        if dl_kwargs['type'] == datasets.DatasetType.DALI:
            pipeline = datasets.VFEDBDataset._get_dataset(
                labeled_video_paths=vids_and_vpaths,
                batch_size=batch_size,
                num_threads=num_workers,
                **dl_kwargs
            )
            if pipeline is None:
                return

            video_iterator = dali_pytorch.DALIGenericIterator(
                pipeline,
                ['frames', 'vid', 'frame_timestamp', 'num_frames_in_window'],
                last_batch_policy=dali_pytorch.LastBatchPolicy.PARTIAL,
                reader_name='reader'
            )
        else:
            pipeline = datasets.VFEDBDataset._get_dataset(
                labeled_video_paths=vids_and_vpaths,
                **dl_kwargs,
            )
            video_iterator = iter(torch.utils.data.DataLoader(
                pipeline,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                collate_fn=datasets.frame.videoFrameDatasetCollateFn,
            ))
        self.measure_extraction(video_iterator, checkpoint)

    def extract_features_from_storagemanager(self, *,
        storagemanager: AbstractStorageManager = None,
        vids = None,
        dl_kwargs: Dict = None,
        batch_size = None,
        num_workers = None,
        checkpoint = 500,
    ):
        self._check_dl_kwargs(dl_kwargs)

        # This seems weird ... it should probably be initialized in __init__.
        self._storagemanager = storagemanager

        video_iterator = iter(torch.utils.data.DataLoader(
            datasets.VFEDBDataset.get_dataset_from_storagemanager(
                vids=vids,
                storagemanager=self._storagemanager,
                mp4=True,
                done_idxs=self._done_videos,
                transform=self._transform,
                **dl_kwargs
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=datasets.frame.videoFrameDatasetCollateFn,
        ))

        self.measure_extraction(video_iterator, checkpoint)
