import clip
from collections import defaultdict
from enum import Enum
from functools import partial
from fractions import Fraction
import re
import torch
import torchaudio
# import torch_scatter
import torch.multiprocessing as mp
import torchaudio.transforms as AT
import torchvision.models as models
from torchvision.models.video import R3D_18_Weights, MViT_V2_S_Weights, MViT_V1_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T
import pytorchvideo.transforms as PVT
import pytorchvideo.data as PVD

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from vfe.features.abstract import *
from . import transforms

try:
    import openl3
    import resampy
except:
    pass


class PretrainedModel(Enum):
    RESNET34 = 'resnet18'
    INCEPTIONV3 = 'inception_v3'
    RESNET3D = 'r3d_18'
    I3D = 'i3d_r50'


class PretrainedModelExtractor(AbstractSegmentProcessor):
    def __init__(self, *, layers=None, device='cuda', extractor_name=None, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers if layers is not None else list(self.layer_info().keys())
        for layer in self.layers:
            if layer not in self.layer_info():
                raise RuntimeError(f'Unrecognized layer: {layer}')
        self.device = device
        # extractor_name is used so that when fps/fstride is included in the feature name, it becomes
        # part of the feature's storage path. Ideally we'd share storage for anything with the same base filename,
        # and just do sampling on top of the features that exist to simulate different fps/fstride.
        self.extractor_name = extractor_name
        self._model = None
        self._ctx = None
        self._opened_arrays = {}

    def initialize_array(self, dataset_name, split):
        for layer in self.layers:
            self._opened_arrays[layer] = get_npy_array(self.features_dir, self.filename(layer), dataset_name, split)

    def ensure_array_exists(self, dataset: datasets.VFEDataset, split):
        for layer in self.layers:
            arr_path = npy_arr_path(self.features_dir, self.filename(layer), dataset.name(), split)
            if not os.path.exists(arr_path):
                self._reset_array(dataset, split, layer)

    def reset_array(self, dataset: datasets.VFEDataset, split):
        for layer in self.layers:
            self._reset_array(dataset, split, layer)

    def _reset_array(self, dataset: datasets.VFEDataset, split, layer):
        n_records = len(dataset.get_labels(split))
        create_npy_array(self.features_dir, self.filename(layer), dataset.name(), split, shape=(n_records, self.featuredim(layer)))

    def add_results(self, idxs, results):
        # order = np.argsort(idxs)
        # idxs = idxs[order]
        for layer in self.layers:
            arr = self._opened_arrays[layer]
            unordered_results = torch.cat([torch.vstack(result[layer]) for result in results])
            arr[idxs] = unordered_results.cpu()
            arr.flush()

    @staticmethod
    def layer_info() -> dict:
        # Return a dictionary of layer name -> layer dimension
        raise NotImplementedError

    def _filename(self):
        raise NotImplementedError

    def filename(self, layer):
        if self.extractor_name is None:
            return self._filename().lower() + '_' + layer.lower()
        elif layer.lower() not in self.extractor_name:
            return self.extractor_name + '_' + layer.lower()
        else:
            return self.extractor_name

    def featuredim(self, layer):
        return self.layer_info()[layer]

    @classmethod
    def transform(cls):
        raise NotImplementedError

    @classmethod
    def base_model(cls):
        raise NotImplementedError

    def process(self, video_path):
        raise NotImplementedError("process() shouldn't be used for a pretrained model extractor")

    @staticmethod
    def postprocess_layer_op(layer):
        return torch.nn.Identity()

    # def coalesce_fn(self, clip_predictions):
    #     return torch.max(clip_predictions, axis=0).values.flatten()

    # def coalesce(self, rows_per_video, predictions):
    #     start_idx = 0
    #     coalesced = defaultdict(list)
    #     for count in rows_per_video:
    #         end_idx = start_idx + count
    #         for layer in self.layers:
    #             coalesced[layer].append(
    #                 # Post-process the outputs of each layer to reduce their dimensionality, if necessary.
    #                 # Then take the element-wise combination to get a single feature vector per video.
    #                 self.coalesce_fn(self.postprocess_layer_op(layer)(predictions[layer][start_idx:end_idx]))
    #             )
    #         start_idx = end_idx
    #     assert end_idx == len(predictions[self.layers[0]]), 'Unaccounted for predictions'
    #     return coalesced

    def coalesce(self, clip_predictions):
        return clip_predictions

    # def coalesce_vectorized(self, rows_per_video, predictions, device):
    #     coalesced = defaultdict(list)
    #     idx_vector = torch.repeat_interleave(torch.arange(len(rows_per_video)), rows_per_video)
    #     for layer in self.layers:
    #         # Expand idx_vector so it has the same shape as predictions[layer].
    #         # Currently idx_vector is 1-D. Stack the indexes vertically then repeat them across all columns.
    #         layer_idx_vector = torch.repeat_interleave(idx_vector.unsqueeze(1), predictions[layer].size(1), dim=1).to(device)
    #         aggregated = torch_scatter.scatter(predictions[layer], dim=0, index=layer_idx_vector, reduce='max')
    #         aggregated = self.postprocess_layer_op(layer)(aggregated)
    #         coalesced[layer] = aggregated.flatten(start_dim=1).cpu().tolist()
    #     return coalesced

    # Shouldn't be used.
    @classmethod
    def _process_segment_features(cls, wrapped_batch):
        # Process a batch at a time from the dataloader.
        # batch['frames'] is a 4D tensor (nvideos, fps, c, h, w)
        # Coalesce so that frames from all videos are in a single, large tensor of shape (nvideos * fps, c, h, w).
        batch = wrapped_batch['value']
        coalesced_frames = batch['frames'].view((-1, *batch['frames'].shape[2:])).to(cls._processor.device)
        with torch.no_grad():
            preds = cls._processor.model(coalesced_frames)

        # Take max over each video in the batch.
        predictions_per_video = cls._processor._coalesce(batch['size'], preds[cls._processor.layer])

        # Insert predictions into the array.
        arr = open_zarr_array(wrapped_batch['arr_path'], wrapped_batch['sync_path'])
        for i, vid_idx in enumerate(batch['video_index']):
            arr[vid_idx.item()] = predictions_per_video[i].cpu()

    @property
    def model(self):
        if self._model is not None:
            return self._model
        base_model = self.base_model()
        return_nodes = { layer: layer for layer in self.layers}
        self._model = create_feature_extractor(base_model, return_nodes=return_nodes).to(self.device).eval()
        return self._model

    # Shouldn't be used.
    def process_dataset(self, dataset: datasets.VFEDataset, split, processes=1):
        # Create an empty zarr file in this directory.
        # Get the total number of records from the dataset, and the number of feature dimensions from this class.
        # Set the chunk size to be n_records // processes
        n_records = len(dataset.get_labels(split))
        chunk_size = math.ceil(n_records / processes) if processes > 1 else 512
        arr_path, sync_path = create_zarr_array(features_dir=self.features_dir, filename=self.filename(), dataset=dataset.name(), split=split, shape=(n_records, self.featuredim()), chunk_size=chunk_size)
        segments_iterator = iter(torch.utils.data.DataLoader(
            dataset.get_dataset(
                split,
                type=datasets.DatasetType.FRAME,
                stride=datasets.frame.CreateStride(fps=1),
                transform=self.transform()
            ),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers
        ))
        wrapped_it = WrappedIterator(segments_iterator, arr_path=arr_path, sync_path=sync_path)
        if processes > 1:
            self._wait_for_imap(self._get_pool(processes).imap_unordered(
                self._process_segment_features,
                wrapped_it
            ))
        else:
            self.__class__._processor = self
            [self._process_segment_features(wrapped_batch) for wrapped_batch in wrapped_it]

    # Shouldn't be used.
    def _get_pool(self, processes):
        if self._pool and processes == self._pool_processes:
            return self._pool
        self._ctx = mp.get_context(method='spawn')
        self._pool = self._ctx.Pool(processes, initializer=self._initialize_processor, initargs=({k:v for k, v in self.__dict__.items() if not k.startswith('_')},))
        self._pool_processes = processes
        return self._pool

class ClipModelExtractor(PretrainedModelExtractor):
    @staticmethod
    def layer_info() -> dict:
        return {'embed': 512}

    def pre_inference(self, batch):
        return batch

    @property
    def model(self):
        if self._model is not None:
            return self._model
        base_clip_model = self.base_model()
        base_clip_model.to(self.device).eval()
        def feature_extractor(inputs):
            return {'embed': base_clip_model.encode_image(
                self.pre_inference(inputs)
            )}
        self._model = feature_extractor
        return self._model

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        # If feature name contains fps, there isn't a good way to use that given dali's arguments.
        dl_kwargs = {}
        fstride = re.findall(r'(\d+)fstride', feature_name)
        fstride = int(fstride[0]) if fstride else None
        if fstride:
            dl_kwargs['step'] = fstride
            # Pick frames that are in the middle of the window.
            dl_kwargs['stride'] = fstride // 2
        else:
            raise RuntimeError(f'Dali failure: step was not specified and it cannot be automatically determined for feature {feature_name}')
        return dl_kwargs

class PooledClipModelExtractor(ClipModelExtractor):
    @classmethod
    def dali_kwargs(cls, feature_name=None):
        # If feature name contains fps, there isn't a good way to use that given dali's arguments.
        pooled_info = re.search(r'(\d+)x(\d+)maxpool', feature_name)
        sequence_length = int(pooled_info[1])
        stride = int(pooled_info[2])
        dl_kwargs = {
            'sequence_length': sequence_length,
            'stride': stride,
            'step': sequence_length * stride,
        }
        return dl_kwargs

class Resnet18Extractor(PretrainedModelExtractor):
    # {'layer2': torch.Size([2, 128, 28, 28]), 'layer4': torch.Size([2, 512, 7, 7]), 'avgpool': torch.Size([2, 512, 1, 1]), 'flatten': torch.Size([2, 512])}
    @staticmethod
    def layer_info():
        return {
            'layer2': 7381, # 100352,
            'layer4': 2295, # 25088,
            'flatten': 512,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'layer2':
            # Reduce output to  61x11x11
            return torch.nn.MaxPool3d(8, stride=2)
        elif layer == 'layer4':
            # Reduce output to 255x3x3
            return torch.nn.MaxPool3d(3, stride=2)
        elif layer == 'flatten':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        return torch.nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            transforms.imagenet_normalization(),
        )

    @classmethod
    def base_model(cls):
        return models.resnet18(pretrained=True)

    def _filename(self):
        return 'Resnet18'


class InceptionV3Extractor(PretrainedModelExtractor):
    # {'Conv2d_4a_3x3': torch.Size([2, 192, 71, 71]), 'Mixed_5d': torch.Size([2, 288, 35, 35]), 'Mixed_6e': torch.Size([2, 768, 17, 17]), 'flatten': torch.Size([2, 2048])}
    @staticmethod
    def layer_info():
       return {
           # Include earlier layers with pool to reduce size.
           'Mixed_5d': 4788, # 352800
            'Mixed_6e': 3411, # 221952
            'flatten': 2048,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'Mixed_5d':
            # Reduce output to  133x6x6
            return torch.nn.MaxPool3d(24, stride=2)
        elif layer == 'Mixed_6e':
            # Reduce output to 379x3x3
            return torch.nn.MaxPool3d(12, stride=2)
        elif layer == 'flatten':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        return torch.nn.Sequential(
            T.Resize(299),
            T.CenterCrop(299),
            transforms.imagenet_normalization(),
        )

    @classmethod
    def base_model(cls):
        return models.inception_v3(pretrained=True)

    def _filename(self):
        return 'InceptionV3'

class ClipViTB32Extractor(ClipModelExtractor):
    """
        Derived fstride32 from fstride16 as:
        select vid, start_time, end_time, feature
        from (
            select vid, start_time, end_time, feature, row_number() over (partition by vid order by start_time) as row from '*.parquet' order by vid, start_time
        )
        where row % 2 = 0;
    """

    @classmethod
    def transform(cls):
        # Copied from https://github.com/openai/CLIP/blob/main/clip/clip.py
        # Specific values from print(model.transform)
        return torch.nn.Sequential(
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        )

    @staticmethod
    def dali_transform(frames):
        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="CFHW",
            crop=(224, 224),
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            mirror=False
        )
        # Remove the first frame as it's just used to get the duration.
        # This also gets rid of the F dimension for us, so squeeze() is not necessary.
        frames = frames[:, 1, :, :]
        # frames = fn.squeeze(frames, axes=1)
        return frames

    @classmethod
    def dali_kwargs(cls, feature_name):
        return dict(
            # Get step from fps/stride in feature_name.
            **super().dali_kwargs(feature_name),
            sequence_length=2,
            resize_kwargs=dict(
                resize_shorter=224
            ),
            transform=cls.dali_transform,
            adjust_timestamp_for_lastframe=True,
        )

    @classmethod
    def base_model(cls):
        model, preprocess = clip.load('ViT-B/32')
        return model

    def _filename(self):
        return 'clip_vitb32'

class ClipViTB32PooledExtractor(ClipViTB32Extractor):
    # TODO: Clean up inheritance. It's confusing with PooledClipModelExtractor
    # not being a superclass.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Currently we set this from the batch dimensions. An alternatie would be to
        # set it from self.extractor_name.
        self.sequence_length = None


    # TODO: get rid of duplication of transform/resize with superclass.
    @classmethod
    def dali_kwargs(cls, feature_name):
        return dict(
            **PooledClipModelExtractor.dali_kwargs(feature_name),
            resize_kwargs=dict(
                resize_shorter=224,
            ),
            transform=cls.dali_transform,
        )

    @staticmethod
    def dali_transform(frames):
        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="CFHW",
            crop=(224, 224),
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            mirror=False
        )
        return frames

    def pre_inference(self, frames):
        # frames.shape: [B, C, F, H, W]
        shape = frames.shape
        self.sequence_length = shape[2]
        return frames.permute(0, 2, 1, 3, 4).reshape(shape[0] * shape[2], shape[1], shape[3], shape[4])

    def coalesce(self, clip_predictions):
        coalesced = {}
        for layer, preds in clip_predictions.items():
            shape = preds.shape
            coalesced[layer] = torch.max(
                preds.reshape(shape[0] // self.sequence_length, self.sequence_length, shape[1]),
                axis=1
            ).values
        return coalesced


class VideoPretrainedModelExtractor(PretrainedModelExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reset extractor_name so that the layer isn't part of it.
        self.extractor_name = None

    @property
    def dataset_type(self):
        return datasets.DatasetType.CLIP

    @staticmethod
    def clip_sampler_fn(video_fps):
        raise NotImplementedError


class Resnet3d18Extractor(VideoPretrainedModelExtractor):
    frames_per_clip = 8
    sampling_rate = 2
    # The paper at https://arxiv.org/pdf/1711.11248.pdf mentions they use 10 clips for evaluation.
    clips_per_video = 10

    # {'layer2': torch.Size([2, 128, 4, 56, 56]), 'layer4': torch.Size([2, 512, 1, 14, 14]), 'flatten': torch.Size([2, 512])}
    @staticmethod
    def layer_info():
        return {
            'flatten': 512,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'flatten':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        # From https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        # These steps are for slow_r50, not r3d_18.
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(lambda x: x / 255.0),
            PVT.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(256),
        ])

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        return PVD.clip_sampling.ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=cls.clips_per_video)

    @classmethod
    def base_model(cls):
        return models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

    def _filename(self):
        return 'R3d_18'

def normalize_pixels(x): # Issue when pickling as a lambda.
    return x / 255.0

class Resnet3d18AccuratePreprocessExtractor(VideoPretrainedModelExtractor):
    # Follow pre-processing notes at https://pytorch.org/vision/master/models.html#video-classification
    frames_per_clip = 16
    sampling_rate = 2
    step = (frames_per_clip * sampling_rate) // 2
    # The paper at https://arxiv.org/pdf/1711.11248.pdf mentions they use 10 clips for evaluation.
    clips_per_video = 11

    @staticmethod
    def layer_info():
        return {
            'flatten': 512,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'flatten':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):

        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(normalize_pixels),
            PVT.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989]),
            PVT.ShortSideScale(size=128),
            T.CenterCrop(112),
        ])

    @staticmethod
    def dali_transform(frames):
        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="CFHW",
            crop=(112, 112),
            mean=[0.43216 * 255, 0.394666 * 255, 0.37645 * 255],
            std=[0.22803 * 255, 0.22145 * 255, 0.216989 * 255],
            mirror=False
        )
        return frames

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=cls.step)

    @classmethod
    def base_model(cls):
        return models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

    def _filename(self):
        return 'R3d_18_ap'

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        return dict(
            sequence_length=cls.frames_per_clip,
            stride=cls.sampling_rate,
            step=cls.step,
            resize_kwargs=dict(
                resize_shorter=128
            ),
            transform=cls.dali_transform,
        )

class Resnet3d18MeanAPExtractor(Resnet3d18AccuratePreprocessExtractor):
    def _filename(self):
        return 'R3d_18_ap_mean'

    def coalesce_fn(self, clip_predictions):
        return torch.mean(clip_predictions, axis=0).flatten()

class Resnet3d18MeanAPStride32Extractor(Resnet3d18MeanAPExtractor):
    """
        Derived from Resnet3d18MeanAPExtractor as:
        select vid, start_time, end_time, feature
        from (
            select vid, start_time, end_time, feature, row_number() over (partition by vid order by start_time) as row from '*.parquet' order by vid, start_time
        )
        where row % 2 = 1;
    """
    frames_per_clip = 16
    sampling_rate = 2
    step = frames_per_clip * sampling_rate

    def _filename(self):
        return 'r3d_18_ap_mean_stride32'

class Resnet3d18SampledExtractor(VideoPretrainedModelExtractor):
    # Follow pre-processing notes at https://pytorch.org/vision/master/models.html#video-classification
    # Follow sampling procedure at https://github.com/pytorch/vision/blob/main/references/video_classification/train.py.
    frames_per_clip = 16
    sampling_fps = 15
    clips_per_video = 5

    @staticmethod
    def layer_info():
        return {
            'flatten': 512,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'flatten':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(lambda x: x / 255.0),
            PVT.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989]),
            PVT.ShortSideScale(size=128),
            T.CenterCrop(112),
        ])

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        # Number of frames to sample = frames_per_clip * video_fps / sampling_fps.
        # Duration = frames to sample / video_fps = frames_per_clip / sampling_fps.
        clip_duration = Fraction(cls.frames_per_clip, cls.sampling_fps)
        return PVD.clip_sampling.ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=cls.clips_per_video)

    @classmethod
    def base_model(cls):
        return models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

    def _filename(self):
        return 'R3d_18_sampled'


class I3DExtractor(VideoPretrainedModelExtractor):
    frames_per_clip = 8
    sampling_rate = 8
    clips_per_video = 10

    # {'blocks.5.res_blocks.2.activation': torch.Size([2, 2048, 4, 7, 7]), 'blocks.6.view':     torch.Size([2, 400])}
    @staticmethod
    def layer_info():
        return {
            'blocks.6.view': 400,
            'blocks.6.dropout': 2048,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        return torch.nn.Identity()

    @classmethod
    def transform(cls):
        # From https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(lambda x: x / 255.0),
            PVT.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
        ])

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        return PVD.clip_sampling.ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=cls.clips_per_video)

    @classmethod
    def base_model(cls):
        return torch.hub.load('facebookresearch/pytorchvideo', model='i3d_r50', pretrained=True)

    def _filename(self):
        return 'I3D'

class I3DMeanExtractor(I3DExtractor):
    def _filename(self):
        return 'I3D_mean'

    def coalesce_fn(self, clip_predictions):
        return torch.mean(clip_predictions, axis=0).flatten()

class I3DImagenetPreprocessExtractor(I3DExtractor):
    def _filename(self):
        return 'I3D_imagenet_ap'

    @classmethod
    def transform(cls):
        # From https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(lambda x: x / 255.0),
            PVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
        ])

class I3DAccuratePreprocessExtractor(VideoPretrainedModelExtractor):
    frames_per_clip = 8
    sampling_rate = 8
    clips_per_video = 10

    # {'blocks.5.res_blocks.2.activation': torch.Size([2, 2048, 4, 7, 7]), 'blocks.6.view':     torch.Size([2, 400])}  'blocks.6.pool': torch.Size([2, 2048, 1, 1, 1])
    # Use blocks.6.dropout, which has dimension 2048 and comes before the linear layer that reduces from 2048 -> 400.
    @staticmethod
    def layer_info():
        return {
            'blocks.6.view': 400,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'blocks.6.view':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        # From https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            T.Lambda(lambda x: x / 255.0),
            PVT.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989]),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
        ])

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        return PVD.clip_sampling.ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=cls.clips_per_video)

    @classmethod
    def base_model(cls):
        return torch.hub.load('facebookresearch/pytorchvideo', model='i3d_r50', pretrained=True)

    def _filename(self):
        return 'I3D_ap'

class I3DRescaleExtractor(VideoPretrainedModelExtractor):
    frames_per_clip = 8
    sampling_rate = 8
    clips_per_video = 10

    # {'blocks.5.res_blocks.2.activation': torch.Size([2, 2048, 4, 7, 7]), 'blocks.6.view':     torch.Size([2, 400])}  'blocks.6.pool': torch.Size([2, 2048, 1, 1, 1])
    @staticmethod
    def layer_info():
        return {
            'blocks.6.view': 400,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        if layer == 'blocks.6.view':
            return torch.nn.Identity()

    @classmethod
    def transform(cls):
        # From https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            # Scale between -1 and 1 as described in https://github.com/deepmind/kinetics-i3d.
            T.Lambda(lambda x: 2 * x / 255.0 - 1.0),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
        ])

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        return PVD.clip_sampling.ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=cls.clips_per_video)

    @classmethod
    def base_model(cls):
        return torch.hub.load('facebookresearch/pytorchvideo', model='i3d_r50', pretrained=True)

    def _filename(self):
        return 'I3D_rescale'

class MViTV1BExtractor(VideoPretrainedModelExtractor):
    frames_per_clip = 16
    sampling_rate = 4
    step = 60

    @classmethod
    def base_model(cls):
        return models.video.mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1)

    def _filename(self):
        return 'mvit_v1_b'

    @staticmethod
    def layer_info():
        return {
            'getitem_1': 768,
            # 'head.0': 768,
            # 'head.1': 400,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        return torch.nn.Identity()

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=cls.step)

    @classmethod
    def transform(cls):
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
            T.Lambda(normalize_pixels),
            PVT.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]),
        ])

    @staticmethod
    def dali_transform(frames):
        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="CFHW",
            crop=(224, 224),
            mean=[0.45 * 255, 0.45 * 255, 0.45 * 255],
            std=[0.225 * 255, 0.225 * 255, 0.225 * 255],
            mirror=False
        )
        return frames

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        return dict(
            sequence_length=cls.frames_per_clip,
            stride=cls.sampling_rate,
            step=cls.step,
            resize_kwargs=dict(
                resize_shorter=256
            ),
            transform=cls.dali_transform,
        )


class MViTV1B16x2Extractor(VideoPretrainedModelExtractor):
    frames_per_clip = 16
    sampling_rate = 2
    step = (frames_per_clip * sampling_rate) // 2

    @classmethod
    def base_model(cls):
        return models.video.mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1)

    def _filename(self):
        return 'mvit_v1_b_16x2'

    @staticmethod
    def layer_info():
        return {
            'getitem_1': 768,
            # 'head.0': 768,
            # 'head.1': 400,
        }

    @staticmethod
    def postprocess_layer_op(layer):
        return torch.nn.Identity()

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.frames_per_clip * cls.sampling_rate, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=cls.step)

    @classmethod
    def transform(cls):
        return T.Compose([
            PVT.UniformTemporalSubsample(cls.frames_per_clip),
            PVT.ShortSideScale(size=256),
            T.CenterCrop(224),
            T.Lambda(normalize_pixels),
            PVT.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225]),
        ])

    @staticmethod
    def dali_transform(frames):
        frames = fn.crop_mirror_normalize(
            frames,
            dtype=types.FLOAT,
            output_layout="CFHW",
            crop=(224, 224),
            mean=[0.45 * 255, 0.45 * 255, 0.45 * 255],
            std=[0.225 * 255, 0.225 * 255, 0.225 * 255],
            mirror=False
        )
        return frames

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        return dict(
            sequence_length=cls.frames_per_clip,
            stride=cls.sampling_rate,
            step=cls.step,
            resize_kwargs=dict(
                resize_shorter=256
            ),
            transform=cls.dali_transform,
        )

class MViTV1B16x2Stride32Extractor(MViTV1B16x2Extractor):
    """
        Derived from MViTV1B16x2Extractor as:
        select vid, start_time, end_time, feature
        from (
            select vid, start_time, end_time, feature, row_number() over (partition by vid order by start_time) as row from '*.parquet' order by vid, start_time
        )
        where row % 2 = 1;
    """

    frames_per_clip = 16
    sampling_rate = 2
    step = frames_per_clip * sampling_rate

    def _filename(self):
        return 'mvit_v1_b_16x2_stride32'

class MViTV1B16x2Stride32RandomExtractor(MViTV1B16x2Stride32Extractor):
    def _filename(self):
        return 'mvit_v1_b_16x2_random_stride32'

    @classmethod
    def base_model(cls):
        return models.video.mvit_v1_b(weights=None)

class Wav2Vec2Stride32Extractor(VideoPretrainedModelExtractor):
    step = 16 * 2

    @property
    def dataset_type(self):
        return datasets.DatasetType.AUDIO

    def _filename(self):
        return 'wav2vec2_base_stride32'

    @staticmethod
    def layer_info():
        return {
            'layer6': 768,
            'layer8': 768,
            'layer10': 768,
            'layer12': 768,
        }

    @property
    def model(self):
        if self._model is not None:
            return self._model
        model = self.bundle().get_model()
        model = model.to(self.device).eval()
        def feature_extractor(inputs):
            with torch.inference_mode():
                features, _ = model.extract_features(inputs)
            return {
                'layer6': torch.max(features[5], axis=1).values,
                'layer8': torch.max(features[7], axis=1).values,
                'layer10': torch.max(features[9], axis=1).values,
                'layer12': torch.max(features[11], axis=1).values,
            }
        self._model = feature_extractor
        return self._model

    @classmethod
    def bundle(cls):
        return torchaudio.pipelines.WAV2VEC2_BASE

    @staticmethod
    def resample(waveform, sample_rate, desired_sample_rate=None):
        if sample_rate == desired_sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, desired_sample_rate)

    @classmethod
    def transform(cls):
        return partial(cls.resample, desired_sample_rate=cls.bundle().sample_rate)

    @staticmethod
    def dali_transform(frames):
        return None

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        # Need sequence_length, stride, step for aligning features.
        return dict(
            sequence_length=cls.step,
            stride=1,
            step=cls.step,
        )

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.step, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=clip_duration)

class AudioLargeStride32Extractor(Wav2Vec2Stride32Extractor):
    @staticmethod
    def layer_info():
        return {
            'layer24': 1024,
        }

    # TODO: share across audio feats.
    @property
    def model(self):
        if self._model is not None:
            return self._model
        model = self.bundle().get_model()
        model = model.to(self.device).eval()
        def feature_extractor(inputs):
            with torch.inference_mode():
                features, _ = model.extract_features(inputs)
            return {
                'layer24': torch.max(features[23], axis=1).values
            }
        self._model = feature_extractor
        return self._model

class HubertLargeStride32Extractor(AudioLargeStride32Extractor):
    def _filename(self):
        return 'hubert_large_stride32'

    @classmethod
    def bundle(cls):
        return torchaudio.pipelines.HUBERT_LARGE

class Wav2Vec2LargeStride32Extractor(AudioLargeStride32Extractor):
    def _filename(self):
        return 'wav2vec2_large_stride32'

    @classmethod
    def bundle(cls):
        return torchaudio.pipelines.WAV2VEC2_LARGE


class AudioClipStride32Extractor(VideoPretrainedModelExtractor):
    step = 16 * 2

    @property
    def dataset_type(self):
        return datasets.DatasetType.AUDIO

    def _filename(self):
        return 'audio_clip_stride32'

    @staticmethod
    def layer_info():
        return {
            'embed': 512,
        }

    @property
    def model(self):
        if self._model is not None:
            return self._model
        base_clip_model = self.base_model()
        base_clip_model.to(self.device).eval()
        def feature_extractor(inputs):
            return {'embed': base_clip_model.encode_image(
                inputs
            )}
        self._model = feature_extractor
        return self._model

    @classmethod
    def base_model(cls):
        model, preprocess = clip.load('ViT-B/32')
        return model

    @classmethod
    def preprocess(cls):
        model, preprocess = clip.load('ViT-B/32')
        return preprocess

    @classmethod
    def transform(cls):
        preprocess = cls.preprocess()
        # Takes as input waveform and sample rate.
        def to_spectrogram(waveform, sample_rate):
            # Ref: https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#melspectrogram
            n_fft = 1024
            win_length = None
            hop_length = 512
            n_mels = 128

            mel_spectogram = AT.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode='reflect',
                power=2.0,
                norm='slaney',
                n_mels=n_mels,
                mel_scale='htk'
            )
            spectrogram = AT.AmplitudeToDB()(mel_spectogram(waveform))
            # Convert from grayscale to RGB.
            # The resulting values won't be between 0-1, but see what happens.
            spectrogram_img = spectrogram.expand(3, -1, -1)
            return preprocess(T.ToPILImage()(spectrogram_img))

        return to_spectrogram

   @staticmethod
    def dali_transform(frames):
        return None

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        # Need sequence_length, stride, step for aligning features.
        return dict(
            sequence_length=cls.step,
            stride=1,
            step=cls.step,
        )

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.step, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=clip_duration)

class OpenL3Stride32Extractor(VideoPretrainedModelExtractor):
    step = 16 * 2

    @property
    def dataset_type(self):
        return datasets.DatasetType.AUDIO

    def _filename(self):
        return 'openl3_music_stride32'

    @staticmethod
    def layer_info():
        return {
            "emb": 512,
        }

    @property
    def model(self):
        if self._model is not None:
            return self._model
        model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
        def feature_extractor(inputs):
            inputs_list = [t.cpu().numpy() for t in inputs]
            sr_list = [openl3.core.TARGET_SR for _ in inputs]
            emb, ts = openl3.get_audio_embedding(inputs_list, sr_list, model=model)
            return {
                "emb": torch.stack([
                    torch.from_numpy(np.mean(e, axis=0))
                    for e in emb
                ]),
            }
        self._model = feature_extractor
        return self._model

    @staticmethod
    def resample(waveform, sample_rate):
        if sample_rate == openl3.core.TARGET_SR:
            return torch.from_numpy(waveform)
        # ref: https://openl3.readthedocs.io/en/latest/_modules/openl3/core.html#preprocess_audio
        # _preprocess_audio_batch
        audio = resampy.resample(waveform, sr_orig=sample_rate, sr_new=openl3.core.TARGET_SR, filter='kaiser_best')
        return torch.from_numpy(audio)

    @classmethod
    def transform(cls):
        return cls.resample

    @staticmethod
    def dali_transform(frames):
        return None

    @classmethod
    def dali_kwargs(cls, feature_name=None):
        # Need sequence_length, stride, step for aligning features.
        return dict(
            sequence_length=cls.step,
            stride=1,
            step=cls.step,
        )

    @classmethod
    def clip_sampler_fn(cls, video_fps):
        clip_duration = Fraction(cls.step, video_fps)
        # Return uniform clip sampler so we can accurately compute clip boundaries.
        return PVD.clip_sampling.UniformClipSampler(clip_duration=clip_duration, stride=clip_duration)