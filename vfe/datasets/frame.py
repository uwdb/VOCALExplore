import io
import math
from fractions import Fraction
import logging
import numpy as np
import os
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Type

from vfe.core import video

import av
import pytorchvideo.transforms as PVT
import torch.utils.data
import torchaudio
import torchvision.transforms as T
from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import ApplyTransformToKey
from torch.utils.data.dataloader import default_collate

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

try:
    import ffmpeg
except:
    # Ffmpeg import may fail since it's not in requirements.
    pass

try:
    # https://openl3.readthedocs.io/en/latest/tutorial.html
    import openl3
    import soundfile as sf
except:
    pass



# Base extracting multiple images from a single video on this class.

# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/data/labeled_video_dataset.html#LabeledVideoDataset


class AbstractStride:
    def step(self, video_fps, video_frames):
        raise NotImplementedError


class FPSStride(AbstractStride):
    def __init__(self, fps):
        self._fps = fps

    def step(self, video_fps, video_frames):
        # Round up in case self._fps > video_fps we don't end up with a step of 0.
        # There was a case where video_fps was 0 (< 1 fps). In that case set it to 1.
        if video_fps == 0:
            video_fps = 1
        return math.ceil(video_fps / self._fps)

class FPVStride(AbstractStride):
    def __init__(self, fpv):
        self._fpv = fpv

    def step(self, video_fps, video_frames):
        return math.ceil(video_frames / self._fpv)

class FrameStride(AbstractStride):
    def __init__(self, fstride):
        self._fstride = fstride

    def step(self, video_fps, video_frames):
        return int(self._fstride)

def CreateStride(fps=None, fpv=None, fstride=None):
    if fps is not None:
        assert fpv is None and fstride is None
        return FPSStride(fps)
    if fpv is not None:
        assert fps is None and fstride is None
        return FPVStride(fpv)
    if fstride is not None:
        assert fps is None and fpv is None
        return FrameStride(fstride)
    assert False, 'Expected one of the arguments to be non-null'

class EncodedVideoFrame:
    def __init__(self, video_path, stride: AbstractStride):
        self._stride = stride
        self._next_frame = 0
        self._video_path = video_path
        self._video_name = Path(video_path).name
        try:
            with open(video_path, 'rb') as fh:
                video_file = io.BytesIO(fh.read())
            self._container = av.open(video_file)
        except Exception as e:
            raise RuntimeError(f'Failed to open video at {video_path}. {e}')

        video_stream = self._container.streams.video[0]
        self._video_time_base = video_stream.time_base
        self._video_start_pts = video_stream.start_time
        if self._video_start_pts is None:
            self._video_start_pts = 0.0
        self._max_frame = video_stream.frames
        self._max_pts = video_stream.duration
        self._video_fps: Fraction = video_stream.average_rate
        self._step: Fraction = self._stride.step(self._video_fps, self._max_frame)

    def _frame_to_pts(self, idx):
        sec = idx / self._video_fps
        return math.floor(sec / self._video_time_base) + self._video_start_pts

    def is_complete(self):
        return self._next_frame >= self._max_frame

    def close(self):
        if self._container is not None:
            self._container.close()

    @property
    def name(self):
        return self._video_name

    @staticmethod
    def _frame_to_torch(frame):
        torch_frame = torch.from_numpy(frame.to_rgb().to_ndarray())
        # Permute tensor from (height, width, channel) to (channel, height, width).
        return torch.clamp(torch_frame.permute(2, 0, 1) / 255., min=0.0, max=1.0)

    def read_next_frame(self):
        # Compute presentation timestamp of next frame.
        frame_idx = self._next_frame
        frame_sec = frame_idx / self._video_fps
        pts = self._frame_to_pts(self._next_frame)
        if pts >= self._max_pts:
            raise RuntimeError('Trying to read past the end of the video')
        # Seek to that timestamp.
        margin = 1024
        seek_offset = max(pts - margin, 0)
        stream = {'video': 0}
        self._container.seek(int(seek_offset), any_frame=False, backward=True, stream=self._container.streams.video[0])
        after_pts_margin = 10
        frames = {}
        for frame in self._container.decode(**stream):
            frames[frame.pts] = frame
            if frame.pts > pts + after_pts_margin:
                # The pts we compute for the desired frame index won't match exactly,
                # but it will be less than the pts for the frame at index + 1.
                break
        # Sort the pts's and find the last one that is <= our desired pts.
        # If none are, return the smallest one.
        ptss = sorted(frames)
        desired_pts = ptss[0]
        for frame_pts in ptss[1:]:
            if frame_pts <= pts:
                desired_pts = frame_pts
            elif frame_pts > pts:
                break
        self._next_frame += self._step

        return frame_idx, frame_sec, self._frame_to_torch(frames[desired_pts])


def videoFrameDatasetCollateFn(batch):
    # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py, line 160
    elem = batch[0]
    collate_fn = lambda key: default_collate if key != 'frames' else torch.vstack
    return {
        key: collate_fn(key)([d[key] for d in batch])
        for key in elem
    }

class VideoFrameDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            labeled_video_paths: List[Tuple[str, Optional[dict]]],
            stride: AbstractStride,
            done_idxs: set,
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.SequentialSampler,
            transform = None,
            decoder: str = 'pyav',
            allowable_idxs: set = None,
    ) -> None:
        self._labeled_videos = labeled_video_paths
        self._stride = stride
        self._done_idxs = done_idxs
        self._decode_audio = False
        self._transform = transform
        if self._transform is not None:
            self._transform = ApplyTransformToKey(
                key="frames",
                transform=self._transform
            )
        self._decoder = decoder
        self._allowable_idxs = allowable_idxs if allowable_idxs is not None else None

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        self._video_sampler_iter = None
        self._loaded_video_label = None

    @property
    def video_sampler(self):
        return self._video_sampler

    @property
    def num_videos(self):
        return len(self._labeled_videos)

    @property
    def num_allowed_videos(self):
        return len(self._allowable_idxs)

    def __next__(self) -> dict:
        if not self._video_sampler_iter:
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        # Reuse the previously stored video if there are still frames to be sampled from it.
        if self._loaded_video_label:
            video, info_dict, video_index = self._loaded_video_label
        else:
            loaded_video = False
            while not loaded_video:
                found_new_index = False
                while not found_new_index:
                    video_index = next(self._video_sampler_iter)
                    video_path, info_dict = self._labeled_videos[video_index]
                    video_id = info_dict['vid']
                    found_new_index = video_id not in self._done_idxs and (video_id in self._allowable_idxs if self._allowable_idxs else True)
                try:
                    video = EncodedVideoFrame(video_path, stride=self._stride)
                    self._loaded_video_label = (video, info_dict, video_index)
                    loaded_video = True
                except Exception as e:
                    print(
                        f'Failed to load video at {self._labeled_videos[video_index]}'
                    )

        # Load next frame.

        frames = []
        frame_indexes = []
        frame_secs = []
        while not video.is_complete():
            frame_index, frame_sec, frame = video.read_next_frame()
            frames.append(frame)
            frame_indexes.append(frame_index)
            frame_secs.append(np.float64(frame_sec))

        # If next frame to load is past the end of the video,
        # close the loaded video and reset loaded_video_label and next_frame.
        self._loaded_video_label[0].close()
        self._loaded_video_label = None

        sample_dict = {
            'frames': torch.stack(frames),
            'size': len(frames),
            'video_index': video_index,
            'video_path': video_path,
            'frame_idxs': torch.Tensor(frame_indexes),
            'frame_secs': torch.Tensor(frame_secs),
            'frame_dur': np.float64(1 / video._video_fps),
            **info_dict,
        }

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict


    def __iter__(self):
        self._video_sampler_iter = None

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

class VideoClipDataset(torch.utils.data.IterableDataset):
    # Modify LabeledVideoDataset to return all clips from each video in a single batch.

    _MAX_CONSECUTIVE_FAILURES = int(1e5)

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler_fn, # Take as argument video fps, return a clip sampler.
        done_idxs: set,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.SequentialSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> None:
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler_fn = clip_sampler_fn
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self._done_idxs = done_idxs

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        return self._video_sampler

    @property
    def num_videos(self):
        return len(self.video_sampler)

    @staticmethod
    def _fps_from_pyav_video(video):
        return video._container.streams.video[0].average_rate

    def __next__(self) -> dict:
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                found_new_index = False
                while not found_new_index:
                    video_index = next(self._video_sampler_iter)
                    found_new_index = video_index not in self._done_idxs
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    print(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            video_is_complete = False
            all_clips = []
            clip_starts = []
            clip_ends = []
            clip_sampler = self._clip_sampler_fn(self._fps_from_pyav_video(video))
            while not video_is_complete:
                (
                    clip_start,
                    clip_end,
                    clip_index,
                    aug_index,
                    is_last_clip,
                ) = clip_sampler(
                    self._next_clip_start_time, video.duration, info_dict
                )

                if isinstance(clip_start, list):  # multi-clip in each sample

                    # Only load the clips once and reuse previously stored clips if there are multiple
                    # views for augmentations to perform on the same clips.
                    if aug_index[0] == 0:
                        self._loaded_clip = {}
                        loaded_clip_list = []
                        for i in range(len(clip_start)):
                            clip_dict = video.get_clip(clip_start[i], clip_end[i])
                            if clip_dict is None or clip_dict["video"] is None:
                                self._loaded_clip = None
                                break
                            loaded_clip_list.append(clip_dict)

                        if self._loaded_clip is not None:
                            for key in loaded_clip_list[0].keys():
                                self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

                else:  # single clip case

                    # Only load the clip once and reuse previously stored clip if there are multiple
                    # views for augmentations to perform on the same clip.
                    if aug_index == 0:
                        self._loaded_clip = video.get_clip(clip_start, clip_end)

                self._next_clip_start_time = clip_end

                video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
                )
                if (
                    is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
                ) or video_is_null:
                    # Close the loaded encoded video and reset the last sampled clip time ready
                    # to sample a new video on the next iteration.
                    self._loaded_video_label[0].close()
                    self._loaded_video_label = None
                    self._next_clip_start_time = 0.0
                    video_is_complete = True
                    if video_is_null:
                        print(
                            "Failed to load clip {}; trial {}".format(video.name, i_try)
                        )
                        continue

                vid = self._loaded_clip["video"]
                if torch.any(torch.isnan(vid)):
                    print('input video (pre-transform) has nan')
                if self._transform:
                    video_successfully_transformed = False
                    while not video_successfully_transformed:
                        # Apply the transformation before we stack the clips from the video so that the dimensions are correct: (C, T, H, W) rather than (N, C, T, H, W).
                        # This also ensures that if we are using random transformations, each clip will have a different transformation applied to it.
                        vid_tf = self._transform(vid)
                        video_successfully_transformed = not torch.any(torch.isnan(vid_tf))
                all_clips.append(vid_tf)
                clip_starts.append(clip_start)
                clip_ends.append(clip_end)

            sample_dict = {
                "frames": torch.stack(all_clips), # Shape is (N, C, T, H, W)
                'size': len(all_clips),
                "video_index": video_index,
                "video_path": video_path,
                "frame_secs": torch.Tensor(clip_starts),
                "frame_ends": torch.Tensor(clip_ends),
                **info_dict,
            }

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

class AudioClipDataset(torch.utils.data.IterableDataset):

    _MAX_CONSECUTIVE_FAILURES = int(1e5)

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler_fn, # Take as argument video fps, return a clip sampler.
        done_idxs: set,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.SequentialSampler,
        transform: Optional[Callable[[Tuple[Any, Any]], Any]] = None, # (waveform, sample_rate) -> waveform
    ) -> None:
        self._transform = transform
        self._clip_sampler_fn = clip_sampler_fn
        self._labeled_videos = labeled_video_paths
        self._done_idxs = done_idxs

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        return self._video_sampler

    @property
    def num_videos(self):
        return len(self.video_sampler)

    @staticmethod
    def _fps_from_pyav_video(video):
        return video._container.streams.video[0].average_rate

    def _copy_audio(self, video_path, output_path):
        ffmpeg.input(video_path).output(output_path, acodec='copy').overwrite_output().run(capture_stdout=True, capture_stderr=True)

    def _transcode_audio(self, video_path, output_path):
        ffmpeg.input(video_path).output(output_path, acodec='pcm_s16le').overwrite_output().run(capture_stdout=True, capture_stderr=True)

    def get_acodec(self, container):
        return container.streams.audio[0].codec.name

    def load_audio(self, pyav_video, video_path):
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            acodec = self.get_acodec(pyav_video._container)
            if acodec == 'pcm_s16le':
                self._copy_audio(video_path, f.name)
            else:
                self._transcode_audio(video_path, f.name)

            # UNCOMMENT FOR torchaudio pipelines
            # waveform, sample_rate = torchaudio.load(f.name)

            # UNCOMMENT FOR openl3 pipelines
            waveform, sample_rate = sf.read(f.name)
            if waveform.ndim == 2:
                # Also from _process_audio_batch.
                waveform = np.mean(waveform, axis=1)

        return waveform, sample_rate

    def __next__(self) -> dict:
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index, video_path = self._loaded_video_label
            else:
                found_new_index = False
                while not found_new_index:
                    video_index = next(self._video_sampler_iter)
                    found_new_index = video_index not in self._done_idxs
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decoder="pyav",
                    )
                    try:
                        waveform, sample_rate = self.load_audio(video, video_path)
                    except Exception as e:
                        # This will error if the video doesn't have an audio track.
                        logging.exception(f"Failed to load audio from {video_path}")
                        continue
                    self._loaded_video_label = (video, info_dict, video_index, video_path)
                except Exception as e:
                    print(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            video_is_complete = False
            all_clips = []
            clip_starts = []
            clip_ends = []
            min_waveform_length = None
            max_waveform_length = None
            clip_sampler = self._clip_sampler_fn(self._fps_from_pyav_video(video))
            while not video_is_complete:
                (
                    clip_start,
                    clip_end,
                    clip_index,
                    aug_index,
                    is_last_clip,
                ) = clip_sampler(
                    self._next_clip_start_time, video.duration, info_dict
                )

                assert not isinstance(clip_start, list)
                assert not isinstance(is_last_clip, list)

                # UNCOMMENT FOR torchaudio pipelines
                # clip_waveform = waveform[0, int(clip_start * sample_rate) : int(clip_end * sample_rate)]

                # UNCOMMENT FOR openl3 pipelines
                clip_waveform = waveform[int(clip_start * sample_rate) : int(clip_end * sample_rate)]

                self._next_clip_start_time = clip_end
                if is_last_clip:
                    # Close the loaded encoded video and reset the last sampled clip.
                    self._loaded_video_label[0].close()
                    self._loaded_video_label = None
                    self._next_clip_start_time = 0.0
                    video_is_complete = True

                if len(clip_waveform):
                    if self._transform:
                        clip_waveform = self._transform(clip_waveform, sample_rate)

                    # This has to come after the transform because the length might change.
                    clip_length = clip_waveform.shape[-1]
                    if min_waveform_length is None or clip_length < min_waveform_length:
                        min_waveform_length = clip_length
                    if max_waveform_length is None or clip_length > max_waveform_length:
                        max_waveform_length = clip_length

                    all_clips.append(clip_waveform)
                    clip_starts.append(clip_start)
                    clip_ends.append(clip_end)

            if min_waveform_length != max_waveform_length:
                # Make sure all clips are the same shape for stacking.
                all_clips = [clip[:min_waveform_length] for clip in all_clips]

            sample_dict = {
                'frames': torch.stack(all_clips), # Shape is (N, ?)
                'size': len(all_clips),
                'video_index': video_index,
                'video_path': video_path,
                'frame_secs': torch.Tensor(clip_starts),
                'frame_ends': torch.Tensor(clip_ends),
                **info_dict
            }
            return sample_dict
        else:
            raise RuntimeError(
                f'Failed to load audio after {self._MAX_CONSECUTIVE_FAILURES} retries'
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

def VideoFrameDaliDataloader(
    labeled_video_paths,
    transform = None,
    sequence_length=None, # frames to load per sequence
    stride=None, # distance between consecutive frames in the sequence
    step=None, # frame interval between each sequence. If < 0, set to sequence_length
    normalized=False,
    device=None,
    resize_kwargs=None, # keys in (resize_shorter, resize_longer, resize_x, resize_y)
    batch_size=None,
    num_threads=None,
    adjust_timestamp_for_lastframe=False,
):
    assert device == 'gpu', 'dali video_resize only supports gpu backend'

    files, labels = zip(*labeled_video_paths)
    files = [*files] # Tuple to list so that we can pop if necessary.
    vids = [label['vid'] for label in labels]

    pad_sequences = False
    @pipeline_def
    def create_pipeline(files, vids):
        frames, label, timestamp = fn.readers.video_resize(
            device=device,
            **resize_kwargs,
            sequence_length=sequence_length,
            stride=stride,
            step=step,
            normalized=normalized,
            random_shuffle=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            initial_fill=None, # Only relevant when shuffle=True
            file_list_include_preceding_frame=False, # Quiet warning about default changing
            pad_last_batch=False,
            pad_sequences=pad_sequences,
            dont_use_mmap=True,
            skip_vfr_check=True,
            enable_timestamps=True,
            filenames=files,
            labels=vids,
            name='reader',
        )
        frames = transform(frames)

        # transform on top of frames
        return frames, label, timestamp, stride if adjust_timestamp_for_lastframe else -1

    def handle_missing_video_stream(files, vids, missing_vpath):
        index_of_missing = files.index(missing_vpath)
        logging.warning(f'Removing video that does not contain video stream at "{missing_vpath}" (vid {vids[index_of_missing]})')
        files.pop(index_of_missing)
        vids.pop(index_of_missing)
        return files, vids

    def handle_too_short_sequences(files, vids):
        nonlocal pad_sequences
        min_frames = sequence_length * stride
        too_short_idxs = [i for i, path in enumerate(files) if video.get_video_fps_and_nframes(path)[1] < min_frames]
        if len(too_short_idxs) == len(files):
            # If all are too short, create a pipeline that pads the last batch.
            logging.warning('All vids have too few frames; pad sequence for this pipeline')
            pad_sequences = True
            return files, vids
        else:
            for idx in too_short_idxs:
                logging.warning(f'Removing video with too few frames at "{files[idx]}" (vid {vids[idx]})')
                files = [file for i, file in enumerate(files) if i not in too_short_idxs]
                vids = [vid for i, vid in enumerate(vids) if i not in too_short_idxs]
                return files, vids

    while True:
        try:
            # Add args batch_size, num_threads, device_id for @pipeline_def
            logging.debug(f'Creating dali pipeline with batch_size={batch_size}, num_threads={num_threads}')
            pipeline = create_pipeline(files, vids, batch_size=batch_size, num_threads=num_threads, device_id=0)
            pipeline.build()
            return pipeline
        except Exception as e:
            missing_vpath = re.findall(r'Could not find video stream in (.+)\n', str(e))
            if not missing_vpath:
                missing_vpath = re.findall(r'Could not open file (.+mp4) because', str(e))
            if missing_vpath:
                files, vids = handle_missing_video_stream(files, vids, missing_vpath[0])
            elif 'Assert on "!frame_starts_.empty()" failed: There are no valid sequences in the provided dataset, check the length of the available videos and the requested sequence length.' in str(e):
                files, vids = handle_too_short_sequences(files, vids)
            else:
                logging.warning('Expection when building pipeline', e)
                return None
            if not files:
                logging.warning('No more files to build a pipeline with')
                return None
