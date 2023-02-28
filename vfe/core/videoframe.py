import cv2
import logging
import numpy as np

from vfe.core import consts
# import vfe.datasets

class VideoInfo:
    def __init__(self, *, frames=None, width=None, height=None, fps=None, nframes=None):
        self.frames = frames
        self.width = width
        self.height = height
        self.fps = fps
        self.nframes = nframes

    @property
    def max_frame(self):
        # There was at least one case where a frame was missing from the frame array (nframes=1198, but len=1197).
        return min(len(self.frames), self.nframes)

def extract_raw_frames(video_path, width=consts.RESIZE_WIDTH, height=consts.RESIZE_HEIGHT) -> VideoInfo:
    # Resizes frame to width/height to reduce memory consumption.
    try:
        cap = cv2.VideoCapture(video_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (width, height)))
        return VideoInfo(frames=np.stack(frames), width=width, height=height, fps=fps, nframes=nframes)
    except Exception as e:
        logging.error(f'(extract_raw_frames) Failed to read frames from video at {video_path}')

def extract_raw_frames_preallocate(video_path, width=consts.RESIZE_WIDTH, height=consts.RESIZE_HEIGHT) -> VideoInfo:
    # Resizes frame to width/height to reduce memory consumption.
    try:
        cap = cv2.VideoCapture(video_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = np.zeros((nframes, height, width, 3))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames[i] = cv2.resize(frame, (width, height))
            i += 1
        return VideoInfo(frames=frames, width=width, height=height, fps=fps, nframes=nframes)
    except Exception as e:
        logging.error(f'(extract_raw_frames) Failed to read frames from video at {video_path}')

def extract_median_frame(video_path, width=consts.RESIZE_WIDTH, height=consts.RESIZE_HEIGHT):
    # Resizes median frame to width/height to reduce memory consumption.
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (width, height)))
        return np.median(frames, axis=0).astype(dtype=np.uint8)
    except:
        logging.error(f'(extract_median_frame) Failed to read frames from video at {video_path}')


def extract_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        return frame
    except:
        logging.error(f'(extract_first_frame) Failed to extract first frame from video at {video_path}')

def extract_fps(video_path, stride):
    # Stride is vfe.datasets.frame.AbstractStride.
    class FrameIterator:
        def __init__(self, video_path, stride):
            self.cap = cv2.VideoCapture(video_path)
            self.current_frame = 0
            self.target_frame = 0
            self.video_fps = None
            self.n_frames = None
            self.stride = stride
            self._step = None # Have to set this after the video fps is known.

        def __iter__(self):
            return self

        def __next__(self):
            if not self.cap.isOpened():
                raise StopIteration
            if not self.n_frames:
                self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not self.video_fps:
                self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                self._step = self.stride.step(self.video_fps, self.n_frames)

            while self.current_frame <= self.target_frame:
                # self.current_frame is the frame number we will read inside of the loop.
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    raise StopIteration
                self.current_frame += 1
            self.target_frame += self._step
            return frame

    return FrameIterator(video_path, stride=stride)

