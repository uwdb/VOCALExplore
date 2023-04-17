import cv2
import fractions
import logging
import os
from pathlib import Path
import re
import subprocess
import shlex

logger = logging.getLogger(__name__)

def get_video_duration(video_path):
    result = subprocess.run(shlex.split(
        (
            'ffprobe -hide_banner -v error '
            '-show_entries format=duration '
            '-of default=noprint_wrappers=1:nokey=1 '
            f'{video_path}'
        )),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def save_thumbnail(video_path, thumbnail_dir) -> str:
    thumbnail_path = str(Path(thumbnail_dir) / Path(video_path).with_suffix('.jpg').name)
    if os.path.exists(thumbnail_path):
        return thumbnail_path

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        res, frame = cap.read()
        cv2.imwrite(thumbnail_path, frame)
        return thumbnail_path
    except Exception as e:
        logger.exception(f'Failed to save thumbnail for {video_path} with error {e}')
        return None

def get_video_fps_and_nframes(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps, nframes

def check_vfr(video_path):
    result = subprocess.run(shlex.split(
        (
            f'ffmpeg -hide_banner -i {video_path} '
            '-vf vfrdet -an -f null -'
        )),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    result = result.stdout.decode().strip()
    return re.findall(r'Parsed_vfrdet_0.*', result)[0]

def get_clips(video_path, sequence_length, stride, step):
    fps, nframes = get_video_fps_and_nframes(video_path)
    sequence_start = 0
    clips = []
    while True:
        sequence_stop = sequence_start + stride * (sequence_length) # -1 because of first frame. Dali needs seq_length-1 to be exact, pytorch doesn't.
        if sequence_stop >= nframes:
            break
        start_time = float(sequence_start / fps)
        stop_time = float(sequence_stop / fps)
        clips.append((start_time, stop_time))
        sequence_start = sequence_start + step
    return clips
