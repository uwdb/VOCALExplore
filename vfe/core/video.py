import cv2
import ffmpeg
import fractions
from functools import partial
import logging
import os
from pathlib import Path
import av
import re
import subprocess
import shlex

logger = logging.getLogger(__name__)

def get_video_duration_ffmpeg(video_path):
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

def get_video_duration(video_path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    return float(video_stream.duration * video_stream.time_base)

def get_thumbnail_path(video_path, thumbnail_dir) -> str:
    return str(Path(thumbnail_dir) / Path(video_path).with_suffix('.jpg').name)

def save_thumbnail(video_path, thumbnail_dir) -> str:
    thumbnail_path = get_thumbnail_path(video_path, thumbnail_dir)
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

def add_silent_audio(video_path) -> None:
    # Check if video already has audio.
    container = av.open(video_path)
    has_audio = container.streams.audio
    if has_audio:
        return

    base, ext = os.path.splitext(video_path)
    tmp_video = base + ".tmp" + ext
    result = subprocess.run(shlex.split(
        (
            'ffmpeg -hide_banner '
            '-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 '
            f'-i {video_path} '
            '-c:v copy -c:a aac -shortest '
            f'{tmp_video}'
        )),
    )
    if result.returncode != 0:
        logger.error(f'Error adding audio track: {result}')
    else:
        os.rename(tmp_video, video_path)


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

def transcode_video(path, output_dir, base_dir):
    base, tail = os.path.split(path)
    name, _ = os.path.splitext(tail)
    output_video_dir = Path(output_dir) / Path(base).relative_to(Path(base_dir))
    output_path = os.path.join(output_video_dir, f'{name}.mp4')
    if os.path.exists(output_path):
        return

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    try:
        stream = ffmpeg.input(path)
        (ffmpeg
            .output(
                stream.video,
                stream.audio,
                output_path,
                vcodec='h264_nvenc',
                acodec='aac')
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        print(f'Exception {e.__class__} occurred: {e.stderr}')
        return
    except Exception as e:
        print(f'Exception: {e}')
        return
