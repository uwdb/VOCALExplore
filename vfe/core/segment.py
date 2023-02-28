import os

class Segment:
    def __init__(self, segment_name, segment_id, segment_path, capture_time, keyframe_path=None):
        self.segment_name: str = segment_name
        self.segment_id: int = segment_id
        self.segment_path: str = segment_path
        self.capture_time = capture_time
        self.keyframe_path: str = keyframe_path

    def __str__(self):
        return f'Segment {self.segment_name}, id {self.segment_id}, path {self.segment_path}, capture_time {self.capture_time}, keyframe {self.keyframe_path}'

    def scaled_video_path(self, w, h):
        root, ext = os.path.splitext(self.segment_path)
        return f'{root}_{w}x{h}.mp4'
