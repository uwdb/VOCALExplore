from typing import Iterable, Tuple

from vfe.api.storagemanager import AbstractStorageManager, VidType, ClipInfo, ClipInfoWithPath
from .abstract import AbstractVideoManager

class BasicVideoManager(AbstractVideoManager):
    def __init__(self, storagemanager: AbstractStorageManager):
        self.storagemanager = storagemanager

    def get_clip_splits(self, vids, clip_duration) -> Iterable[ClipInfo]:
        return self.storagemanager.get_clip_splits(vids, clip_duration)

    def get_physical_clips_for_expanded_clip(self, clip_info: ClipInfo, total_duration) -> Iterable[ClipInfoWithPath]:
        missing_duration = total_duration - (clip_info.end_time - clip_info.start_time)
        # If missing_duration is negative, the clip is already at least as long as total_duration.
        # Don't expand its start/end in this case.
        delta = max(0, missing_duration / 2)
        start_time = clip_info.start_time - delta
        end_time = clip_info.end_time + delta
        return self.storagemanager.get_physical_clips_for_clip(ClipInfo(clip_info.vid, clip_info.vstart, start_time, end_time))

    def get_video_paths(self, vids) -> Iterable[Tuple[VidType, str]]:
        return self.storagemanager.get_video_paths(vids)

    def get_all_vids(self) -> Iterable[VidType]:
        return self.storagemanager.get_all_vids()
