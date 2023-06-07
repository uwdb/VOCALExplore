import logging
from typing import Iterable, Tuple, Union, List

from vfe.core import video
from vfe.api.storagemanager import AbstractStorageManager, VidType, ClipInfo, ClipInfoWithPath, LabelInfo
from .abstract import AbstractVideoManager

logger = logging.getLogger(__name__)

class BasicVideoManager(AbstractVideoManager):
    def __init__(self, storagemanager: AbstractStorageManager):
        self.storagemanager = storagemanager

    def get_clip_splits(self, vids, clip_duration) -> Iterable[ClipInfo]:
        return self.storagemanager.get_clip_splits(vids, clip_duration)

    def get_clipinfo_with_path(self, vids: List[VidType]) -> List[ClipInfoWithPath]:
        return self.storagemanager.get_clipinfo_with_path(vids)

    def get_physical_clips_for_expanded_clip(self, clip_info: ClipInfo, total_duration) -> Iterable[ClipInfoWithPath]:
        missing_duration = total_duration - (clip_info.end_time - clip_info.start_time)
        # If missing_duration is negative, the clip is already at least as long as total_duration.
        # Don't expand its start/end in this case.
        delta = max(0, missing_duration / 2)
        start_time = clip_info.start_time - delta
        end_time = clip_info.end_time + delta
        return self.storagemanager.get_physical_clips_for_clip(ClipInfo(clip_info.vid, clip_info.vstart, start_time, end_time))

    def get_video_paths(self, vids, thumbnails=False) -> Iterable[Tuple[VidType, str, Union[str, None]]]:
        return self.storagemanager.get_video_paths(vids, thumbnails=thumbnails)

    def get_all_vids(self) -> Iterable[VidType]:
        return self.storagemanager.get_all_vids()

    def get_labels(self, vids) -> Iterable[LabelInfo]:
        return self.storagemanager.get_labels(vids)

    def get_unique_labels(self) -> Iterable[str]:
        return self.storagemanager.get_unique_labels()

    def search_videos(self, date_range=None, labels=None, predictions=None, prediction_confidence=None, mid=None):
        return self.storagemanager.search_videos(date_range=date_range, labels=labels, predictions=predictions, prediction_confidence=prediction_confidence, mid=mid)

    def reset_annotations(self):
        # Intended for debugging only.
        self.storagemanager.reset_annotations()

    def add_silent_audio(self):
        vpaths = self.get_video_paths(vids=None)
        count = 0
        for vid, vpath in vpaths:
            count += 1
            if count and count % 20 == 0:
                logger.debug(f'Added audio to {count}/{len(vpaths)}')
            video.add_silent_audio(vpath)
