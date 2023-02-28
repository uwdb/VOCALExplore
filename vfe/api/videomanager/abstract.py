from typing import Iterable, Tuple

from vfe.api.storagemanager import ClipInfo, ClipInfoWithPath, VidType

class AbstractVideoManager:
    def get_clip_splits(self, vids, clip_duration) -> Iterable[ClipInfo]:
        raise NotImplementedError

    def get_physical_clips_for_expanded_clip(self, clip_info: ClipInfo, total_duration) -> Iterable[ClipInfoWithPath]:
        raise NotImplementedError

    def get_video_paths(self, vids) -> Iterable[Tuple[VidType, str]]:
        raise NotImplementedError

    def get_all_vids(self) -> Iterable[VidType]:
        raise NotImplementedError
