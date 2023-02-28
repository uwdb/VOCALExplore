from typing import Iterable, Tuple, List, Union

from vfe.api.storagemanager.abstract import FeatureSet, VidType, ClipSet

class AbstractFeatureManager:
    def add_video(self, path, start_time=None, duration=None) -> VidType:
        raise NotImplementedError

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        raise NotImplementedError

    def get_features(self, feature_names: Union[str, List[str]], vids) -> FeatureSet:
        raise NotImplementedError

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet, only_already_extracted=False) -> FeatureSet:
        # only_already_extracted: don't proactively extract any missing features; return immediately with the subset
        # that is already stored.
        raise NotImplementedError

    def get_extracted_features_info(self, feature_names: Union[str, List[str]]) -> Tuple[Iterable[VidType], Iterable[VidType]]:
        # Return (vids _with_ features extracted, vids _without_ features extracted)
        raise NotImplementedError

    def pause(self):
        pass

    def resume(self):
        pass

class AbstractAsyncFeatureManager(AbstractFeatureManager):
    def extract_features_async(self, feature_names: Union[str, List[str]], vids, callback=None) -> None:
        # Callback may not be called from the main thread.
        raise NotImplementedError
