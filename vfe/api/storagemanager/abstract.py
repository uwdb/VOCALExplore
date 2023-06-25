from collections import namedtuple
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from typing import Iterable, List, Tuple, Union, Dict

VidType = int
FeatureSet = ds.Scanner # columns ['fid', 'vid', 'start_time', 'end_time', 'feature'])
LabeledFeatureSet = pa.Table # columns: ['vid', 'labels', 'start_time', 'end_time', 'fid', 'feature']
ClipSet = ds.Scanner # columns ['vid', 'start_time', 'end_time']
LabeledClipSet = pa.Table # columns: ['vid', 'labels', 'start_time', 'end_time']
LabelSet = pa.Table # columns ['vid', 'start_time', 'end_time', 'label']
LabelInfo = namedtuple('LabelInfo', ['lid', 'vid', 'start_time', 'end_time', 'label']) # Eventually this will need to also handle sub-frame labels.
ModelInfo = namedtuple('ModelInfo', ['model_type', 'model_path', 'model_labels', 'feature_name', 'f1_threshold', 'mid'], defaults=(None, None,))
ClipInfo = namedtuple('ClipInfo', ['vid', 'vstart', 'start_time', 'end_time'])
ClipInfoWithPath = namedtuple('ClipInfoWithPath', [*ClipInfo._fields, 'vpath', 'thumbpath'], defaults=(None,))
AnyClipInfo = Union[ClipInfo, ClipInfoWithPath, LabelInfo] # Anything with values for vid, start_time, end_time.
PredictionRow = namedtuple('PredictionRow', ['vid', 'start_time', 'end_time', 'pred_dict'])

def clipinfo_to_clipset(clipinfos: Iterable[AnyClipInfo]) -> ClipSet:
    return ds.dataset(
        pa.Table.from_pylist([c._asdict() for c in clipinfos])
    ).scanner(
        columns=['vid', 'start_time', 'end_time']
    )

def labelset_to_labelinfo(labelset: LabelSet) -> Iterable[LabelInfo]:
    return [
        LabelInfo(**labelrow)
        for labelrow in labelset.to_pylist()
    ]

def overlaps(s1, e1, s2, e2):
    return s2 < s1 < e2 \
            or s2 < e1 < e2

def contains(s_outer, e_outer, s_inner, e_inner):
    return s_outer <= s_inner  and e_inner <= e_outer

def clips_overlap(clip1: ClipInfo, clip2: ClipInfo):
    if clip1.vid != clip2.vid:
        return False
    return overlaps(clip1.start_time, clip1.end_time, clip2.start_time, clip2.end_time) \
        or contains(clip1.start_time, clip1.end_time, clip2.start_time, clip2.end_time) \
        or contains(clip2.start_time, clip2.end_time, clip1.start_time, clip1.end_time)

class AbstractStorageManager:
    def add_video(self, path, start_time, duration) -> VidType:
        raise NotImplementedError # impl

    def add_videos(self, video_csv_path) -> Iterable[VidType]:
        # Expect video_csv_path to have a header of: path,start,duration
        raise NotImplementedError # impl

    def get_video_paths(self, vids, thumbnails=False) -> Iterable[Tuple[VidType, str, Union[str, None]]]:
        raise NotImplementedError # impl

    def add_feature_batch(self, feature_name, vids, starts, ends, feature_vectors) -> None:
        raise NotImplementedError # impl

    def add_feature(self, feature_name, vid, start, end, feature_vector) -> None:
        raise NotImplementedError # impl

    def get_feature_names(self) -> Iterable[str]:
        raise NotImplementedError

    def get_stored_feature_vids(self, feature_names: Union[str, List[str]]) -> Iterable[VidType]:
        raise NotImplementedError

    def get_features(self, feature_names: Union[str, List[str]], vids=None) -> FeatureSet:
        # If vids=None, then return all stored features.
        raise NotImplementedError

    def get_features_for_clips(self, feature_names: Union[str, List[str]], clipset: ClipSet) -> FeatureSet:
        raise NotImplementedError

    def update_label(self, vid, start, end, add_label=None, remove_label=None) -> bool:
        pass # impl

    def add_labels(self, labels: Iterable[LabelInfo]) -> bool:
        pass # impl

    def remove_label(self, vid, start, end, label) -> bool:
        pass # impl

    def add_labels_bulk(self, label_csv_path):
        # Expect label_csv_path to have a header of: path,start,end,label
        pass # impl

    def get_labels(self, vids=None, before_label_time=None, ignore_labels=[]) -> Iterable[LabelInfo]:
        raise NotImplementedError # impl

    def get_vids_with_labels(self) -> Iterable[VidType]:
        raise NotImplementedError

    def get_all_vids(self) -> Iterable[VidType]:
        raise NotImplementedError

    def get_vids_for_paths(self, paths_csv) -> Iterable[VidType]:
        # paths_csv should have a header row consisting of "vpath"
        raise NotImplementedError

    def get_distinct_labels(self) -> Iterable[str]:
        raise NotImplementedError

    def get_labels_for_features(self, featureset: FeatureSet, ignore_labels=[]) -> LabeledFeatureSet:
        raise NotImplementedError

    def get_labels_for_clips_aggregated_fulloverlap(self, clipset: ClipSet, full_overlap=True) -> LabeledClipSet:
        raise NotImplementedError

    def get_labels_for_clips_nonaggregated_overlapping(self, clipset: ClipSet) -> LabelSet:
        raise NotImplementedError

    def get_label_counts(self, feature_name) -> Dict[str, int]:
        # Count the number of feature vectors with each label.
        raise NotImplementedError

    def get_total_label_time(self) -> Dict[str, float]:
        # Get the total seconds annotated with each label.
        raise NotImplementedError

    def get_models_dir(self):
        raise NotImplementedError

    def add_model(
        self,
        model_type: str,
        feature_name: str,
        creation_time: np.datetime64,
        batch_size: int,
        epochs: int,
        learningrate: float,
        ntrain: int,
        labels: List[str],
        model_path: str,
        labels_path: str,
        f1_threshold: float,
    ) -> None:
        raise NotImplementedError

    def get_model_info(self, feature_name, ignore_labels=[], include_labels=[]) -> ModelInfo:
        raise NotImplementedError

    def get_model_info_for_mid(self, mid) -> ModelInfo:
        raise NotImplementedError

    def get_clip_splits(self, vids, clip_duration) -> Iterable[ClipInfo]:
        raise NotImplementedError

    def get_physical_clips_for_clip(self, clip_info: ClipInfo) -> Iterable[ClipInfoWithPath]:
        raise NotImplementedError
