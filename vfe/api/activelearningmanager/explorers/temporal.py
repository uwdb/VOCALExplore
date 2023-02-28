import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import logging
from typing import Iterable, List

from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager import AbstractFeatureManager
from vfe.api.modelmanager import AbstractModelManager
from vfe.api.videomanager import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer

class TemporalExplorer(AbstractExplorer):
    def __init__(self, rng=None):
        self.logger = logging.getLogger(__name__)

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None) -> Iterable[ClipInfo]:
        self.logger.debug('Temporal explore')

        clip_duration = t
        if vids is None:
            vids = set(videomanager.get_all_vids()) - set(modelmanager.get_vids_with_labels())
        clips: Iterable[ClipInfo] = list(videomanager.get_clip_splits(vids, clip_duration=clip_duration))
        # Don't use clipinfo_to_clipset because we don't want to lose vstart.
        clipsset = pa.Table.from_pylist([c._asdict() for c in clips])
        labeled_clips = modelmanager.get_labels_for_clips(clipsset, full_overlap=False)
        # Filter to rows without labels.
        unlabeled_clips = labeled_clips.filter(pc.equal(labeled_clips['labels'], 'none'))
        # Sort in order of vstart.
        ordered_clips = duckdb.connect().execute("""
            SELECT c.vid, c.vstart, c.start_time, c.end_time
            FROM clipsset c, unlabeled_clips ul
            WHERE c.vid=ul.vid AND c.start_time=ul.start_time AND c.end_time=ul.end_time
            ORDER BY c.vstart, c.start_time
            LIMIT ?
        """, [k]).arrow()
        clips = []
        for row in ordered_clips.to_pylist():
            vid = row['vid']
            start_time = row['start_time']
            end_time = row['end_time']
            self.logger.debug(f'Adding clip ({vid}, {start_time:.2f}-{end_time:.2f})')
            clips.append(ClipInfo(vid=vid, vstart=None, start_time=start_time, end_time=end_time))
        return clips
