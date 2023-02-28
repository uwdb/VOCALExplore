import os
import duckdb
from enum import Enum
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

from . import utils
from . import frame


class DatasetType(Enum):
    FILE = 'file'
    FRAME = 'frame'
    CLIP = 'clip'
    DALI = 'dali'

class SplitType(Enum):
    INTERSPERSED = 'interspersed'
    STRATIFIED = 'stratified'

class FileVideoDataset(Dataset):
    def __init__(self, annotations_df, video_dir, extension='.mp4', label_col='label', path_fn=None, fullpath_fn=None, start_idx=0):
        assert path_fn is None or fullpath_fn is None, 'Only one of path_fn or fullpath_fn may be specified'
        self.labels = annotations_df
        self.video_dir = video_dir
        self.extension = extension
        self.label_col = label_col
        self.path_fn = path_fn
        self.fullpath_fn = fullpath_fn
        self.start_idx = start_idx

    def __len__(self):
        return len(self.labels) - self.start_idx

    def __getitem__(self, idx):
        idx = self.start_idx + idx
        if self.fullpath_fn:
            video_path = self.fullpath_fn(self.labels.iloc[idx])
        elif self.path_fn:
            video_path = os.path.join(self.video_dir, self.path_fn(self.labels.iloc[idx]) + self.extension)
        else:
            video_path = os.path.join(self.video_dir, self.labels.iloc[idx].filename + self.extension)
        label = self.labels.iloc[idx][self.label_col]
        return video_path, label, idx


class VFEDBDataset():
    def __init__(self, base_dir, split_dir, db_path):
        self.base_dir = base_dir
        self.split_dir = split_dir
        self.con = duckdb.connect(db_path, read_only=True)
        self.has_splits_db = self._check_has_splits_db()

    def name(self):
        raise NotImplementedError

    def _check_has_splits_db(self):
        query = """
            SELECT 'dataset_splits'
            IN (SELECT table_name FROM information_schema.tables)
        """
        return self.con.execute(query).fetchall()[0][0]

    def get_dataset(self, type: DatasetType = DatasetType.FRAME, mp4=False, **kwargs):
        # unlabeled_video_paths_from_db returns everything from video_metadata without filtering.
        if type == DatasetType.FRAME:
            labeled_video_paths = utils.unlabeled_video_paths_from_db(self.con, self.base_dir)
            return frame.VideoFrameDataset(labeled_video_paths, **kwargs)
        elif type == DatasetType.CLIP:
            labeled_video_paths = utils.unlabeled_video_paths_from_db(self.con, self.base_dir)
            return frame.VideoClipDataset(labeled_video_paths, **kwargs)
        elif type == DatasetType.DALI:
            labeled_video_paths = utils.unlabeled_video_paths_from_db(self.con, self.base_dir, mp4=mp4)
            return frame.VideoFrameDaliDataloader(labeled_video_paths, **kwargs)

    @staticmethod
    def _get_dataset(type: DatasetType, labeled_video_paths, **kwargs):
        if type == DatasetType.FRAME:
            return frame.VideoFrameDataset(labeled_video_paths, **kwargs)
        elif type == DatasetType.CLIP:
            return frame.VideoClipDataset(labeled_video_paths, **kwargs)
        elif type == DatasetType.DALI:
            return frame.VideoFrameDaliDataloader(labeled_video_paths, **kwargs)
        else:
            assert False, f'Unrecognized dataset type: {type}'

    @staticmethod
    def get_dataset_from_vids(type: DatasetType, vids, dbcon, **kwargs):
        labeled_video_paths = utils.unlabeled_video_paths_for_vids(dbcon, vids)
        return VFEDBDataset._get_dataset(type, labeled_video_paths, **kwargs)

    @staticmethod
    def get_dataset_from_storagemanager(type: DatasetType, vids, storagemanager, mp4=False, **kwargs):
        labeled_video_paths = utils.unlabeled_video_paths_for_vids_from_storagemanager(storagemanager, vids, mp4=mp4)
        return VFEDBDataset._get_dataset(type, labeled_video_paths, **kwargs)

    def get_n_splits(self, split_type: SplitType):
        split_dir = os.path.join(self.split_dir, split_type.value)
        return len(glob.glob(os.path.join(split_dir, 'split-*')))

    def get_split_vids(self, split_type: SplitType, idx):
        if self.has_splits_db:
            vs = self.con.execute("""
                SELECT split, vids
                FROM dataset_splits
                WHERE name=? AND split_type=? AND split_idx=?
            """, [self.name(), split_type.value, idx]).fetch_numpy()
            train_idx = np.where(vs['split'] == 'train')[0]
            test_idx = np.where(vs['split'] == 'test')[0]
            assert len(train_idx) == 1, f'Found {len(train_idx)} training splits'
            assert len(test_idx) == 1, f'Found {len(test_idx)} test splits'
            # vs['vids'][idx] is an array of length 1, and the first item is a list of vids.
            return np.array(vs['vids'][train_idx][0]), np.array(vs['vids'][test_idx][0])
        else:
            split_dir = os.path.join(self.split_dir, split_type.value, f'split-{idx}')
            return np.load(os.path.join(split_dir, 'train.npy')), np.load(os.path.join(split_dir, 'test.npy'))

class NW1MF9_2014_Dataset(VFEDBDataset):
    def __init__(self):
        super().__init__(
            base_dir='/gscratch/balazinska/mdaum/data/temporal-deer/videos/deer-data/season1',
            split_dir='/gscratch/balazinska/mdaum/data/temporal-deer/splits',
            db_path='/gscratch/balazinska/mdaum/data/temporal-deer/annotations.duckdb'
        )

    @classmethod
    def name(cls):
        return 'NW1MF9_2014'

class VFEDataset():
    def __init__(self, base_dir, nclasses, extension='.mp4', label_col='label', path_fn=None, fullpath_fn=None):
        assert path_fn is None or fullpath_fn is None, 'Only one of path_fn and fullpath_fn may be specified'
        self.base_dir = base_dir
        self.nclasses = nclasses
        self.extension = extension
        self.label_col = label_col
        self.path_fn = path_fn
        self.fullpath_fn = fullpath_fn
        self._annotations_df = {}

    def name(self):
        raise NotImplementedError

    @classmethod
    def label_cols(cls):
        return ['label']

    def annotations_file(self, split):
        return os.path.join(self.base_dir, split + '.csv')

    def annotations_df(self, split):
        if split not in self._annotations_df:
            self._annotations_df[split] = pd.read_csv(self.annotations_file(split))
        return self._annotations_df[split]

    def video_dir(self, split):
        return os.path.join(self.base_dir, split)

    def get_labels(self, split):
        return self.annotations_df(split)[self.label_col].values

    def get_stratify(self, split, strategy):
        return None

    def get_all_labels(self, split, idx=None):
        cols = self.annotations_df(split)[self.label_cols()]
        return cols if idx is None else cols.iloc[idx]

    def get_all_nonfilename_cols(self, split, idx=None):
        df = self.annotations_df(split)
        cols = [c for c in df.columns] # if c != 'filename']
        return df[cols] if idx is None else df[cols].iloc[idx]

    def get_video_path(self, split, idx):
        if self.fullpath_fn:
            return self.fullpath_fn(self.annotations_df(split).iloc[idx])
        elif self.path_fn:
            return os.path.join(self.video_dir(split), self.path_fn(self.annotations_df(split).iloc[idx]) + self.extension)
        else:
            return os.path.join(self.video_dir(split), self.annotations_df(split).iloc[idx].filename + self.extension)

    def add_metadata(self, split, idx, label_col, val):
        df = self.annotations_df(split)
        if label_col not in df.columns:
            df[label_col] = ''
        df.at[df.index.values[idx], label_col] = val
        # Save updated dataframe.
        df.to_csv(self.annotations_file(split), index=False)

    def labeled_video_paths(self, split):
        return utils.labeled_video_paths_from_annotations(self.annotations_df(split), self.video_dir(split), self.extension, self.nclasses, label_col=self.label_col, path_fn=self.path_fn, fullpath_fn=self.fullpath_fn)

    def get_dataset(self, split, type: DatasetType = DatasetType.FILE, **kwargs):
        if type == DatasetType.FILE:
            return FileVideoDataset(self.annotations_df(split), self.video_dir(split), self.extension, self.label_col, path_fn=self.path_fn, fullpath_fn=self.fullpath_fn, **kwargs)
        elif type == DatasetType.FRAME:
            labeled_video_paths = utils.labeled_video_paths_from_annotations(self.annotations_df(split), self.video_dir(split), self.extension, self.nclasses, label_col=self.label_col, path_fn=self.path_fn, fullpath_fn=self.fullpath_fn)
            return frame.VideoFrameDataset(labeled_video_paths, **kwargs)
        elif type == DatasetType.CLIP:
            labeled_video_paths = utils.labeled_video_paths_from_annotations(self.annotations_df(split), self.video_dir(split), self.extension, self.nclasses, label_col=self.label_col, path_fn=self.path_fn, fullpath_fn=self.fullpath_fn)
            return frame.VideoClipDataset(labeled_video_paths, **kwargs)


class DeerDataset(VFEDataset):
    def __init__(self, **kwargs):
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/deer', extension='.lvm', nclasses=3, **kwargs)

    @classmethod
    def name(cls):
        return 'DeerData'

class MoreDeerDataset(VFEDataset):
    def __init__(self, **kwargs):
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/deer-more', extension='.lvm', nclasses=4, **kwargs)

    @classmethod
    def name(cls):
        return 'MoreDeerData'

class IntraDeerDataset(VFEDataset):
    def __init__(self, **kwargs):
        fullpath_fn = lambda row: row['fullpath']
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/DeerDataIntra', extension='.lvm', nclasses=3, fullpath_fn=fullpath_fn, **kwargs)

    @classmethod
    def name(cls):
        return 'DeerDataIntra'

    def get_stratify(self, split, strategy):
        col = 'camera'
        if strategy == 'temporalsample':
            col = 'filename'
        return self.annotations_df(split)[col].values

    def get_stratify_split_fn(self, strategy):
        assert strategy == 'temporalsample'
        split_fn = lambda fn: fn.split('-')[0]
        return split_fn

class BDDDataset(VFEDataset):
    def __init__(self, label_col='scene'):
        # 6 not undefined classes for "scene"
        # 6 not undefined classes for "weather"
        assert label_col in ('scene', 'weather'), f'Unexpected label {label_col}'
        super().__init__(base_dir='/data/bdd100k/videos/100k', extension='', nclasses=7, label_col=label_col)
        self.annotations_dir = '/gscratch/balazinska/mdaum/video-features-exploration/datasets/bdd'

    @classmethod
    def name(cls):
        return 'BDD'

    @classmethod
    def label_cols(cls):
        return ['scene', 'weather']

    def annotations_file(self, split):
        return os.path.join(self.annotations_dir, f'bdd100k_labels_images_{split}.csv')


class Kinetics700Dataset(VFEDataset):
    def __init__(self, **kwargs):
        path_fn = lambda row: os.path.join(row[self.label_col], '_'.join([row['youtube_id'], f'{row["time_start"]:06d}', f'{row["time_end"]:06d}']))
        super().__init__(base_dir='/data/kinetics700/', extension='.mp4', nclasses=700, path_fn=path_fn, **kwargs)

    @classmethod
    def name(cls):
        return 'Kinetics700'

class Kinetics400Dataset(VFEDataset):
    def __init__(self, **kwargs):
        path_fn = lambda row: '_'.join([row['youtube_id'], f'{row["time_start"]:06d}', f'{row["time_end"]:06d}'])
        super().__init__(base_dir='/data/kinetics400/', extension='.mp4', nclasses=400, path_fn=path_fn, **kwargs)

    @classmethod
    def name(cls):
        return 'Kinetics400'

    def annotations_file(self, split):
        return os.path.join('/gscratch/balazinska/mdaum/data/kinetics400', split + '.csv')

class Kinetics7M4Dataset(VFEDataset):
    def __init__(self, **kwargs):
        path_fn = lambda row: os.path.join(row[self.label_col], '_'.join([row['youtube_id'], f'{row["time_start"]:06d}', f'{row["time_end"]:06d}']))
        super().__init__(base_dir='/data/kinetics700/', extension='.mp4', nclasses=100, path_fn=path_fn, **kwargs)

    @classmethod
    def name(cls):
        return 'Kinetics7M4'

    def annotations_file(self, split):
        return os.path.join('/gscratch/balazinska/mdaum/data/kinetics700-400', split + '.csv')

class BearDataset(VFEDataset):
    def __init__(self, **kwargs):
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/bears', extension='.mp4', nclasses=1, **kwargs)

    @classmethod
    def name(cls):
        return 'Bears'

    @classmethod
    def label_cols(cls):
        return ['label', 'train-activity', 'train-wire', 'train-numbears', 'train-timeofday']


class Bear5Dataset(VFEDataset):
    def __init__(self, **kwargs):
        if 'label_col' not in kwargs:
            kwargs['label_col'] = 'train-bearnobear'
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/bears-split', extension='.mp4', nclasses=2, **kwargs)

    @classmethod
    def name(cls):
        return 'Bears5'

    @classmethod
    def label_cols(cls):
        return ['train-bearnobear']

class IntraBear5Dataset(VFEDataset):
    def __init__(self, **kwargs):
        if 'label_col' not in kwargs:
            kwargs['label_col'] = 'train-bearnobear'
        fullpath_fn = lambda row: row['fullpath']
        super().__init__(base_dir='/gscratch/balazinska/mdaum/data/Bears5Intra', extension='.mp4', nclasses=2, fullpath_fn=fullpath_fn, **kwargs)

    @classmethod
    def name(cls):
        return 'Bears5Intra'

    @classmethod
    def label_cols(cls):
        return ['train-bearnobear']

    def get_stratify(self, split, strategy):
        col = 'label'
        if strategy == 'temporalsample':
            col = 'filename'
        return self.annotations_df(split)[col].values

    def get_stratify_split_fn(self, strategy):
        assert strategy == 'temporalsample'
        split_fn = lambda fn: fn.split('_')[0]
        return split_fn

def get_db_dataset(dataset_name, class_only=False, **kwargs):
    if dataset_name.lower() in ['deer1', NW1MF9_2014_Dataset.name().lower()]:
        cls = NW1MF9_2014_Dataset
    return cls if class_only else cls(**kwargs)

def get_dataset(dataset_name, class_only=False, **kwargs):
    if dataset_name.lower() in ('bdd', BDDDataset.name().lower()):
        cls = BDDDataset
    elif dataset_name.lower() in ('kinetics700', Kinetics700Dataset.name().lower()):
        cls = Kinetics700Dataset
    elif dataset_name.lower() in ('kinetics400', Kinetics400Dataset.name().lower()):
        cls = Kinetics400Dataset
    elif dataset_name.lower() in ('kinetics7m4', Kinetics7M4Dataset.name().lower()):
        cls = Kinetics7M4Dataset
    elif dataset_name.lower() in ('deer', DeerDataset.name().lower()):
        cls = DeerDataset
    elif dataset_name.lower() in ('moredeer', MoreDeerDataset.name().lower()):
        cls = MoreDeerDataset
    elif dataset_name.lower() in ('bears', BearDataset.name().lower()):
        cls = BearDataset
    elif dataset_name.lower() in ('bears5', Bear5Dataset.name().lower()):
        cls = Bear5Dataset
    elif dataset_name.lower() in ('deerintra', IntraDeerDataset.name().lower()):
        cls = IntraDeerDataset
    elif dataset_name.lower() in ('bears5intra', IntraBear5Dataset.name().lower()):
        cls = IntraBear5Dataset
    return cls if class_only else cls(**kwargs)
