import os
import pandas as pd
import pytorchvideo.data
import sklearn.preprocessing

from vfe.api.storagemanager import AbstractStorageManager

def labeled_video_paths_from_annotations(annotations_df, video_dir, extension, expected_nclasses, label_col='label', path_fn=None, fullpath_fn=None) -> pytorchvideo.data.labeled_video_paths.LabeledVideoPaths:
    assert path_fn is None or fullpath_fn is None, 'Only one of path_fn or fullpath_fn may be specified'
    # Based on factory functions in https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_paths.py
    # assert len(set(annotations_df[label_col].values)) == expected_nclasses, "Missing classes: label encoder won't be reproducible"
    annotations_df['transformed_label'] = sklearn.preprocessing.LabelEncoder().fit_transform(annotations_df[label_col].values)

    video_paths_and_label = []
    for row in annotations_df.itertuples():
        if fullpath_fn:
            filename = fullpath_fn(row._asdict())
            video_paths_and_label.append((filename, int(row.transformed_label)))
        else:
            filename = path_fn(row._asdict()) if path_fn else row.filename
            video_paths_and_label.append((filename + extension, int(row.transformed_label)))
    return pytorchvideo.data.labeled_video_paths.LabeledVideoPaths(video_paths_and_label, path_prefix=("" if fullpath_fn else video_dir))

def unlabeled_video_paths_from_db(dbcon, video_dir, mp4=False) -> pytorchvideo.data.labeled_video_paths.LabeledVideoPaths:
    results = dbcon.execute("""
        SELECT vpath, vid
        FROM video_metadata
    """).fetchall()
    update_vpath = lambda vpath: os.path.splitext(vpath)[0] + '.mp4' if mp4 else vpath
    return [
        (os.path.join(video_dir, update_vpath(vpath)), {'vid': vid, 'label': 'undefined'})
        for vpath, vid in results
    ]

def unlabeled_video_paths_for_vids(dbcon, vids):
    placeholders = ','.join(['?' for v in vids])
    results = dbcon.execute("""
        SELECT vpath, vid
        FROM video_metadata
        WHERE vid=ANY([{placeholders}])
    """.format(placeholders=placeholders), [*vids]).fetchall()
    return [
        (vpath, {'vid': vid, 'label': 'undefined'})
        for vpath, vid in results
    ]

def unlabeled_video_paths_for_vids_from_storagemanager(storagemanager: AbstractStorageManager, vids, mp4=False):
    vids_and_paths = storagemanager.get_video_paths(vids)
    update_vpath = lambda vpath: os.path.splitext(vpath)[0] + '.mp4' if mp4 else vpath
    return [
        (update_vpath(vpath), {'vid': vid, 'label': 'undefined'})
        for vid, vpath in vids_and_paths
    ]
