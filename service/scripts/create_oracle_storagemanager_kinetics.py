import argparse
import datetime
import os
import shutil
import sys
import torch

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
from vfe.api.featuremanager.basic import BasicFeatureManager
from vfe.api.featuremanager.background import BackgroundFeatureManager

def print_and_flush(*args):
    print(*args)
    sys.stdout.flush()

if __name__ == '__main__':
    core.logging.configure_logger()

    print_and_flush('Time', datetime.datetime.now())
    print_and_flush('Args:', sys.argv)

    ap = argparse.ArgumentParser()
    ap.add_argument('--split')
    ap.add_argument('--oracle-dir')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device')
    ap.add_argument('--vid-split', type=int)
    ap.add_argument('--dataset', default='kinetics700-400', choices=['kinetics700-400', 'kinetics400'])
    ap.add_argument('--feature-name', default='r3d_18_ap_mean_flatten')
    ap.add_argument('--async-batch-size', type=int, default=100)
    ap.add_argument('--checkpoint', type=int, default=100)
    args = ap.parse_args()

    split = args.split
    oracle_dir = args.oracle_dir + f'-{split}'
    if split == 'train':
        oracle_dir = os.path.join(oracle_dir, 'oracle')
    features_dir = os.path.join(oracle_dir, 'features')
    dataset = args.dataset

    if args.vid_split is None:
        # Delete existing directory if specified.
        if os.path.exists(oracle_dir) and args.overwrite:
            if args.force:
                should_overwrite = True
            else:
                should_overwrite = input(f'WARNING: Delete {oracle_dir}? ').lower() == 'y'
            if should_overwrite:
                print(f'Deleting {oracle_dir}')
                shutil.rmtree(oracle_dir)
        core.filesystem.create_dir(oracle_dir)

        # Create oracle storage manager.
        sm = StorageManager(db_dir=oracle_dir, features_dir=features_dir, models_dir=None)

        metadata_path = f'/gscratch/balazinska/mdaum/data/{dataset}/{split}_videometadata.csv'
        sm.add_videos(metadata_path)

        try:
            labels_path = f'/gscratch/balazinska/mdaum/data/{dataset}/{split}_annotations.csv'
            sm.add_labels_bulk(labels_path)
        except Exception as e:
            print(f'Failed to ingest labels. this is expected if this script has already been run. Exception {e}')

    # Feature extraction.
    vid_split = args.vid_split
    if vid_split is not None:
        vids_len = 15000
        start_idx = vid_split * vids_len
        suffix = f'-vs{vid_split}'
        fid_offset = start_idx * 20
    else:
        vids_len = 0
        start_idx = 0
        suffix = ''
        fid_offset = 0

    sm = StorageManager(db_dir=oracle_dir, features_dir=features_dir + suffix, models_dir=None, read_only=True, fid_offset=fid_offset)
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        fm = BasicFeatureManager(sm, num_workers=args.num_workers, device=device, checkpoint=args.checkpoint, batch_size=11, dali_preprocess=True)
    else:
        fm = BackgroundFeatureManager(sm, num_workers=args.num_workers, device=device, checkpoint=args.checkpoint, batch_size=11, dali_preprocess=True, num_processes=2, async_batch_size=args.async_batch_size, quiettime_async=False)
    feature_name = args.feature_name
    vids = sorted(sm.get_all_vids())

    start_idx = min(start_idx, len(vids))
    end_idx = min(start_idx + vids_len, len(vids)) if vids_len else len(vids)
    fm.get_features(feature_name, vids[start_idx:end_idx])
