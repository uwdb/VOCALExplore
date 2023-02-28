import argparse
import logging
import os
import shutil
import torch

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
from vfe.api.featuremanager.basic import BasicFeatureManager
from vfe.api.featuremanager.background import BackgroundFeatureManager

def add_base_dir(base_dir, csv_path, out_csv_path):
    first = True
    nlines = 0
    with open(csv_path, 'r') as in_f:
        with open(out_csv_path, 'w+') as out_f:
            print_csv = lambda *args: print(*args, sep=',', file=out_f)
            for line in in_f:
                line = line.strip()
                if first:
                    # Print headers.
                    print_csv(line)
                    first = False
                    continue

                pieces = line.split(',')
                path, rest = pieces[0], pieces[1:]
                path = os.path.join(base_dir, path).replace('.lvm', '.mp4')
                print_csv(path, *rest)
                nlines += 1
    return nlines

def add_base_dir_and_apply(base_dir, csv_path, apply_fn):
    out_csv_path = 'tmp.csv'
    add_base_dir(base_dir, csv_path, out_csv_path)
    return_val = apply_fn(out_csv_path)
    os.remove(out_csv_path)
    return return_val

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-dir')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--num-workers', type=int, default=1)
    ap.add_argument('--device')
    ap.add_argument('--feature-name', default='r3d_18_ap_mean_flatten')
    ap.add_argument('--async-batch-size', type=int, default=100)
    args = ap.parse_args()

    oracle_dir = args.oracle_dir
    features_dir = os.path.join(oracle_dir, 'features')

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
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        fm = BasicFeatureManager(sm, num_workers=args.num_workers, device=device, checkpoint=100, batch_size=11, dali_preprocess=True)
    else:
        fm = BackgroundFeatureManager(sm, num_workers=args.num_workers, device=device, checkpoint=100, batch_size=11, dali_preprocess=True, num_processes=2, async_batch_size=args.async_batch_size, quiettime_async=False)

    # Import video metadata and annotations.
    # These may fail if ingest is already done.
    video_base_dir = '/gscratch/balazinska/mdaum/data/temporal-deer/videos/deer-data/season1'
    metadata_path = '/gscratch/balazinska/mdaum/data/temporal-deer/annotations_videometadata.csv'
    add_base_dir_and_apply(video_base_dir, metadata_path, sm.add_videos)
    labels_path = '/gscratch/balazinska/mdaum/data/temporal-deer/annotations.csv'
    try:
        add_base_dir_and_apply(video_base_dir, labels_path, sm.add_labels_bulk)
    except Exception as e:
        print(f'Failed to ingest labels. This is expected if this script has already been run. Exception {e}')

    # Extract features for all vids.
    feature_name = args.feature_name
    vids = sm.get_all_vids()
    fm.get_features(feature_name, vids)
