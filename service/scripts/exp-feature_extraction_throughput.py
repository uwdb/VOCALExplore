import argparse
import atexit
import datetime
from functools import partial
import itertools
import os
import shutil
import sys
import time

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
from vfe.api.featuremanager.background import BackgroundFeatureManager

from initialize_storage_manager import setup_dirs

def print_and_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def cleanup(db_dir=None):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

if __name__ == '__main__':
    core.logging.configure_logger()

    print_and_flush('Time', datetime.datetime.now())
    print_and_flush('Args:', sys.argv)

    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset')
    ap.add_argument('--db-dir')
    ap.add_argument('--oracle-dir')
    ap.add_argument('--oracle-dump-dir')
    ap.add_argument('--feature-name')
    ap.add_argument('--no-store', action='store_true')
    args = ap.parse_args()

    dataset_name = args.dataset
    base_db_dir = args.db_dir
    oracle_dir = args.oracle_dir
    oracle_dump_dir = args.oracle_dump_dir
    feature_name = args.feature_name
    no_store = args.no_store

    vid_ssd_options = [False] # [True, False]
    feat_ssd_options = [True] # [True, False]
    use_dali_options = [True] # [True, False]
    nworkers_options = [4]
    npipe_options = [2] # [1, 2, 3]

    for i in range(2):
        for vid_ssd, feat_ssd, use_dali, nworkers, npipe in itertools.product(vid_ssd_options, feat_ssd_options, use_dali_options, nworkers_options, npipe_options):
            if use_dali and nworkers != 4:
                # nworkers doesn't matter for dali. Skip.
                continue
            if not use_dali and npipe != 1:
                # npipe doesn't matter for pytorch. Skip.
                continue

            print_and_flush(f'vid_ssd: {vid_ssd}, feat_ssd: {feat_ssd}, use_dali: {use_dali}, nworkers: {nworkers}, npipe: {npipe}')

            iter_db_dir = base_db_dir if not feat_ssd else base_db_dir.replace('/gscratch/balazinska/', '/scr/')
            db_dir, features_dir, models_dir = setup_dirs(iter_db_dir, oracle_dir, oracle_dump_dir, None, None, None, False, False)

            cleanup_fun = partial(cleanup, db_dir=db_dir)
            atexit.register(cleanup_fun)

            sm = StorageManager(db_dir=db_dir, features_dir=features_dir, models_dir=models_dir)
            batch_size = 8 if use_dali else 1
            nvids = 100
            async_batch_size = nvids // npipe
            if nvids % npipe:
                async_batch_size += 1
            fm = BackgroundFeatureManager(sm, num_workers=nworkers, batch_size=batch_size, checkpoint=10, num_processes=npipe, dali_preprocess=use_dali, async_batch_size=async_batch_size, quiettime_async=False, vid_ssd=vid_ssd, no_store=no_store)
            vids = sorted(sm.get_all_vids())[:nvids]

            start = time.perf_counter()
            fm.get_features(feature_name, vids)
            end = time.perf_counter()
            print_and_flush(f'Results: {{dataset_name: {dataset_name} | feature_name: {feature_name} | vid_ssd: {vid_ssd} | feat_ssd: {feat_ssd} | use_dali: {use_dali} | nworkers: {nworkers} | npipe: {npipe} | batch_size: {batch_size} | no_store: {no_store} | vidtime: {(end - start) / float(nvids)}}}')

            fm.shutdown()
            sm.shutdown()
            fm = None
            sm = None
            cleanup_fun()
            time.sleep(2)
