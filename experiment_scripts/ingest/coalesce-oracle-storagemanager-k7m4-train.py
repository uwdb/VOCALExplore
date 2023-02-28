import argparse
import duckdb
import glob
import os
import re
import shutil

from vfe.core import filesystem
from vfe.featurestore import parquet

def copy_files(source, dest):
    source_id = re.findall(r'-vs(\d+)', source)[0]
    for pq_file in glob.glob(os.path.join(source, '*.parquet')):
        pq_basename = os.path.basename(pq_file)
        shutil.copy(pq_file, os.path.join(dest, f'{source_id}-{pq_basename}'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['kinetics7m4', 'kinetics400'])
    ap.add_argument('--feature-name', default='r3d_18_ap_mean_flatten')
    args = ap.parse_args()

    oracle_dir = f'/gscratch/balazinska/mdaum/video-features-exploration/service/storage/{args.dataset}-train/oracle'
    feature_name = args.feature_name
    features_dir = os.path.join(oracle_dir, 'features', feature_name)
    filesystem.create_dir(features_dir)
    sub_features_dirs = glob.glob(os.path.join(oracle_dir, 'features-vs*'))
    for fd in sub_features_dirs:
        copy_files(os.path.join(fd, feature_name), features_dir)

    with open(os.path.join(features_dir, '_version.txt'), 'w+') as f:
        print('0', file=f)

    parquet.validate_parquet_dir(features_dir)

    con = duckdb.connect()
    max_fid = con.execute("SELECT MAX(fid) FROM read_parquet('{features_dir}/*.parquet')".format(features_dir=features_dir)).fetchall()[0][0]
    with open(os.path.join(features_dir, '_fid.txt'), 'w+') as f:
        print(max_fid + 1, file=f)
