import argparse
import duckdb
import numpy as np
import os
import shutil

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager

get_db_path = lambda dir: os.path.join(dir, 'annotations.duckdb')

def initialize_environment(db_dir):
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    paths = (db_dir, os.path.join(db_dir, 'features'), os.path.join(db_dir, 'models'))
    for path in paths:
        core.filesystem.create_dir(path)
    return paths

def get_split_path(split_dir, split_type, split_idx, split):
    return os.path.join(split_dir, split_type, f'split-{split_idx}', f'{split}.csv')

def get_split_vids(split_dir, split_type, split_idx, split, sm):
    split_csv = get_split_path(split_dir, split_type, split_idx, split)
    return sm.get_vids_for_paths(split_csv)

def setup_dirs(db_dir, oracle_dir, oracle_dump_dir, split_dir, split_type, split_idx, include_features, include_labels=False, feature_names=None):
    db_dir, features_dir, models_dir = initialize_environment(db_dir)
    initialize_storage_manager(oracle_dir, oracle_dump_dir, db_dir, get_split_path(split_dir, split_type, split_idx, 'train') if split_type is not None else None, include_features=include_features, include_labels=include_labels, feature_names=feature_names)
    return db_dir, features_dir, models_dir

def dump_db(db_dir, dump_dir):
    con = duckdb.connect(get_db_path(db_dir), read_only=True)
    con.execute(f"EXPORT DATABASE '{dump_dir}'") # Formatted string isn't ideal, but read only connection.
    con.close()

def import_db(db_dir, dump_dir, vpath_filter_file, include_labels=False):
    assert os.path.exists(dump_dir)
    con = duckdb.connect(get_db_path(db_dir))
    con.execute(f"IMPORT DATABASE '{dump_dir}'")
    # Get all tables besides video_metadata.
    tables = con.execute("DESCRIBE TABLES").fetchall()
    keep_tables = set(['video_metadata'])
    if include_labels:
        keep_tables.add('annotations')
    for row in tables:
        table = row[0]
        if table not in keep_tables:
            print(f'Dropping table {table}')
            con.execute(f"DROP TABLE {table}")
    if vpath_filter_file:
        con.execute(f"CREATE TEMP TABLE vpaths AS SELECT * FROM read_csv_auto('{vpath_filter_file}', header=True)")
        if include_labels:
            con.execute("""
                DELETE FROM annotations
                WHERE vid NOT IN (
                    SELECT vid FROM vpaths v, video_metadata m where v.vpath=m.vpath
                )
            """)
        con.execute("""
            DELETE FROM video_metadata
            WHERE vpath NOT IN (
                SELECT vpath FROM vpaths
            )
        """)

def import_features(target_dir, base_db_dir, base_features_dir, needs_filtering=True):
    assert os.path.exists(base_db_dir)
    assert os.path.exists(base_features_dir)
    if needs_filtering:
        if not os.path.exists(target_dir):
            core.filesystem.create_dir(target_dir)
        con = duckdb.connect(get_db_path(base_db_dir), read_only=True)
        source_files = os.path.join(base_features_dir, '*.parquet')
        con.execute(f"CREATE TEMP TABLE features AS SELECT * FROM read_parquet('{source_files}')")
        con.execute("""
            CREATE TEMP TABLE filtered_features AS
                SELECT f.* FROM features f
                WHERE f.vid IN (SELECT vid FROM video_metadata)
        """)
        output_file = os.path.join(target_dir, '0.parquet')
        con.execute(f"COPY filtered_features TO '{output_file}' (FORMAT PARQUET)")
        # These metadata files are tied to the setup code in parquet.py.
        with open(os.path.join(target_dir, '_fid.txt'), 'w+') as f:
            fid = con.execute("SELECT MAX(fid) FROM filtered_features").fetchall()[0][0]
            print(fid + 1, file=f)
        with open(os.path.join(target_dir, '_version.txt'), 'w+') as f:
            print('1', file=f)
    else:
        print('Copying features without filtering')
        shutil.copytree(base_features_dir, target_dir)

def initialize_storage_manager(oracle_dir, export_dir, db_dir, vpath_filter_file, include_features=False, include_labels=False, feature_names=None):
    if not os.path.exists(export_dir) or not len(os.listdir(export_dir)):
        print(f'Dumping database from {oracle_dir} to {export_dir}')
        dump_db(oracle_dir, export_dir)
    else:
        print(f'Database dump already exists at {export_dir}')

    if os.path.exists(db_dir):
        print(f'Removing existing db_dir at {db_dir}')
        shutil.rmtree(db_dir)
    core.filesystem.create_dir(db_dir)
    print(f'Importing database from {export_dir} to {db_dir}')
    print(f'Filtering to vpaths found in {vpath_filter_file}')
    import_db(db_dir, export_dir, vpath_filter_file, include_labels=include_labels)
    if include_features:
        for feature in os.scandir(os.path.join(oracle_dir, 'features')):
            if not feature.is_dir():
                continue
            if feature_names is not None and feature.name not in feature_names:
                continue
            target_dir = os.path.join(db_dir, 'features', feature.name)
            base_dir = feature.path
            import_features(target_dir, db_dir, base_dir, needs_filtering=vpath_filter_file)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle-dir')
    ap.add_argument('--oracle-export-dir')
    ap.add_argument('--db-dir')
    ap.add_argument('--vpath-filter-file')
    args = ap.parse_args()

    oracle_dir = args.oracle_dir
    export_dir = args.oracle_export_dir
    db_dir = args.db_dir
    vpath_filter_file = args.vpath_filter_file
    initialize_storage_manager(oracle_dir, export_dir, db_dir, vpath_filter_file)
