import argparse
import duckdb
import json
import glob
import os
import re

def initialize_db(path):
    con = duckdb.connect(path)
    con.execute("CREATE SEQUENCE IF NOT EXISTS exp_seq")
    con.execute("""
        CREATE TABLE IF NOT EXISTS experiment(
            experiment_id INTEGER PRIMARY KEY DEFAULT(nextval('exp_seq')),
            dataset_name VARCHAR,
            feature_name VARCHAR,
            vid_ssd BOOLEAN,
            feat_ssd BOOLEAN,
            use_dali BOOLEAN,
            nworkers UINTEGER,
            npipe UINTEGER,
            batch_size UINTEGER,
            vidtime DOUBLE,
            logpath VARCHAR
        )
    """)

def add_results(con, dataset_name, feature_name, vid_ssd, feat_ssd, use_dali, nworkers, npipe, batch_size, vidtime, logpath):
    con.execute("""
        INSERT INTO experiment
        (dataset_name, feature_name, vid_ssd, feat_ssd, use_dali, nworkers, npipe, batch_size, vidtime, logpath)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [dataset_name, feature_name, vid_ssd, feat_ssd, use_dali, int(nworkers), int(npipe), int(batch_size), float(vidtime), logpath])

def is_path_processed(con, path):
    results = con.execute("""
        SELECT count(*)
        FROM experiment
        WHERE logpath=?
    """, [path]).fetchall()
    return results[0][0]

def process_file(con, path):
    logpath = os.path.abspath(path)
    if is_path_processed(con, logpath):
        return

    with open(logpath, 'r') as f:
        for line in f:
            if not line.startswith('Results: {'):
                continue

            vals = {}
            for key in ['dataset_name', 'feature_name', 'vid_ssd', 'feat_ssd', 'use_dali', 'nworkers', 'npipe', 'batch_size', 'vidtime']:
                pattern = re.compile(f'{key}: (.*?)[ }}]')
                value = pattern.search(line)[1]
                vals[key] = value

            add_results(con, logpath=logpath, **vals)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir')
    ap.add_argument('--db-path')
    args = ap.parse_args()

    db_path = args.db_path
    initialize_db(db_path)

    con = duckdb.connect(db_path)
    log_dir = args.log_dir
    for file in glob.glob(os.path.join(log_dir, 'log*.txt')):
        process_file(con, file)
