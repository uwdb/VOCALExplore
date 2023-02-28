import argparse
import duckdb
import functools
import glob
import json
import os
import re

def add_column_if_necessary(con, col_name, col_type):
    try:
        con.execute(f"ALTER TABLE experiment ADD COLUMN {col_name} {col_type}")
    except:
        # Expect to fail when column already exists.
        pass

def setup_db(db_path):
    con = duckdb.connect(db_path)
    con.execute("CREATE SEQUENCE IF NOT EXISTS exp_seq")

    experiment_exists = 'experiment' in set([
        row[0]
        for row in con.execute("select table_name from information_schema.tables").fetchall()
    ])

    if experiment_exists:
        add_column_if_necessary(con, 'cost_weight', 'FLOAT')
        add_column_if_necessary(con, 'bandit_eval', 'VARCHAR')
    else:
        con.execute("""
            CREATE TABLE IF NOT EXISTS experiment(
                experiment_id INTEGER DEFAULT(nextval('exp_seq')),
                prefix VARCHAR,
                metric VARCHAR,
                C int,
                T int,
                bandit VARCHAR,
                w int,
                logpath VARCHAR,
                explorer VARCHAR,
                validation_size int,
                cost_weight float,
                bandit_eval VARCHAR,
                UNIQUE(logpath),
                PRIMARY KEY (experiment_id)
            )
        """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS elimination(
            experiment_id INTEGER,
            feature VARCHAR,
            step INTEGER,
            FOREIGN KEY (experiment_id) REFERENCES experiment(experiment_id)
        )
    """)

def path_exists(con, path):
    return con.execute("SELECT COUNT(*) FROM EXPERIMENT WHERE logpath=?", [os.path.abspath(path)]).fetchall()[0][0] > 0

def add_experiment(con, prefix, metric, C, T, bandit, window, logpath, explorer, validation_size, cost_weight, bandit_eval):
    con.execute("""
        INSERT INTO experiment (prefix, metric, C, T, bandit, w, logpath, explorer, validation_size, cost_weight, bandit_eval)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [prefix, metric, C, T, bandit, window, logpath, explorer, validation_size, cost_weight, bandit_eval])
    return con.execute("SELECT experiment_id FROM experiment WHERE logpath=?", [logpath]).fetchall()[0][0]

def mark_elimination(con=None, experiment_id=None, feature=None, step=None):
    con.execute("""
        INSERT INTO elimination (experiment_id, feature, step)
        VALUES (?, ?, ?)
    """, [experiment_id, feature, step])

def parse_file(path, mark_elimination_fn, features_already_eliminated):
    try:
        with open(path, 'r') as f:
            info = json.load(f)
    except Exception as e:
        print(f'Exception on {os.path.abspath(path)}: {e}')
        return

    if features_already_eliminated:
        elimination_steps = []
        feature_steps = {feature: len(info[feature]['y_k']) for feature in info.keys()}
        max_step = max(feature_steps.values())
        candidates = set(info.keys())
        for feature, step in feature_steps.items():
            if step < max_step:
                elimination_steps.append(dict(feature=feature, step=step))
                candidates.remove(feature)
        if len(candidates) > 1:
            ordered_candidates = sorted(list(candidates))
            ncandidates = len(candidates)
            for fi in range(ncandidates):
                feati = ordered_candidates[fi]
                if feati not in candidates:
                    continue
                for fj in range(ncandidates):
                    if fi == fj:
                        continue
                    featj = ordered_candidates[fj]
                    if featj not in candidates:
                        continue
                    if info[feati]['l_k'][-1] >= info[featj]['u_k'][-1]:
                        candidates.remove(featj)
                        elimination_steps.append(dict(feature=featj, step=max_step))
        if len(candidates) != 1:
            # print(f'Failed to eliminate down to one feature: {feature_steps}, {candidates}')
            return 0
        for params in elimination_steps:
            mark_elimination_fn(**params)
        mark_elimination_fn(feature=list(candidates)[0], step=max_step+1)
    else:
        features = list(info.keys())
        nsteps = len(info[features[0]]['y_k'])
        candidate_features = set(features)
        for i in range(2, nsteps):
            ncandidates = len(candidate_features)
            ordered_candidates = sorted(list(candidate_features))
            for fi in range(ncandidates):
                for fj in range(ncandidates):
                    if fi == fj:
                        continue
                    featj = ordered_candidates[fj]
                    if featj not in candidate_features:
                        # It's possible featj was already removed this iteration.
                        continue
                    if info[ordered_candidates[fi]]['l_k'][i] >= info[featj]['u_k'][i]:
                        candidate_features.remove(featj)
                        mark_elimination_fn(feature=featj, step=i)
            if len(ordered_candidates) == 1:
                break
        # For plotting, keep track of features kept around at the end.
        if nsteps > 2:
            for feature in ordered_candidates:
                mark_elimination_fn(feature=feature, step=nsteps+1)

    return 1

def process_file(con, path, features_already_eliminated):
    prefix = os.path.basename(path).split('labelt')[0][:-2]
    matches = re.search(r'_ek(.*)_C(\d+)_T(\d+)_b(.*?)[_\.]', path)
    if matches is None:
        # Case for original json files (not simulated output).
        return
    metric = matches[1]
    C = int(matches[2])
    T = int(matches[3])
    bandit = matches[4]
    window_match = re.search('_w(\d+)', path)
    window = int(window_match[1]) if window_match else None
    validation_match = re.search('_v(\d+)-', path)
    validation_size = int(validation_match[1]) if validation_match else None
    explorer = os.path.basename(path).split('_', maxsplit=2)[1]
    cost_weight_match = re.search('_cw(\d+\.\d+)', path)
    cost_weight = float(cost_weight_match[1]) if cost_weight_match else None
    bandit_eval_match = re.search('_be(k\d+)', path)
    bandit_eval = bandit_eval_match[1] if bandit_eval_match else None
    experiment_id = add_experiment(con, prefix, metric, C, T, bandit, window, os.path.abspath(path), explorer, validation_size, cost_weight, bandit_eval)
    return parse_file(path, functools.partial(mark_elimination, con=con, experiment_id=experiment_id), features_already_eliminated)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-path')
    ap.add_argument('--db-path')
    ap.add_argument('--features-not-eliminated', action='store_true')
    args = ap.parse_args()

    setup_db(args.db_path)
    con = duckdb.connect(args.db_path)
    features_already_eliminated = not args.features_not_eliminated

    count = 0.0
    successes = 0.0
    for json_file in glob.glob(os.path.join(args.json_path, '*.json')):
        if path_exists(con, json_file):
            continue
        successes += process_file(con, json_file, features_already_eliminated)
        count += 1
    if count:
        print(f'Successes: {successes}, success rate: {successes / count}')
