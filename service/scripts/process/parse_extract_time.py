import argparse
import datetime
import duckdb
import json
import glob
from pathlib import Path
import os
import random
import re

def add_column_if_necessary(con, col_name, col_type):
    try:
        con.execute(f"ALTER TABLE experiment ADD COLUMN {col_name} {col_type}")
    except:
        # Expect to fail when column already exists.
        pass

def initialize_db(path):
    con = duckdb.connect(path)
    con.execute("CREATE SEQUENCE IF NOT EXISTS exp_seq")

    experiment_exists = 'experiment' in set([
        row[0]
        for row in con.execute("select table_name from information_schema.tables").fetchall()
    ])

    if experiment_exists:
        add_column_if_necessary(con, 'evalontrained', 'UINTEGER')
    else:
        con.execute("""
            CREATE TABLE IF NOT EXISTS experiment(
                experiment_id INTEGER PRIMARY KEY,
                splittype VARCHAR,
                splitidx UINTEGER,
                featurename VARCHAR,
                oracle VARCHAR,
                k UINTEGER,
                labelt UINTEGER,
                watcht UINTEGER,
                playbackspeed UINTEGER,
                nsteps UINTEGER,
                startwithfeatures INTEGER,
                backgroundmm INTEGER,
                backgroundfm INTEGER,
                explorer VARCHAR,
                date TIMESTAMP,
                logpath VARCHAR,
                cpus INTEGER,
                gpus INTEGER,
                evalgroupbyvid UINTEGER,
                strategy VARCHAR,
                condition VARCHAR,
                efl BOOLEAN,
                eful BOOLEAN,
                emt BOOLEAN,
                asyncbandit BOOLEAN,
                usepriority BOOLEAN,
                suspendlowp BOOLEAN,
                evalontrained UINTEGER,
            )
        """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS time(
            experiment_id INTEGER,
            step UINTEGER,
            key VARCHAR,
            time DOUBLE,
            UNIQUE (experiment_id, step, key)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS nfeatures(
            experiment_id INTEGER,
            step UINTEGER,
            nfeatures INTEGER,
            UNIQUE (experiment_id, step)
        )
    """)

def print_csv(*args, file=None):
    updated_args = [a if a is not None else '' for a in args]
    print(*updated_args, sep='|', file=file)

def add_experiment(con, csv, splittype, splitidx, featurename, oracle, k, labelt, watcht, playbackspeed, nsteps, startwithfeatures, backgroundmm, backgroundfm, explorer, date, logpath, cpus, gpus, evalgroupbyvid, strategy, condition, efl, eful, emt, asyncbandit, usepriority, suspendlowp, evalontrained):
    exp_id = con.execute("SELECT nextval('exp_seq')").fetchall()[0][0]
    with open(csv, 'a+') as f:
        print_csv(exp_id, splittype, int(splitidx) if splitidx is not None else None, featurename, oracle, int(k), int(labelt), int(watcht), int(playbackspeed), int(nsteps), int(startwithfeatures), int(backgroundmm), int(backgroundfm), explorer, str(datetime.datetime.strptime(date, '%Y%m%d-%H%M%S')), logpath, int(cpus), int(gpus), int(evalgroupbyvid), strategy, condition, efl, eful, emt, asyncbandit, usepriority, suspendlowp, int(evalontrained), file=f)
    return exp_id

def add_args(csv, *args):
    with open(csv, 'a+') as f:
        print_csv(*args, file=f)

def is_path_processed(con, path):
    results = con.execute("""
        SELECT COUNT(*)
        FROM experiment
        WHERE logpath=?
    """, [path]).fetchall()
    return results[0][0]

date_from_logline = lambda line: datetime.datetime.fromisoformat(' '.join(line.split(' ')[2:4]).replace(',', '.'))

def process_file(con, path, tmp_prefix, min_step):
    logpath = os.path.abspath(path)
    if is_path_processed(con, logpath):
        return

    csvs = {
        'experiment': f'{tmp_prefix}_tmp_experiment.csv',
        'time': f'{tmp_prefix}_tmp_time.csv',
        'nfeatures': f'{tmp_prefix}_tmp_nfeatures.csv',
    }
    for csv in csvs.values():
        if os.path.exists(csv):
            os.remove(csv)

    pieces = logpath.split('_')
    labelt_idx = [i for i, v in enumerate(pieces) if v.startswith('labelt')][0]
    timestamp = pieces[labelt_idx+1].split('.')[0]

    condition = re.search(r'_tex-(.*?)_', logpath)[1]

    setup_kwargs = dict(
        splittype = None,
        splitidx = None,
        featurename = None,
        oracle = None,
        k = -1,
        labelt = -1,
        watcht = -1,
        playbackspeed = -1,
        nsteps = -1,
        startwithfeatures = 1,
        backgroundmm = -1,
        backgroundfm = -1,
        explorer=None,
        date = timestamp,
        logpath = logpath,
        cpus = -1,
        gpus = -1,
        evalgroupbyvid = 0,
        strategy = None,
        condition = condition,
        efl = False,
        eful = False,
        emt = False,
        asyncbandit = False,
        usepriority = False,
        suspendlowp = False,
        evalontrained = 0,
    )
    setup_keys_args = ['explorer', 'labelt', 'watcht', 'nsteps', 'oracle', 'split-type', 'start-with-features', 'split-type', 'split-idx', 'strategy', 'k', 'cpus', 'gpus', 'playback-speed']
    setup_keys_nargs = [('feature-names', 'featurename')]
    setup_keys_flags = [('eval-groupby-vid', ''), ('eager-feature-extraction-labeled', 'efl'), ('eager-feature-extraction-unlabeled', 'eful'), ('eager-model-training', 'emt'), ('async-bandit', ''), ('use-priority', ''), ('suspend-lowp', ''), ('eval-on-trained', '')]

    step = -1
    step_times = []

    bandit_nfeatures = []

    past_setup = False
    selecting_clips = False
    predicting_clips = False

    with open(path, 'r') as f:
        for line in f:
            if not past_setup:
                if line.startswith('Args'):
                    past_setup = True
                    args = json.loads(line.strip().split(' ', maxsplit=1)[1].replace("'", '"'))
                    for key in setup_keys_args:
                        try:
                            key_idx = args.index(f'--{key}')
                            value = args[key_idx + 1]
                            setup_kwargs[key.replace('-', '')] = value
                        except ValueError:
                            pass
                    for arg_key, setup_key in setup_keys_nargs:
                        try:
                            val_idx = args.index(f'--{arg_key}') + 1
                            values = []
                            while True:
                                value = args[val_idx]
                                if value.startswith('--'):
                                    break
                                values.append(value)
                                val_idx += 1
                            setup_kwargs[setup_key] = '+'.join(sorted(values))
                        except ValueError:
                            pass
                    for (flag, key) in setup_keys_flags:
                        try:
                            flag_idx = args.index(f'--{flag}')
                            if not key:
                                key = flag.replace('-', '')
                            setup_kwargs[key] = 1
                        except ValueError:
                            pass
                    experiment_id = add_experiment(con, csvs['experiment'], **setup_kwargs)
            else:
                if line.startswith('*** Step'):
                    for key, value in step_times:
                        add_args(csvs['time'], experiment_id, step, key, value)

                    for bandit_step, nfeatures in bandit_nfeatures:
                        add_args(csvs['nfeatures'], experiment_id, bandit_step, nfeatures)
                    bandit_nfeatures = []

                    step_times = []
                    step = int(line.strip().split(' ')[-1])

                elif 'Rising bandit step' in line:
                    # Combine date + time, and replace ",{ms}" to ".{ms}"
                    rising_bandit_start = date_from_logline(line)

                elif 'and after pruning, candidates are' in line:
                    rising_bandit_end = date_from_logline(line)
                    bandit_duration = (rising_bandit_end - rising_bandit_start).total_seconds()
                    step_times.append(('bandit_duration', bandit_duration))

                elif 'explore: explore (feature' in line:
                    selecting_clips = True
                    select_clips_start = date_from_logline(line)

                elif selecting_clips and 'get_predictions: Getting predictions with feature' in line:
                    selecting_clips = False
                    select_clips_end = date_from_logline(line)
                    step_times.append(('select_clips', (select_clips_end - select_clips_start).total_seconds()))

                    predicting_clips = True
                    predict_clips_start = select_clips_end

                elif 'explore took' in line:
                    if predicting_clips:
                        predict_clips_end = date_from_logline(line)
                        step_times.append(('predict_clips', (predict_clips_end - predict_clips_start).total_seconds()))

                    explore_duration = re.search(r'([\d\.]+)', line.strip().split('explore took', maxsplit=1)[1])[1]
                    step_times.append(('explore', explore_duration))

                elif re.search(r'randomexp explore:.*unlabeled vids with features$', line):
                    # Random sampling log structure.
                    unlabeled_vids_with_features = int(re.search(r'(\d+) unlabeled vids with features', line)[1])
                    step_times.append(('unlabeled_with_feats', unlabeled_vids_with_features))

                elif re.search(r"coresets explore:.*unlabeled vids with features;.*vids without features", line):
                    # Coresets sampling log structure.
                    unlabeled_vids_with_features = int(re.search(r'(\d+) unlabeled vids with features', line)[1])
                    step_times.append(('unlabeled_with_feats', unlabeled_vids_with_features))

                if 'after pruning, candidates are' in line:
                    bandit_step = int(re.search(r'After step (\d+)', line)[1])
                    nfeatures = line.split('candidates are')[1].count(',') + 1
                    bandit_nfeatures.append((bandit_step, nfeatures))

    for key, value in step_times:
        add_args(csvs['time'], experiment_id, step, key, value)

    for bandit_step, nfeatures in bandit_nfeatures:
        add_args(csvs['nfeatures'], experiment_id, bandit_step, nfeatures)

    if min_step < 0:
        min_step = int(setup_kwargs['nsteps']) - 1
    if step < min_step:
        print(f'Aborting processing because missing steps (max={step})', logpath)
    else:
        add_results_for_csv = lambda table, csv: con.execute(f"INSERT INTO {table} SELECT * FROM read_csv_auto('{csv}', delim='|', header=False)")

        con.execute("BEGIN TRANSACTION")
        try:
            for table, csv in csvs.items():
                if os.path.exists(csv):
                    add_results_for_csv(table, csv)
            con.execute("COMMIT")
        except Exception as e:
            con.execute("ROLLBACK TRANSACTION")

    for csv_path in csvs.values():
        if os.path.exists(csv_path):
            os.remove(csv_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir')
    ap.add_argument('--db-path')
    ap.add_argument('--min-step', type=int, default=-1)
    args = ap.parse_args()

    db_path = args.db_path
    initialize_db(db_path)

    con = duckdb.connect(db_path)
    log_dir = args.log_dir
    min_step = args.min_step
    tmp_prefix = f'{Path(log_dir).parent.name}-{random.randint(0, 10000)}'
    count = 0
    for file in glob.glob(os.path.join(log_dir, 'log*.txt')):
        count += 1
        if count and count % 50 == 0:
            print(f'Num done: {count}')

        # try:
        process_file(con, file, tmp_prefix, min_step)
        # except Exception as e:
        #     print(f'Exception for file {os.path.abspath(file)}: {e}')

    # Correct for naming problem in bash script where suspend=False but it's in the name.
    con.execute("""
        UPDATE experiment
        SET condition=replace(condition, 'suspend', '')
        WHERE suspendlowp=False AND condition like '%eful%'
    """)
    con.commit()
