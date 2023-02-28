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
                explorelabel VARCHAR,
                explorelabelthreshold INTEGER,
                efl BOOLEAN,
                eful BOOLEAN,
                emt BOOLEAN,
                asyncbandit BOOLEAN,
                usepriority BOOLEAN,
                suspendlowp BOOLEAN,
                evalontrained UINTEGER,
                noreturnpredictions BOOLEAN,
                alvidsx INTEGER,
                serialkfold BOOLEAN,
            )
        """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS results(
            experiment_id INTEGER,
            step UINTEGER,
            label VARCHAR,
            ap DOUBLE,
            UNIQUE (experiment_id, step, label)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS label_f1_results(
            experiment_id INTEGER,
            step UINTEGER,
            label VARCHAR,
            f1 DOUBLE,
            UNIQUE (experiment_id, step, label)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS generic_aggregate_results(
            experiment_id INTEGER,
            step UINTEGER,
            metric VARCHAR,
            value DOUBLE,
            UNIQUE (experiment_id, step, metric)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS aggregate_results(
            experiment_id INTEGER,
            step UINTEGER,
            mc_f1_score_micro DOUBLE,
            mc_f1_score_macro DOUBLE,
            mc_accuracy_micro DOUBLE,
            mc_accuracy_macro DOUBLE,
            mc_accuracy_top5_micro DOUBLE,
            mc_accuracy_top5_macro DOUBLE,
            UNIQUE (experiment_id, step)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS labelcounts(
            experiment_id INTEGER,
            step UINTEGER,
            label VARCHAR,
            count UINTEGER,
            UNIQUE (experiment_id, step, label)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS trainingsize(
            experiment_id INTEGER,
            step UINTEGER,
            trainingsize UINTEGER,
            UNIQUE (experiment_id, step)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS evalsize(
            experiment_id INTEGER,
            step UINTEGER,
            evalsize UINTEGER,
            UNIQUE (experiment_id, step)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS exploretime(
            experiment_id INTEGER,
            step UINTEGER,
            exploretime DOUBLE,
            UNIQUE (experiment_id, step)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS evalfeature(
            experiment_id INTEGER,
            step UINTEGER,
            featurename VARCHAR,
            UNIQUE (experiment_id, step)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS stepdouble(
            experiment_id INTEGER,
            step UINTEGER,
            key VARCHAR,
            value DOUBLE,
            UNIQUE (experiment_id, step, key)
        )
    """)

def print_csv(*args, file=None):
    updated_args = [a if a is not None else '' for a in args]
    print(*updated_args, sep='|', file=file)

def add_experiment(con, csv, splittype, splitidx, featurename, oracle, k, labelt, watcht, playbackspeed, nsteps, startwithfeatures, backgroundmm, backgroundfm, explorer, date, logpath, cpus, gpus, evalgroupbyvid, strategy, explorelabel, explorelabelthreshold, efl, eful, emt, asyncbandit, usepriority, suspendlowp, evalontrained, noreturnpredictions, alvidsx, serialkfold):
    exp_id = con.execute("SELECT nextval('exp_seq')").fetchall()[0][0]
    with open(csv, 'a+') as f:
        print_csv(exp_id, splittype, int(splitidx) if splitidx is not None else None, featurename, oracle, int(k), int(labelt), int(watcht), int(playbackspeed), int(nsteps), int(startwithfeatures), int(backgroundmm), int(backgroundfm), explorer, str(datetime.datetime.strptime(date, '%Y%m%d-%H%M%S')), logpath, int(cpus), int(gpus), int(evalgroupbyvid), strategy, explorelabel, explorelabelthreshold, efl, eful, emt, asyncbandit, usepriority, suspendlowp, int(evalontrained), noreturnpredictions, alvidsx, serialkfold, file=f)
    return exp_id

def add_results(csv, experiment_id, step, label, ap):
    with open(csv, 'a+') as f:
        print_csv(experiment_id, step, label, ap, file=f)

def add_aggregate_results(csv, experiment_id, step, mc_f1_score_micro, mc_f1_score_macro, mc_accuracy_micro, mc_accuracy_macro, mc_accuracy_top5_micro, mc_accuracy_top5_macro):
    with open(csv, 'a+') as f:
        print_csv(experiment_id, step, mc_f1_score_micro, mc_f1_score_macro, mc_accuracy_micro, mc_accuracy_macro, mc_accuracy_top5_micro, mc_accuracy_top5_macro, file=f)

def add_counts(csv, experiment_id, step, label, count):
    with open(csv, 'a+') as f:
        print_csv(experiment_id, step, label, count, file=f)

def add_trainingsize(csv, experiment_id, step, trainingsize):
    with open(csv, 'a+') as f:
        print_csv(experiment_id, step, trainingsize, file=f)

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

def process_file(con, path, tmp_prefix, min_step=-1, try_f1_if_exists=False):
    logpath = os.path.abspath(path)
    process_only_f1 = False
    experiment_id = None
    if is_path_processed(con, logpath):
        # print('Skipping already-processed file', path)
        if not try_f1_if_exists:
            return
        f1_is_handled = con.execute("""
            SELECT count(*)
            FROM label_f1_results
            WHERE experiment_id IN (
                SELECT experiment_id
                FROM experiment
                WHERE logpath=?
            )
        """, [logpath]).fetchall()[0][0]
        if f1_is_handled:
            return
        process_only_f1 = True
        experiment_id = con.execute("SELECT experiment_id FROM experiment WHERE logpath=?", [logpath]).fetchall()[0][0]

    csvs = {
        'experiment': f'{tmp_prefix}_tmp_experiment.csv',
        'results': f'{tmp_prefix}_tmp_results.csv',
        'aggregate_results': f'{tmp_prefix}_tmp_aggresults.csv',
        'labelcounts': f'{tmp_prefix}_tmp_counts.csv',
        'trainingsize': f'{tmp_prefix}_tmp_trainingsize.csv',
        'evalsize': f'{tmp_prefix}_tmp_evalsize.csv',
        'exploretime': f'{tmp_prefix}_tmp_exploretime.csv',
        'evalfeature': f'{tmp_prefix}_tmp_evalfeature.csv',
        'label_f1_results': f'{tmp_prefix}_tmp_label_f1_results.csv',
        'generic_aggregate_results': f'{tmp_prefix}_tmp_generic_aggregate_results.csv',
        'stepdouble': f'{tmp_prefix}_tmp_stepdouble.csv',
    }
    for csv in csvs.values():
        if os.path.exists(csv):
            os.remove(csv)

    # print('Processing log at', os.path.abspath(path))
    basename = os.path.basename(path).replace('_pval', '-pval')
    pieces = basename.split('_')
    labelt_idx = [i for i, v in enumerate(pieces) if v.startswith('labelt')][0]
    timestamp = pieces[labelt_idx+1].split('.')[0]

    setup_kwargs = dict(
        explorer = None,
        splittype = None,
        splitidx = None,
        labelt = None,
        date = timestamp,
        logpath = logpath,
        startwithfeatures = 1,
        backgroundmm = -1,
        backgroundfm = -1,
        oracle = None,
        cpus = -1,
        gpus = -1,
        evalgroupbyvid = 0,
        strategy = None,
        explorelabel = None,
        explorelabelthreshold = -1,
        efl = False,
        eful = False,
        emt = False,
        asyncbandit = False,
        usepriority = False,
        suspendlowp = False,
        evalontrained = 0,
        noreturnpredictions = False,
        alvidsx = -1,
        serialkfold = False,
    )
    setup_keys = ['featurename', 'watcht', 'playbackspeed', 'nsteps', 'startwithfeatures', 'backgroundmm', 'backgroundfm', 'cpus', 'gpus']
    setup_keys_args = ['explorer', 'labelt', 'nsteps', 'oracle', 'split-type', 'feature-name', 'watcht', 'playback-speed', 'nsteps', 'start-with-features', 'cpus', 'gpus', 'split-type', 'split-idx', 'strategy', 'k', 'explore-label', 'explore-label-threshold', 'al-vids-x']
    setup_keys_nargs = [('feature-names', 'featurename')]
    setup_keys_flags = [('eval-groupby-vid', ''), ('eager-feature-extraction-labeled', 'efl'), ('eager-feature-extraction-unlabeled', 'eful'), ('eager-model-training', 'emt'), ('async-bandit', ''), ('use-priority', ''), ('suspend-lowp', ''), ('eval-on-trained', ''), ('no-return-predictions', ''), ('serial-kfold', '')]
    past_setup = False
    in_eval = False
    step_results = []
    step_agg_results = {}
    agg_metrics = ['mc_f1_score_micro', 'mc_f1_score_macro', 'mc_accuracy_micro', 'mc_accuracy_macro', 'mc_accuracy_top5_micro', 'mc_accuracy_top5_macro']

    step_f1_results = []
    generic_metric_results = [] # (metric, value)
    generic_metrics = ['ml_f1_score_macro',]

    step_counts = []
    step_trainingsize = -1
    step_evalsize = -1
    step = -1
    step_exploretime = None
    step_evalfeature = None

    step_doubles = []

    with open(path, 'r') as f:
        for line in f:
            if not past_setup:
                # Don't do this for nvidia-smi print statements.
                if '=' in line and "===" not in line:
                    key, value = line.strip().split('=')
                    if key in setup_keys:
                        setup_kwargs[key] = value.replace('"', '') # Remove quotes around strings.
                if line.startswith('Args'):
                    # if '--no-eval-on-test' in line:
                    #     # Skip because this log doesn't have evaluation information.
                    #     return

                    if path.endswith('_nof.txt'):
                        assert int(setup_kwargs['startwithfeatures']) == 0, f'Unexpected startwithfeatures at {logpath}'
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
                    if experiment_id is None:
                        if min_step > 0 and int(setup_kwargs['nsteps'])-1 < min_step:
                            print(f'Aborting processing because missing steps (max possible={setup_kwargs["nsteps"]})', logpath)
                            return
                        experiment_id = add_experiment(con, csvs['experiment'], **setup_kwargs)
            else:
                if line.startswith('*** Step'):
                    if step_results:
                        for label, ap in step_results:
                            add_results(csvs['results'], experiment_id, step, label, ap)
                    if step_agg_results:
                        add_aggregate_results(csvs['aggregate_results'], experiment_id, step, **step_agg_results)
                    if step_counts:
                        for label, count in step_counts:
                            add_counts(csvs['labelcounts'], experiment_id, step, label, count)
                    if step_trainingsize >= 0:
                        add_trainingsize(csvs['trainingsize'], experiment_id, step, step_trainingsize)
                    if step_evalsize >= 0:
                        add_args(csvs['evalsize'], experiment_id, step, step_evalsize)
                    if step_exploretime is not None:
                        add_args(csvs['exploretime'], experiment_id, step, step_exploretime)
                    if step_evalfeature is not None:
                        add_args(csvs['evalfeature'], experiment_id, step, step_evalfeature)
                    if step_f1_results:
                        for label, f1 in step_f1_results:
                            add_args(csvs['label_f1_results'], experiment_id, step, label, f1)
                    if generic_metric_results:
                        for metric, value in generic_metric_results:
                            add_args(csvs['generic_aggregate_results'], experiment_id, step, metric, value)
                    if step_doubles:
                        for key, value in step_doubles:
                            add_args(csvs['stepdouble'], experiment_id, step, key, value)

                    step_results = []
                    step_agg_results = {}
                    step_counts = []
                    step_trainingsize = -1
                    step_exploretime = None
                    step_evalfeature = None
                    step_f1_results = []
                    generic_metric_results = []
                    step_doubles = []
                    step = int(line.strip().split(' ')[-1])

                elif '*** Performance' in line and in_eval:
                    in_eval = False
                    # Special case where we didn't evaluate.
                    if '*** Performance: 0' in line:
                        continue

                    results_dict = json.loads(line.strip().split('*** Performance ')[-1].replace("person's", "persons").replace("'", '"').replace('nan', '"-1"'))
                    agg_results_keys = [f'pred_{metric}' for metric in agg_metrics]
                    generic_metrics_keys = [f'pred_{metric}' for metric in generic_metrics]
                    for key, value in results_dict.items():
                        if '_avgprecision_' in key:
                            label = key.split('_', maxsplit=2)[-1]
                            step_results.append([label, value])
                        elif key in agg_results_keys:
                            step_agg_results[key.split('_', maxsplit=1)[1]] = value
                        elif key in generic_metrics_keys:
                            generic_metric_results.append([key.split('_', maxsplit=1)[1], value])
                        elif key.startswith('pred_ml_f1_score_') and 'macro' not in key and 'micro' not in key:
                            label = key.split('pred_ml_f1_score_', maxsplit=1)[1]
                            step_f1_results.append([label, value])

                elif '*** Evaluating labels' in line:
                    in_eval = True

                elif in_eval and '---skip--- Label counts' in line:
                    counts_dict = json.loads(line.strip().split('Label counts: ')[-1].replace("person's", "persons").replace("'", '"'))
                    for label, count in counts_dict.items():
                        step_counts.append([label, count])

                # There will be two such lines in the evaluation section. The first trains over the complete
                # labeled set, and the second trains over a subset to set the f1 threshold. Only take the counts
                # from the first one.
                elif in_eval and 'y_train (len=' in line and not len(step_counts):
                    counts = re.search(r'y_train.*(\[.*\])', line)[1]
                    counts = json.loads(counts.replace("person's", "persons").replace("'", '"').replace('(', '[').replace(')', ']'))
                    # Counts: [[label, count],...]
                    step_counts.extend([[l, c] for l, c in counts if c > 0])

                elif in_eval and 'Training set size' in line:
                    step_trainingsize = int(line.strip().rsplit(' ', maxsplit=1)[-1])

                elif in_eval and 'Results for split pred (len=' in line:
                    step_evalsize = int(re.search(r'Results for split pred \(len=(\d+)\)', line)[1])

                elif 'multifeature explore: explore (feature ' in line:
                    step_evalfeature = re.search(r"feature \['(.*)'\]", line)[1]

                # elif in_eval and 'Reading features from' in line:
                #     feature = line.strip().rsplit('/', maxsplit=1)[-1]
                #     if step_evalfeature is not None and '+' not in setup_kwargs['featurename']:
                #         assert feature == step_evalfeature
                #     else:
                #         step_evalfeature = feature if '+' not in setup_kwargs['featurename'] else None

                elif 'explore took' in line:
                    duration = re.search(r'took (.*) seconds', line)[1]
                    assert step_exploretime is None
                    step_exploretime = duration

                elif in_eval and 'Passed multilabel_threshold=' in line:
                    f1_threshold = re.search(r'multilabel_threshold=([\d\.]+);', line)[1]
                    best_threshold = re.search(r'best on split=([\d\.]+)', line)[1]
                    step_doubles.append(('f1_threshold', float(f1_threshold)))
                    step_doubles.append(('best_threshold', float(best_threshold)))

        if step_results:
            for label, ap in step_results:
                add_results(csvs['results'], experiment_id, step, label, ap)
        if step_agg_results:
            add_aggregate_results(csvs['aggregate_results'], experiment_id, step, **step_agg_results)
        if step_counts:
            for label, count in step_counts:
                add_counts(csvs['labelcounts'], experiment_id, step, label, count)
        if step_trainingsize >= 0:
            add_trainingsize(csvs['trainingsize'], experiment_id, step, step_trainingsize)
        if step_evalsize >= 0:
            add_args(csvs['evalsize'], experiment_id, step, step_evalsize)
        if step_evalfeature is not None:
            add_args(csvs['evalfeature'], experiment_id, step, step_evalfeature)
        if step_exploretime is not None:
            add_args(csvs['exploretime'], experiment_id, step, step_exploretime)
        if step_f1_results:
            for label, f1 in step_f1_results:
                add_args(csvs['label_f1_results'], experiment_id, step, label, f1)
        if generic_metric_results:
            for metric, value in generic_metric_results:
                add_args(csvs['generic_aggregate_results'], experiment_id, step, metric, value)
        if step_doubles:
            for key, value in step_doubles:
                add_args(csvs['stepdouble'], experiment_id, step, key, value)

        if min_step <= 0 and past_setup:
            min_step = int(setup_kwargs['nsteps']) - 1
        if not past_setup or step < min_step:
            # The log file is missing steps. Don't save it.
            print(f'Aborting processing because missing steps (max={step})', logpath)
        else:
            add_results_for_csv = lambda table, csv: con.execute(f"INSERT INTO {table} SELECT * FROM read_csv_auto('{csv}', delim='|', header=False)")

            con.execute("BEGIN TRANSACTION")
            try:
                for table, csv in csvs.items():
                    if process_only_f1 and table not in ('label_f1_results', 'generic_aggregate_results'):
                        continue
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
    ap.add_argument('--stop-after', type=int, default=-1)
    ap.add_argument('--f1-only', action='store_true')
    args = ap.parse_args()

    db_path = args.db_path
    initialize_db(db_path)

    con = duckdb.connect(db_path)
    log_dir = args.log_dir
    min_step = args.min_step
    stop_after = args.stop_after
    f1_only = args.f1_only
    tmp_prefix = f'{Path(log_dir).parent.name}-{random.randint(0, 10000)}'
    count = 0
    for file in glob.glob(os.path.join(log_dir, 'log*.txt')):
        if 'bandit-noelim' in file:
            continue
        count += 1
        if count and count % 50 == 0:
            print(f'Num done: {count}')
        if stop_after > 0 and count > stop_after:
            break

        try:
            process_file(con, file, tmp_prefix, min_step, try_f1_if_exists=f1_only)
        except Exception as e:
            print(f'Exception for file {os.path.abspath(file)}; {e}')

    con.execute("UPDATE experiment SET oracle='exact' WHERE oracle is NULL")
