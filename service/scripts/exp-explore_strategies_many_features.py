import argparse
import atexit
import datetime
import duckdb
from functools import partial
import logging
import os
import re
import shutil
import sys
import time

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
from vfe.api.videomanager.basic import BasicVideoManager as VideoManager
from vfe.api.featuremanager.basic import BasicFeatureManager
from vfe.api.featuremanager.background_directed import BackgroundAsyncFeatureManager
from vfe.api.modelmanager.background_directed import BackgroundAsyncModelManager
from vfe.api.activelearningmanager.multifeature import MultiFeatureActiveLearningManager, FeatureEvalStrategy
from vfe.api.activelearningmanager import explorers
from vfe.api.activelearningmanager.risingbandit import BanditTypes
from vfe.api.scheduler import Priority, UserPriority, ChorePriority
from vfe.api.scheduler.priority import PriorityScheduler

from initialize_storage_manager import setup_dirs, get_split_vids
from user_interaction import ExploreUser, ExploreLabelUser, OracleLabeler, OracleContextLabeler

def print_and_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def EXPLORER_TO_CLASS(explorer):
    if explorer == 'random':
        return explorers.RandomExplorer
    elif explorer == 'temporal':
        return explorers.TemporalExplorer
    elif explorer == 'coreset':
        return explorers.CoresetsExplorer
    elif explorer == 'cluster':
        return explorers.ClusterExplorer
    elif explorer == 'clustercoreset':
        return explorers.ClusterCoresetsExplorer
    elif explorer.startswith('randomifuniform'):
        return explorers.AKSRandomIfUniformExplorer
    elif explorer.startswith('cvmrandomifuniform'):
        return explorers.CVMRandomIfUniformExplorer

def cleanup(models_dir=None, features_dir=None, db_dir=None):
    shutil.rmtree(db_dir)

def feature_to_tput_feature(feature_name):
    return feature_name

    # This is for the when we want to use the cost of getting the un-pooled
    # features.
    if 'pool' not in feature_name:
        return feature_name
    matches = re.search(r'_(\d+)x(\d+).*pool', feature_name)
    sequence_length = int(matches[1])
    stride = int(matches[2])
    base = feature_name.rsplit('_', maxsplit=1)[0]
    return base + f'_{stride}fstride'

COST_DB_PATH = '/gscratch/balazinska/mdaum/video-features-exploration/service/scripts/feature_extraction_throughput/parsed.duckdb'

def feature_cost(features, cost_dataset):
    con = duckdb.connect(COST_DB_PATH, read_only=True)
    feature_params = ','.join(['?' for _ in features])
    cost_name_to_feature_name = {
        feature_to_tput_feature(feature): feature
        for feature in features
    }
    rows = con.execute("""
        SELECT feature_name, avg(vidtime)
        FROM experiment
        WHERE use_dali=True AND nworkers=4 AND npipe=2 AND vid_ssd=False AND feat_ssd=False
            AND feature_name IN ({feature_params})
            AND dataset_name=?
        GROUP BY ALL
    """.format(feature_params=feature_params), [*cost_name_to_feature_name.keys(), cost_dataset]).fetchall()
    cost_dict = {
        cost_name_to_feature_name[feature_name]: cost
        for feature_name, cost in rows
    }
    assert len(cost_dict) == len(features)
    return cost_dict

def main():
    core.logging.configure_logger()

    print_and_flush('Time', datetime.datetime.now())
    print_and_flush('Args:', sys.argv)
    ap = argparse.ArgumentParser()
    data_group = ap.add_argument_group('Data')
    data_group.add_argument('--db-dir', help='Base directory used by the storage manager to store metadata/features/models.')
    data_group.add_argument('--oracle-dir', help='Path to directory containing oracle databases (used for evaluation)')
    data_group.add_argument('--oracle-dump-dir', help='Path to dump of oracle directory (used to initialize metadata in experiments)')
    data_group.add_argument('--val-dir', help='Path to directory containing validation database (used for evaluation)')
    data_group.add_argument('--split-dir', help='Path to directory containing videos belonging to each split')
    data_group.add_argument('--split-type', help='Type of split (corresponds to subdirectory within split-dir)')
    data_group.add_argument('--split-idx', type=int, help='Specific split (corresponds to subdirectory within split-dir and split-type)')
    data_group.add_argument('--suffix', help='Suffix to append to database dir')
    data_group.add_argument('--cleanup', action='store_true', help='Remove db_dir before exiting')
    data_group.add_argument('--no-eval-on-test', action='store_true', help='Do not evaluate model performance')
    data_group.add_argument('--eval-on-trained', action='store_true', help='Evaluate model performance using the last model VOCALExplore used to make predictions (rather than training a new model over the labels collected so far)')

    other = ap.add_argument_group('Other')
    other.add_argument('--feature-names', nargs='+', help='List of candidate feature names')
    other.add_argument('--explorer', choices=['random', 'coreset', 'randomifuniform'], help='Acquisition function (randomifuniform corresponds to VE-sample)')
    other.add_argument('--oracle', default='exact', choices=['exact', 'context-from-center',], help='Behavior of oracle labeler; paper only evaluated with "exact"')
    other.add_argument('--cpus', type=int, help='Maximum number of tasks to execute at once')
    other.add_argument('--gpus', type=int, help='Maximum number of tasks to execute at once on the GPU (GPU tasks count towards the CPU task limit)')

    user = ap.add_argument_group('User')
    user.add_argument('--k', type=int, help='Number of video segments to return from each call to "Explore"')
    user.add_argument('--labelt', type=float, help='Duration of each video segment labeled by oracle')
    user.add_argument('--watcht', type=float, help='Simulated duration of videos that users watch when labeling')
    user.add_argument('--playback-speed', type=float, help='Simulated speed that users watch videos (so idle time is watcht / playback_speed)')
    user.add_argument('--nsteps', type=int, help='Number of "Explore" steps to execute')
    user.add_argument('--explore-label', default=None, help='Not used in experiments')
    user.add_argument('--explore-label-threshold', type=int, default=-1, help='Not used in experiments')

    mm = ap.add_argument_group('Model manager')
    mm.add_argument('--mm-device', choices=('cpu', 'cuda'), help='Device to use to train and perform inference')
    mm.add_argument('--min-labels', type=int, default=5, help='Number of labels required to start training a model')
    mm.add_argument('--serial-kfold', action='store_true', help='Whether to evaluate k-fold splits in serial or parallel')

    fm = ap.add_argument_group('Feature manager')
    fm.add_argument('--start-with-features', type=int, default=1, choices=(0, 1), help='For evaluation, whether the feature manager should be initialized with all features')

    alm = ap.add_argument_group('Active learning manager')
    alm.add_argument('--strategy', choices=[f.value for f in FeatureEvalStrategy], help='How to evaluate candidate features. Use "wait" or "risingbandit" when using one feature, "concat" to concatenate all features, or "risingbandit" to perform feature selection over multiple features')
    alm.add_argument('--no-return-predictions', action='store_true', help='Do not return predictions along with selected video clips (removes latency of inference, and additionally feature extraction for random sampling)')
    alm.add_argument('--al-vids-x', type=int, default=-1, help='For incremental active learning, minimum number of candidate videos to preprocess')

    bandit = ap.add_argument_group('Bandit')
    bandit.add_argument('--bandit-C', type=int, default=7)
    bandit.add_argument('--bandit-T', type=int, default=100)
    bandit.add_argument('--bandit-type', type=str, default='basic', choices=[b.value for b in BanditTypes], help='Type of smoothing to perform (experiments use "exp")')
    bandit.add_argument('--bandit-window', type=int, help='Smoothing window')
    bandit.add_argument('--bandit-validation-size', type=int, default=-1, help='Not used; kept for compatibility with scripts')
    bandit.add_argument('--bandit-eval', choices=('testset', 'kfold'), default='testset', help='How to evaluate candidate features (using held out evaluation set or kfold on top of labels)')
    bandit.add_argument('--bandit-eval-metric', help='Metric used to evaluate feature performance')
    bandit.add_argument('--bandit-kfold-k', type=int)
    bandit.add_argument('--bandit-cost-dataset', help='Not used')
    bandit.add_argument('--bandit-cost-weight', type=float, help='Not used')
    bandit.add_argument('--bandit-keep-all', action='store_true', help='Not used')

    opts = ap.add_argument_group('Optimizations')
    opts.add_argument('--eager-feature-extraction-labeled', action='store_true', help='Whether to schedule background feature extraction tasks for labeled videos')
    opts.add_argument('--eager-model-training', action='store_true', help='Whether to schedule background tasks to train models')
    opts.add_argument('--async-bandit', action='store_true', help='Whether to perform feature evaluation asynchronously')
    opts.add_argument('--use-priority', action='store_true', help='Whether to prioritize tasks based on priority (if not specified, tasks are executed FIFO)')
    opts.add_argument('--suspend-lowp', action='store_true', help='Whether to suspend low-priority tasks during user interactions')
    opts.add_argument('--eager-feature-extraction-unlabeled', action='store_true', help='Whether to eagerly schedule feature extraction tasks for unlabeled videos')

    args = ap.parse_args()

    UserPriority.priority = Priority.USER if args.use_priority else Priority.DEFAULT
    ChorePriority.priority = Priority.CHORE if args.use_priority else Priority.DEFAULT

    db_dir = args.db_dir + '_' + args.suffix
    status_file = None
    oracle_dir = args.oracle_dir
    feature_names = args.feature_names
    ignore_labels = set(['neutral', 'lie down', 'stand up', 'snow'])
    start_with_features = bool(args.start_with_features)
    eager_feature_extraction_unlabeled = args.eager_feature_extraction_unlabeled

    db_dir, features_dir, models_dir = setup_dirs(db_dir, oracle_dir, args.oracle_dump_dir, args.split_dir, args.split_type, args.split_idx, include_features=start_with_features, feature_names=feature_names)

    if args.cleanup:
        cleanup_fun = partial(cleanup, models_dir=models_dir, features_dir=features_dir, db_dir=db_dir)
        atexit.register(cleanup_fun)

    oracle_sm = StorageManager(db_dir=oracle_dir, features_dir=os.path.join(oracle_dir, 'features'), models_dir=None, read_only=True)
    # If we are evaluating on an already-trained model, it needs to be able to predict all of the classes in the evaluation set.
    train_labels = set(oracle_sm.get_distinct_labels()) - ignore_labels if args.eval_on_trained else None

    sm = StorageManager(db_dir=db_dir, features_dir=features_dir, models_dir=models_dir)
    vm = VideoManager(sm)
    scheduler = PriorityScheduler(cpus=args.cpus, gpus=args.gpus, suspend_lowp=args.suspend_lowp)
    fm = BackgroundAsyncFeatureManager(sm, scheduler, batch_size=8)
    mm = BackgroundAsyncModelManager(sm, fm, scheduler=scheduler, min_trainsize=args.min_labels, device=args.mm_device, train_labels=train_labels, parallel_kfold=(not args.serial_kfold))
    explorer_cls = EXPLORER_TO_CLASS(args.explorer)
    explorer_kwargs = {}
    if args.explorer == 'random':
        explorer_kwargs['limit_to_extracted'] = eager_feature_extraction_unlabeled
    elif args.explorer == 'coreset':
        explorer_kwargs['missing_vids_X'] = args.al_vids_x
    if issubclass(explorer_cls, explorers.AbstractRandomIfUniformExplorer):
        explorer_kwargs['limit_to_extracted'] = eager_feature_extraction_unlabeled
        explorer_kwargs['missing_vids_X'] = args.al_vids_x
        explorer = explorers.RandomIfUniformExplorerFromName(explorer_cls, args.explorer, **explorer_kwargs)
    else:
        explorer = explorer_cls(**explorer_kwargs)

    if args.val_dir is None:
        # When split is not None, both train and val exist in the same database.
        test_vids = get_split_vids(args.split_dir, args.split_type, args.split_idx, 'test', oracle_sm)
        eval_sm = oracle_sm
        oracle_fm = BasicFeatureManager(oracle_sm) # Other args don't matter since everything is extracted already.
        eval_fm = oracle_fm
    else:
        test_vids = None
        oracle_fm = BasicFeatureManager(oracle_sm)
        eval_sm = StorageManager(db_dir=args.val_dir, features_dir=os.path.join(args.val_dir, 'features'), models_dir=None, read_only=True)
        eval_fm = BasicFeatureManager(eval_sm)
    oracle = args.oracle
    if oracle == 'exact':
        labeler = OracleLabeler(oracle_sm)
    elif oracle == 'context-from-center':
        labeler = OracleContextLabeler(oracle_sm)
    else:
        assert False, f'Unrecognized labeler option: {oracle}'

    if not args.explore_label:
        user = ExploreUser(args.k, args.labelt, args.watcht, labeler, args.playback_speed, ignore_labels, status_file)
    else:
        user = ExploreLabelUser(args.explore_label, args.k, args.labelt, args.watcht, labeler, args.playback_speed, ignore_labels, status_file)

    strategy = FeatureEvalStrategy(args.strategy)
    if strategy == FeatureEvalStrategy.RISINGBANDIT:
        if args.bandit_eval == 'testset':
            def bandit_eval_fn(feature_names, callback):
                performance = user.evaluate_labels(sm, oracle_fm, eval_sm, eval_fm, feature_names, test_vids, ignore_labels, sample_from_validation=args.bandit_validation_size)
                callback(performance['pred_' + args.bandit_eval_metric])
        elif args.bandit_eval == 'kfold':
            assert args.bandit_kfold_k > 0, f'--bandit-kfold-k must be positive when --bandit-eval is "kfold"; {args.bandit_kfold_k}'
            def bandit_eval_fn(feature_names, callback):
                def parse_performance(performance, callback=None):
                    if performance is None:
                        logging.warning(f'Kfold returned no performance metrics for {feature_names}')
                        callback(0)
                        return

                    performance_to_print = {
                        key.replace('test_', 'pred_'): value
                        for key, value in performance.items() if key.startswith('test_')
                    }
                    logging.info(f'*** Performance {performance_to_print}')

                    callback(performance['test_' + args.bandit_eval_metric])

                mm.check_label_quality_async(feature_names, n_splits=args.bandit_kfold_k, min_size=args.min_labels, f1_val=0.2, callback=partial(parse_performance, callback=callback))

        else:
            assert False, f'Unhandled bandit eval strategy {args.bandit_eval}'
        strategy_kwargs = dict(
            bandit_type=BanditTypes(args.bandit_type),
            candidate_feature_names=feature_names,
            C=args.bandit_C,
            T=args.bandit_T,
            eval_candidate=bandit_eval_fn,
            consider_all=args.bandit_keep_all,
            async_step=args.async_bandit,
        )
        if args.bandit_window:
            strategy_kwargs['window'] = args.bandit_window
        if args.bandit_cost_dataset:
            assert args.bandit_cost_weight, 'Both bandit-cost-dict and bandit-cost-weight must be specified'
            strategy_kwargs['cost_dict'] = feature_cost(feature_names, args.bandit_cost_dataset)
            strategy_kwargs['cost_weight'] = args.bandit_cost_weight
    else:
        strategy_kwargs = dict()

    return_predictions = not args.no_return_predictions
    alm = MultiFeatureActiveLearningManager(fm, mm, vm, explorer, feature_names, scheduler, strategy=FeatureEvalStrategy(args.strategy), strategy_kwargs=strategy_kwargs, eager_feature_extraction_labeled=args.eager_feature_extraction_labeled, eager_model_training=args.eager_model_training, eager_feature_extraction_unlabeled=eager_feature_extraction_unlabeled, explore_label_threshold=args.explore_label_threshold, return_predictions=return_predictions)

    nsteps = args.nsteps
    core.timing.TIME_LOGGER.info(f'Workload started at {datetime.datetime.now()}')

    start = time.perf_counter()
    start_process = time.process_time()

    training_time = 0
    training_process_time = 0

    for i in range(nsteps):
        print_and_flush('*** Step', i)
        feature_names = user.perform_action(alm)

        if not args.no_eval_on_test:
            scheduler.suspend_all()
            print_and_flush(f'Step {i} evaluating on feature {feature_names}')
            start_train = time.perf_counter()
            start_train_process = time.process_time()
            user.evaluate_labels(sm, oracle_fm, eval_sm, eval_fm, feature_names, test_vids, ignore_labels, eval_on_trained=args.eval_on_trained, trained_mid=mm._feature_to_last_prediction_mid.get(core.typecheck.ensure_str(feature_names), None))
            training_time += time.perf_counter() - start_train
            training_process_time += time.process_time() - start_train_process
            scheduler.resume_all()

    end = time.perf_counter()
    end_process = time.process_time()

    core.timing.TIME_LOGGER.info(f'Workload took {end - start} seconds ({end - start - training_time} minus training time)')
    core.timing.TIME_LOGGER.info(f'Workload took {end_process - start_process} CPU seconds, ({end_process - start_process - training_process_time} minus training process time)')
    core.timing.TIME_LOGGER.info(f'Workload ended at {datetime.datetime.now()}')

if __name__ == '__main__':
    main()
