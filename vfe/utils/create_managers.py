from functools import partial
import logging
import os
import shutil
import yaml

from vfe import core
from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
from vfe.api.videomanager.basic import BasicVideoManager as VideoManager
from vfe.api.featuremanager.background_directed import BackgroundAsyncFeatureManager
from vfe.api.modelmanager.background_directed import BackgroundAsyncModelManager
from vfe.api.activelearningmanager.multifeature import MultiFeatureActiveLearningManager, FeatureEvalStrategy
from vfe.api.activelearningmanager import explorers
from vfe.api.activelearningmanager.risingbandit import BanditTypes
from vfe.api.scheduler.priority import PriorityScheduler


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


def initialize_environment(db_dir, delete_if_exists=False):
    if delete_if_exists and os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    paths = (
        db_dir,
        os.path.join(db_dir, 'features'),
        os.path.join(db_dir, 'models'),
    )
    for path in paths:
        core.filesystem.create_dir(path)
    return paths


def load_options(config_path):
    return yaml.safe_load(open(config_path))


def get_alm(config_path) -> MultiFeatureActiveLearningManager:
    options = load_options(config_path)
    ve_options = options['vocalexplore']

    has_db_dir = ve_options['db_dir']
    has_thumbnail_dir = ve_options['thumbnail_dir']
    if not has_db_dir or not has_thumbnail_dir:
        raise RuntimeError(f'Both db_dir and thumbnail_dir must be specified in configuration file ({config_path})')

    # Set up base objects.
    db_dir, features_dir, models_dir = initialize_environment(ve_options['db_dir'])

    thumbnail_dir = ve_options.get('thumbnail_dir', None)
    if thumbnail_dir:
        core.filesystem.create_dir(thumbnail_dir)

    sm = StorageManager(db_dir=db_dir, features_dir=features_dir, models_dir=models_dir, full_overlap=ve_options.get('label_fully_overlaps_feature', True))
    vm = VideoManager(sm)
    scheduler = PriorityScheduler(cpus=ve_options['cpus'], gpus=ve_options['gpus'], suspend_lowp=ve_options['suspend_lowp'])
    fm = BackgroundAsyncFeatureManager(sm, scheduler, batch_size=8, async_batch_size=ve_options.get('async_batch_size', -1))
    mm = BackgroundAsyncModelManager(sm, fm, scheduler=scheduler, min_trainsize=ve_options['min_labels'], device=ve_options['mm_device'], parallel_kfold=(not ve_options['serial_kfold']))

    # Set up acquisition function selection.
    eager_feature_extraction_unlabeled = ve_options['eager_feature_extraction_unlabeled']
    al_vids_X = ve_options['al_vids_x']

    explorer = ve_options['explorer']
    explorer_cls = EXPLORER_TO_CLASS(explorer)
    explorer_kwargs = {}
    if explorer == 'random':
        explorer_kwargs['limit_to_extracted'] = eager_feature_extraction_unlabeled
    elif explorer == 'coreset':
        explorer_kwargs['missing_vids_X'] = al_vids_X

    if issubclass(explorer_cls, explorers.AbstractRandomIfUniformExplorer):
        explorer_kwargs['limit_to_extracted'] = eager_feature_extraction_unlabeled
        explorer_kwargs['missing_vids_X'] = al_vids_X
        explorer = explorers.RandomIfUniformExplorerFromName(explorer_cls, explorer, **explorer_kwargs)
    else:
        explorer = explorer_cls(**explorer_kwargs)

    # Set up feature extractor selection.
    bandit_eval_metric = ve_options['bandit_eval_metric']
    bandit_kfold_k = ve_options['bandit_kfold_k']
    min_labels = ve_options['min_labels']
    f1_val = ve_options['fraction_f1_val']
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

            callback(performance['test_' + bandit_eval_metric])

        mm.check_label_quality_async(feature_names, n_splits=bandit_kfold_k, min_size=min_labels, f1_val=f1_val, callback=partial(parse_performance, callback=callback))

    feature_names = ve_options['feature_names']
    strategy_kwargs = dict(
        bandit_type=BanditTypes(ve_options['bandit_type']),
        candidate_feature_names=feature_names,
        C=ve_options['bandit_C'],
        T=ve_options['bandit_T'],
        window=ve_options['bandit_window'],
        eval_candidate=bandit_eval_fn,
        consider_all=False,
        async_step=ve_options['async_bandit'],
    )

    eager_feature_extraction_batch_size = (
        ve_options.get('eager_feature_extraction_unlabeled_batch_size', 10)
        if eager_feature_extraction_unlabeled
        else 0
    )

    alm = MultiFeatureActiveLearningManager(
        fm, mm, vm, explorer, feature_names, scheduler,
        strategy=FeatureEvalStrategy.RISINGBANDIT,
        strategy_kwargs=strategy_kwargs,
        eager_feature_extraction_labeled=ve_options['eager_feature_extraction_labeled'],
        eager_model_training=ve_options['eager_model_training'],
        eager_feature_extraction_unlabeled=eager_feature_extraction_batch_size,
        thumbnail_dir=thumbnail_dir,
    )
    return alm
