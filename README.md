# VOCALExplore

This repository contains the code for VOCALExplore.

## Project structure
`vfe/` contains the implementation of VOCALExplore. `vfe/api` houses the various managers and task scheduler, and the rest of the subdirectories implement utilities.

`experiment_scripts/api` contains the scripts used to produce the results in the paper
* Specifically, `exp-explore_strategies_many_features_<>.sh`, `exp-explore_strategies_risingbandit_<>.sh`, and `exp-graph1-<>.sh`

## Setup
A singularity container definition with setup processes is located at `singularity/vfe.def`.

[TODO] Add an equivalent Dockerfile to aid in reproducibility.

## Setting up ground truth datasets for experiment
[TODO]

## Example usage
The primary way to exercise VOCALExplore that is used by all of the experiments is `service/scripts/exp-explore_strategies_many_features.py`.

```
Data:
  --db-dir DB_DIR       Base directory used by the storage manager to store metadata/features/models.
  --oracle-dir ORACLE_DIR
                        Path to directory containing oracle databases (used for evaluation)
  --oracle-dump-dir ORACLE_DUMP_DIR
                        Path to dump of oracle directory (used to initialize metadata in experiments)
  --val-dir VAL_DIR     Path to directory containing validation database (used for evaluation)
  --split-dir SPLIT_DIR
                        Path to directory containing videos belonging to each split
  --split-type SPLIT_TYPE
                        Type of split (corresponds to subdirectory within split-dir)
  --split-idx SPLIT_IDX
                        Specific split (corresponds to subdirectory within split-dir and split-type)
  --suffix SUFFIX       Suffix to append to database dir
  --cleanup             Remove db_dir before exiting
  --no-eval-on-test     Do not evaluate model performance
  --eval-on-trained     Evaluate model performance using the last model VOCALExplore used to make predictions (rather than training a new model over the labels collected so far)

Other:
  --feature-names FEATURE_NAMES [FEATURE_NAMES ...]
                        List of candidate feature names
  --explorer {random,coreset,randomifuniform}
                        Acquisition function (randomifuniform corresponds to VE-sample)
  --oracle {exact,context-from-center}
                        Behavior of oracle labeler; paper only evaluated with "exact"
  --cpus CPUS           Maximum number of tasks to execute at once
  --gpus GPUS           Maximum number of tasks to execute at once on the GPU (GPU tasks count towards the CPU task limit)

User:
  --k K                 Number of video segments to return from each call to "Explore"
  --labelt LABELT       Duration of each video segment labeled by oracle
  --watcht WATCHT       Simulated duration of videos that users watch when labeling
  --playback-speed PLAYBACK_SPEED
                        Simulated speed that users watch videos (so idle time is watcht / playback_speed)
  --nsteps NSTEPS       Number of "Explore" steps to execute
  --explore-label EXPLORE_LABEL
                        Not used in experiments
  --explore-label-threshold EXPLORE_LABEL_THRESHOLD
                        Not used in experiments

Model manager:
  --mm-device {cpu,cuda}
                        Device to use to train and perform inference
  --min-labels MIN_LABELS
                        Number of labels required to start training a model
  --serial-kfold        Whether to evaluate k-fold splits in serial or parallel

Feature manager:
  --start-with-features {0,1}
                        For evaluation, whether the feature manager should be initialized with all features

Active learning manager:
  --strategy {wait,concat,risingbandit}
                        How to evaluate candidate features. Use "wait" or "risingbandit" when using one feature, "concat" to concatenate all features, or "risingbandit" to perform feature selection over multiple features
  --no-return-predictions
                        Do not return predictions along with selected video clips (removes latency of inference, and additionally feature extraction for random sampling)
  --al-vids-x AL_VIDS_X
                        For incremental active learning, minimum number of candidate videos to preprocess

Bandit:
  --bandit-C BANDIT_C
  --bandit-T BANDIT_T
  --bandit-type {basic,moving,exp,weighted}
                        Type of smoothing to perform (experiments use "exp")
  --bandit-window BANDIT_WINDOW
                        Smoothing window
  --bandit-validation-size BANDIT_VALIDATION_SIZE
                        Not used; kept for compatibility with scripts
  --bandit-eval {testset,kfold}
                        How to evaluate candidate features (using held out evaluation set or kfold on top of labels)
  --bandit-eval-metric BANDIT_EVAL_METRIC
                        Metric used to evaluate feature performance
  --bandit-kfold-k BANDIT_KFOLD_K
  --bandit-cost-dataset BANDIT_COST_DATASET
                        Not used
  --bandit-cost-weight BANDIT_COST_WEIGHT
                        Not used
  --bandit-keep-all     Not used

Optimizations:
  --eager-feature-extraction-labeled
                        Whether to schedule background feature extraction tasks for labeled videos
  --eager-model-training
                        Whether to schedule background tasks to train models
  --async-bandit        Whether to perform feature evaluation asynchronously
  --use-priority        Whether to prioritize tasks based on priority (if not specified, tasks are executed FIFO)
  --suspend-lowp        Whether to suspend low-priority tasks during user interactions
  --eager-feature-extraction-unlabeled
                        Whether to eagerly schedule feature extraction tasks for unlabeled videos
```
