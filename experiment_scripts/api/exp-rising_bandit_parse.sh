#!/bin/bash

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts/process

logdir=$1

python parse_rising_bandit.py --log-dir ../$logdir/logs

python compute_feature_elimination_stats.py --json-path ../$logdir/json --db-path ../$logdir/bandit.duckdb
