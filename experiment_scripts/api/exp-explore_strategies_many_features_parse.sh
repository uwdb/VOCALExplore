#!/bin/bash

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts/process

logdir=$1

python parse_workload_logs_csv.py --log-dir ../$logdir/logs --db-path ../$logdir/parsed.duckb
