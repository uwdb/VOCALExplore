#!/bin/bash

cd //gscratch/balazinska/mdaum/video-features-exploration/datasets/temporal-deer

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create-splits.py \
        --split-type interspersed \
        --db-path /gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle/annotations.duckdb \
        --output-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle-splits \
        --output-type vpath \
        --output-file csv
