#!/bin/bash

DATA_DIR="/home/maureen/video_exploration/data"

# Specify env vars to fix issue with "failed to load libnvcuvid.so"
# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
docker run --rm -itd \
    --runtime=nvidia \
    --name vocalexplore \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v $DATA_DIR:$DATA_DIR \
    -p 8890:8890 \
    -e "NVIDIA_VISIBLE_DEVICES=0" \
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility" \
    vocalexplore/vocalexplore:latest \
    ./debug_install.sh
