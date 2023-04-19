#!/bin/bash

while getopts "e:n:" opt; do
    case "$opt" in
        e) ENV_FILE=$OPTARG ;;
        n) DOCKER_NAME=$OPTARG ;;
    esac
done

if [ -z "$ENV_FILE" ]; then
    ENV_FILE=".env"
fi

if [ -z "$DOCKER_NAME" ]; then
    DOCKER_NAME="vocalexplore"
fi

# Set env vars from env vile so we can access data dir.
export $(cat $ENV_FILE | xargs)

# Specify env vars to fix issue with "failed to load libnvcuvid.so"
# https://github.com/NVIDIA/nvidia-docker/wiki/Usage
docker run --rm -itd \
    --runtime=nvidia \
    --name $DOCKER_NAME \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v $VIDEO_DIR:$VIDEO_DIR \
    -p $SERVER_PORT:$SERVER_PORT \
    -p $THUMBNAIL_PORT:$THUMBNAIL_PORT \
    -p $VIDEO_PORT:$VIDEO_PORT \
    -e "NVIDIA_VISIBLE_DEVICES=0" \
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility" \
    --env-file $ENV_FILE \
    vocalexplore/vocalexplore:latest \
    ./debug_install.sh
