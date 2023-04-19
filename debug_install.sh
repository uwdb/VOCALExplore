#!/bin/bash

python3 setup.py develop

./start_thumbnail_server.sh &
./start_video_server.sh &

/bin/bash
