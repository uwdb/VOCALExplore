# VOCALExplore

This repository contains the code for VOCALExplore.
The code for the experiments in the [technical report](https://arxiv.org/abs/2303.04068) is on the `experiments` branch.
The `main` branch extends the experimental implementation to serve as the backend for an exploration interface.

## Demonstration

[
  <picture>
    <img src="https://github.com/uwdb/VOCALExplore/assets/44246059/531206d3-9a89-4449-939a-242f167d9d27" width="75%">
  </picture>
](https://db.cs.washington.edu/projects/visualworld)

_Clicking on the preview image will navigate to a demonstration video on the UWDB website._

## Project structure
`vfe/` contains the implementation of VOCALExplore. `vfe/api` houses the various managers and task scheduler, and the rest of the subdirectories implement utilities.

# Docker
Build the dockerfile:
```
docker build -t vocalexplore/vocalexplore -f docker/Dockerfile .
```

Copy `.env` and modify any ports or paths. Pass this env file to `start_docker` below.
**Make sure that the thumbnail directory matches the directory that will be specified in the server's configuration yaml file.**
- `SERVER_PORT`: The port that the server will be available on.
- `THUMBNAIL_DIR`: The directory where thumbnails will stored.
- `THUMBNAIL_PORT`: The port where thumbnails can be accessed.
- `VIDEO_DIR`: The directory where videos are stored.
- `VIDEO_PORT`: The port where videos can be accessed.

When running the frontend, copy the contents of `.env` into `config.py`.

Start the docker container in the background:
```
./start_docker.sh

Args:
  -e <path to env file>, or default .env
  -n <docker container name>, or default vocalexplore
```

# Server
The server can be started by running
```
python server/server.py

Arguments:
  -c/--config-file <path to config file>
```

Sample config files can be found in `server/configs`.
The `db_dir` is the directory where VOCALExplore's storage manager will store its data.
If `thumbnail_dir` is specified, VOCALExplore will extract thumbnail images from loaded videos and store them in this directory.
