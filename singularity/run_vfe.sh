#!/bin/bash

module load apptainer
# cp vfe.sif /tmp/vfe.sif
apptainer exec --nv --bind /gscratch --bind /data --bind /home vfe.sif bash setup_vfe_install.sh "$@"
