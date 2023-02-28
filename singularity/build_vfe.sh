#!/bin/bash

module load apptainer
echo "Building container"
apptainer build /tmp/vfe.sif ./vfe.def
echo "Copying container"
mv /tmp/vfe.sif ./
