#!/bin/bash
# We can't install it to the site-packages directory because we don't have write access after the container is built.
# Work around by installing to a local directory that is on the python path.

if [[ -d /gscratch/balazinska/mdaum/video-features-exploration/singularity/vfe-install ]]; then
    echo "Skipping vfe python package install"
    export PYTHONPATH=$PYTHONPATH:/gscratch/balazinska/mdaum/video-features-exploration/singularity/vfe-install
    cd /gscratch/balazinska/mdaum/video-features-exploration
else
    mkdir /gscratch/balazinska/mdaum/video-features-exploration/singularity/vfe-install
    export PYTHONPATH=$PYTHONPATH:/gscratch/balazinska/mdaum/video-features-exploration/singularity/vfe-install
    cd /gscratch/balazinska/mdaum/video-features-exploration
    python setup.py develop --install-dir=singularity/vfe-install/
fi

if [[ $# -eq 0 ]]; then
    /bin/bash
else
    $@
fi
