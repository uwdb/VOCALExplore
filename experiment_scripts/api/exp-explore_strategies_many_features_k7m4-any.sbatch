#!/bin/bash
#SBATCH --job-name=k7m4-manyfeat
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

featurenames=""
while getopts "e:b:l:n:w:s:f:c:d:" opt; do
    case "$opt" in
        e) explorer=$OPTARG ;;
        b) suffixid=$OPTARG ;;
        l) labelt=$OPTARG ;;
        n) nsteps=$OPTARG ;;
        w) startwithfeatures=$OPTARG ;;
        s) strategy=$OPTARG ;;
        f) featurenames="$featurenames $OPTARG" ;;
        c) nclasses=$OPTARG ;;
        d)  outputdirflag=1
            logdir=$OPTARG ;;
    esac
done

if [ -z "$outputdirflag" ]; then
    logdir="many_features_k7m4-${nclasses}"
else
    logdir="${logdir}-${nclasses}"
fi

remainder=${@:OPTIND}

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

# nclasses=$8

dbdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/fresh"
oracledir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/oracle"
oracledumpdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/oracle-dump"
valdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-val"
# featurenames="${@:9}"
k=5
# labelt=$4
watcht=30
playbackspeed=1000000
# nsteps=$5
mmdevice="cpu"
oracle="exact"
cpus=8
gpus=0
# strategy=$7
# startwithfeatures=$6

# explorer=$1
# suffixid=$2
suffix=$(date +"%Y%m%d-%H%M%S")
suffix="${explorer}_${splittype}${splitidx}_labelt${labelt}_${suffix}_${suffixid}"
# logdir="many_features_k7m4-${nclasses}"
mkdir -p "${logdir}/logs"
outfile="$(pwd)/${logdir}/logs/log_${suffix}.txt"
echo "${suffix}, ${outfile}"

cat $0 > $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python $(pwd)/exp-explore_strategies_many_features.py \
        --db-dir $dbdir \
        --oracle-dir $oracledir \
        --oracle-dump-dir $oracledumpdir \
        --val-dir $valdir \
        --suffix $suffix \
        --feature-names $featurenames \
        --explorer $explorer \
        --k $k \
        --labelt $labelt \
        --watcht $watcht \
        --playback-speed $playbackspeed \
        --nsteps $nsteps \
        --mm-device $mmdevice \
        --oracle $oracle \
        --strategy $strategy \
        --cpus $cpus \
        --gpus $gpus \
        --start-with-features $startwithfeatures \
        --cleanup \
        $remainder \
    &>> $outfile
