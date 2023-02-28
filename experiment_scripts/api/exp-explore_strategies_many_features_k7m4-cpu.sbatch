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
while getopts "k:e:b:i:t:l:n:w:s:v:f:d:c:g:p:a" opt; do
    case "$opt" in
        k) nclasses=$OPTARG ;;
        e) explorer=$OPTARG ;;
        b) suffixid=$OPTARG ;;
        i) splitidx=$OPTARG ;;
        t) splittype=$OPTARG ;;
        l) labelt=$OPTARG ;;
        n) nsteps=$OPTARG ;;
        w) startwithfeatures=$OPTARG ;;
        s) strategy=$OPTARG ;;
        v)  banditvalidationflag=1
            banditvalidationsize=$OPTARG ;;
        f) featurenames="$featurenames $OPTARG" ;;
        d)  outputdirflag=1
            logdir=$OPTARG ;;
        c)  cpusflag=1
            cpus=$OPTARG ;;
        g)  gpusflag=1
            gpus=$OPTARG ;;
        p)  playbackspeedflag=1
            playbackspeed=$OPTARG ;;
        a)  ssdflag=1
            basedir="/scr"

    esac
done

if [ -z "$banditvalidationflag" ]; then
    banditvalidationsize=-1
fi

if [ -z "$outputdirflag" ]; then
    logdir="many_features_k7m4"
fi
logdir="${logdir}-${nclasses}"

if [ -z "$cpusflag" ]; then
    cpus=8
fi

if [ -z "$gpusflag" ]; then
    gpus=1
fi

if [ -z "$playbackspeedflag" ]; then
    playbackspeed=1000000
fi

if [ -z "$ssdflag" ]; then
    basedir="/gscratch/balazinska"
fi

splitdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/oracle-splits"
if [ -z "$splittype" ]; then
    splitflags=""
else
    splitflags="--split-dir $splitdir --split-type $splittype --split-idx $splitidx"
fi


remainder=${@:OPTIND}

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

dbdir="$basedir/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/fresh"
oracledir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/oracle"
oracledumpdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-train/oracle-dump"
valdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/k7m4-${nclasses}-val"
# featurenames="${@:7}"
k=5
# labelt=$4
watcht=30
# playbackspeed=1000000
# nsteps=$5
mmdevice="cpu"
oracle="exact"
# cpus=8
# gpus=1
# strategy="wait"
# startwithfeatures=$6

# explorer=$1
# suffixid=$2
suffix=$(date +"%Y%m%d-%H%M%S")
suffix="${explorer}_${splittype}${splitidx}_labelt${labelt}_${suffix}_${suffixid}"
# logdir="many_features_k7m4"
mkdir -p "${logdir}/logs"
outfile="$(pwd)/${logdir}/logs/log_${suffix}.txt"
echo "${suffix}, ${outfile}"

cat $0 > $outfile

echo "$(nvidia-smi)" >> $outfile

echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python $(pwd)/exp-explore_strategies_many_features.py \
        --db-dir $dbdir \
        --oracle-dir $oracledir \
        --oracle-dump-dir $oracledumpdir \
        --val-dir $valdir \
        $splitflags \
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
        --bandit-validation-size $banditvalidationsize \
        $remainder \
    &>> $outfile
