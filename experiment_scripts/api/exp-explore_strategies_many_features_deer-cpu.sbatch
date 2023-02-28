#!/bin/bash
#SBATCH --job-name=deer-manyfeat
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
while getopts "e:b:i:l:n:w:s:v:f:d:c:g:p:a" opt; do
    case "$opt" in
        e) explorer=$OPTARG ;;
        b) suffixid=$OPTARG ;;
        i) splitidx=$OPTARG ;;
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
    logdir="many_features_deer"
fi

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

remainder=${@:OPTIND}

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

dbdir="$basedir/mdaum/video-features-exploration/service/storage/deer/fresh"
oracledir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle"
oracledumpdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle-dump"
splitdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle-splits"
splittype="interspersed"
k=5
watcht=30
mmdevice="cpu"
oracle="exact"

suffix=$(date +"%Y%m%d-%H%M%S")
suffix="${explorer}_${splittype}${splitidx}_labelt${labelt}_${suffix}_${suffixid}"
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
        --split-dir $splitdir \
        --split-type $splittype \
        --split-idx $splitidx \
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
