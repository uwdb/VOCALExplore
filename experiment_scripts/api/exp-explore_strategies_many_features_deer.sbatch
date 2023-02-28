#!/bin/bash
#SBATCH --job-name=deer-manyfeat
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=61G
#SBATCH --time=04:59:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

featurenames=""
while getopts "e:b:i:l:n:w:s:v:f:d:" opt; do
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
    esac
done

if [ -z "$banditvalidationflag" ]; then
    banditvalidationsize=-1
fi

if [ -z "$outputdirflag" ]; then
    logdir="many_features_deer"
fi

remainder=${@:OPTIND}

echo $explorer, $suffixid, $splitidx, $labelt, $nsteps, $startwithfeatures, $strategy, $banditvalidationflag, $banditvalidationsize, $featurenames, $remainder

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

dbdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/fresh"
oracledir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle"
oracledumpdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle-dump"
splitdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle-splits"
splittype="interspersed"
k=5
watcht=30
playbackspeed=1000000
mmdevice="cpu"
oracle="exact"
cpus=8
gpus=0

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
