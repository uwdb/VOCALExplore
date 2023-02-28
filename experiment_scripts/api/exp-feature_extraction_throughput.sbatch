#!/bin/bash
#SBATCH --job-name=feats
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=61G
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

dataset=$1
featurename=$2
dbdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/testfeat/fresh-${dataset}-${featurename}"
oracledir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/oracle"
oracledumpdir="/gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/oracle-dump"

suffix=$(date +"%Y%m%d-%H%M%S")
mkdir -p "$(pwd)/feature_extraction_throughput/logs"

outfile="$(pwd)/feature_extraction_throughput/logs/log-${dataset}-${featurename}_${suffix}.txt"
echo "${suffix}, ${outfile}"

cat $0 > $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python $(pwd)/exp-feature_extraction_throughput.py \
        --dataset $dataset \
        --feature-name $featurename \
        --db-dir "${dbdir}_${suffix}" \
        --oracle-dir $oracledir \
        --oracle-dump-dir $oracledumpdir \
        &>> $outfile
