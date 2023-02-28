#!/bin/bash
#SBATCH --job-name=k400-train
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=61G
#SBATCH --time=12:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

# vidsplit=$1

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

outfile="$(pwd)/output/temporal/create-oracle-sm-kinetics400-train-${vidsplit}.txt"

featurename="mvit_v1_b_head.1"

cat $0 >> $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split train \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics400 \
        --device cuda \
        --dataset kinetics400 \
        --feature-name $featurename \
    &>> $outfile

# Do a second time to get any videos too short.
bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split train \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics400 \
        --device cuda \
        --dataset kinetics400 \
        --feature-name $featurename \
    &>> $outfile
