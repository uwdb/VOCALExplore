#!/bin/bash
#SBATCH --job-name=k400-val
#SBATCH --partition=gpu-a40
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=61G
#SBATCH --time=20:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

featurename="mvit_v1_b_head.1"

outfile="$(pwd)/output/temporal/create-oracle-sm-kinetics400-val.txt"

cat $0 >> $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split val \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics400 \
        --device cuda \
        --dataset kinetics400 \
        --feature-name $featurename \
    &>> $outfile

# Do a second time to get any videos that are too short.
bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split val \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics400 \
        --device cuda \
        --dataset kinetics400 \
        --feature-name $featurename \
    &>> $outfile
