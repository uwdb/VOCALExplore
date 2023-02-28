#!/bin/bash
#SBATCH --job-name=k7m4-train
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=61G
#SBATCH --time=15:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

vidsplit=$1

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

# featurename="clip_vitb32_embed_30fstride"
# featurename="clip_vitb32_embed_16fstride"
# featurename="clip_vitb32_embed_2fstride"
featurename="clip_vitb32_embed_dali_16x2maxpool"

outfile="$(pwd)/output/temporal/create-oracle-sm-k7m4-train-${featurename}-${vidsplit}.txt"

cat $0 >> $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split train \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics7m4 \
        --device cuda \
        --feature-name $featurename \
        --checkpoint 20 \
    &>> $outfile
        # --vid-split $vidsplit \


# Again to catch too-short videos.
bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager_kinetics.py \
        --split train \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics7m4 \
        --device cuda \
        --feature-name $featurename \
        --checkpoint 20 \
    &>> $outfile
        # --vid-split $vidsplit \

