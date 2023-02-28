#!/bin/bash
#SBATCH --job-name=create-oracle-sm
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=61G
#SBATCH --time=11:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdaum@uw.edu
#SBATCH --output="slurm-%j-%x.out"

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

featurename="clip_vitb32_embed_dali_16x2maxpool"
# featurename="clip_vitb32_embed_2fstride"
# featurename="mvit_v1_b_16x2_random_stride32"
# featurename="r3d_18_ap_mean_flatten"

outfile="$(pwd)/output/temporal/create-oracle-sm.txt"

cat $0 >> $outfile
echo "Date" $(date) >> $outfile
echo "--------------------" >> $outfile

bash /gscratch/balazinska/mdaum/video-features-exploration/scripts/singularity-script.sh \
    python \
        $(pwd)/create_oracle_storagemanager.py \
        --oracle-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle \
        --num-workers 4 \
        --feature-name $featurename \
    &>> $outfile
