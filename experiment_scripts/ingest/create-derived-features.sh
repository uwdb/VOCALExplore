#!/bin/bash

dataset=$1
# Ex: deer/oracle, k7m4-10-train/oracle, k7m4-10-val

cd /gscratch/balazinska/mdaum/video-features-exploration/service/scripts

python derive_sequential_features.py \
    --base-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/r3d_18_ap_mean_flatten \
    --target-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/r3d_18_ap_mean_stride32_flatten

for layer in "getitem_1" "head.0" "head.1"
do
    python derive_sequential_features.py \
        --base-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/mvit_v1_b_16x2_${layer} \
        --target-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/mvit_v1_b_16x2_stride32_${layer}
done

python derive_sequential_features.py \
    --base-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/clip_vitb32_embed_16fstride \
    --target-dir /gscratch/balazinska/mdaum/video-features-exploration/service/storage/${dataset}/features/clip_vitb32_embed_32fstride
