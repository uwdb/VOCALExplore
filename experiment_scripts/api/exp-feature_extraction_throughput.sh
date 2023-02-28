#!/bin/bash

# sbatch exp-feature_extraction_throughput.sbatch deer clip_vitb32_embed_20fstride

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train r3d_18_ap_mean_flatten

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train clip_vitb32_embed_30fstride

# sbatch exp-feature_extraction_throughput.sbatch deer r3d_18_ap_mean_flatten

# sbatch exp-feature_extraction_throughput.sbatch deer clip_vitb32_embed_16fstride

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train clip_vitb32_embed_16fstride

# sbatch exp-feature_extraction_throughput.sbatch deer mvit_v1_b_getitem_1

# sbatch exp-feature_extraction_throughput.sbatch deer mvit_v1_b_head.0

# sbatch exp-feature_extraction_throughput.sbatch deer mvit_v1_b_head.1

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train mvit_v1_b_getitem_1

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train mvit_v1_b_head.0

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train mvit_v1_b_head.1

# sbatch exp-feature_extraction_throughput.sbatch deer mvit_v1_b_16x2_getitem_1

# sbatch exp-feature_extraction_throughput.sbatch deer mvit_v1_b_16x2_stride32_getitem_1

# sbatch exp-feature_extraction_throughput.sbatch deer clip_vitb32_embed_32fstride

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train clip_vitb32_embed_32fstride

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train mvit_v1_b_16x2_getitem_1

# sbatch exp-feature_extraction_throughput.sbatch kinetics7m4-train mvit_v1_b_16x2_stride32_getitem_1

for i in {1..5}
do
    for dataset in "deer" "kinetics7m4-train"
    do
        for feature in "clip_vitb32_embed_16x2maxpool" # "r3d_18_ap_mean_stride32_flatten" "clip_vitb32_embed_32fstride" "mvit_v1_b_16x2_stride32_getitem_1" "mvit_v1_b_16x2_random_stride32_getitem_1" "clip_vitb32_embed_2fstride"
        do
            sbatch exp-feature_extraction_throughput.sbatch $dataset $feature
        done
    done
done
