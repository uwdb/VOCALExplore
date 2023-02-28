#!/bin/bash

nsteps=100
labelt=1
startwithfeatures=1
strategy="wait"
banditvalidationsize=-1
logdir="-d many_features_deer"

for splitidx in {0..4}
do
    for explorer in "randomifuniform" "random" "coreset"
    do
        for i in {1..5}
        do
            for featurenames in "-f clip_vitb32_embed_32fstride" "-f r3d_18_ap_mean_stride32_flatten" "-f mvit_v1_b_16x2_stride32_getitem_1" "-f mvit_v1_b_16x2_random_stride32_getitem_1" "-f clip_vitb32_embed_16x2maxpool"
            do
                echo $splitidx, $i, $labelt, $nsteps
                sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $banditvalidationsize $featurenames $logdir
            done
        done
    done
done


# Evaluate what happens when we concatenate all of the features.
# Make sure that clip_vitb32_embed_32fstride isn't first because it messes up the alignment.
featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
for explorer in "random" "randomifuniform"
do
    for i in {1..5}
    do
        for splitidx in {0..4}
        do
            echo $splitidx, $i, $labelt, $nsteps
            sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i}concat -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s concat -v $banditvalidationsize $featurenames $logdir
        done
    done
done
