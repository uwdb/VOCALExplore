#!/bin/bash

labelt=1
nsteps=100

startwithfeatures=1
strategy="wait"
logdir="-d many_features_k7m4"

explorer="randomifuniform"
for featurenames in "-f clip_vitb32_embed_32fstride" "-f r3d_18_ap_mean_stride32_flatten" "-f mvit_v1_b_16x2_stride32_getitem_1" "-f mvit_v1_b_16x2_random_stride32_getitem_1" "-f clip_vitb32_embed_16x2maxpool"
do
    for nclasses in "20"
    do
        for i in {1..10}
        do
            sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i -1 -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir
        done
        for splitidx in {0..9}
        do
            sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i $splitidx -l $labelt -n $nsteps -t "zipf-N${nclasses}-s2.0" -w $startwithfeatures -s $strategy $featurenames $logdir
        done
    done
done

# Also evaluate on random and coresets.
for featurenames in "-f clip_vitb32_embed_32fstride" "-f r3d_18_ap_mean_stride32_flatten" "-f mvit_v1_b_16x2_random_stride32_getitem_1" "-f clip_vitb32_embed_16x2maxpool" "-f mvit_v1_b_16x2_stride32_getitem_1"
do
    for explorer in "coreset" "random"
    do
        for nclasses in "20"
        do
            for i in {1..10}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir
            done
            for splitidx in {0..9}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i $splitidx -l $labelt -n $nsteps -t "zipf-N${nclasses}-s2.0" -w $startwithfeatures -s $strategy $featurenames $logdir
            done
        done
    done
done


# Evaluate what happens when we concatenate all of the features.
# Make sure that clip_vitb32_embed_32fstride isn't first because it messes up the alignment.
strategy="concat"
featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
for explorer in "random" "randomifuniform"
do
    for nclasses in "20"
    do
        for i in {1..10}
        do
            sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i -1 -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir
        done
        for splitidx in {0..9}
        do
            sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b ${featurenames:21:1}${featurenames: -1}${i} -i $splitidx -l $labelt -n $nsteps -t "zipf-N${nclasses}-s2.0" -w $startwithfeatures -s $strategy $featurenames $logdir
        done
    done
done
