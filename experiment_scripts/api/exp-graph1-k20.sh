#!/bin/bash

# nclasses=$1

# if [ -z $nclasses ]; then
#     echo "Error: nclasses must be specified"
#     exit 1
# fi

nclasses=20

labelt=1
strategy="risingbandit"
nsteps=100
banditmetric="ml_f1_score_macro"
logdir="-d quality_vs_latency_k20"
validationsize=-1
bandittype="exp"
C=5
T=50
w=5
banditeval="kfold"
banditkfold=3
# Store data on local ssd instead of network file system.
ssdflag="-a"
remaindereval="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-window $w --bandit-eval $banditeval --bandit-kfold-k $banditkfold --bandit-eval-metric $banditmetric"

splitidx=0
for splittype in "-t zipf-N${nclasses}-s2.0"  ""
do
    condition="randomlazy"
    startwithfeatures=0
    explorer="random"
    playbackspeed=10000
    cpus=2
    gpus=2
    for splitidx in {8..9}
    do
        for featurename in "-f r3d_18_ap_mean_stride32_flatten" "-f clip_vitb32_embed_32fstride" "-f mvit_v1_b_16x2_random_stride32_getitem_1" "-f clip_vitb32_embed_16x2maxpool" "-f mvit_v1_b_16x2_stride32_getitem_1"
        do
            for predictionsflag in "--eval-on-trained"
            do
                sbatch exp-explore_strategies_many_features_k7m4-gpu.sbatch -k $nclasses -i $splitidx $splittype -e $explorer -b ${predictionsflag: -1}${featurename:0:1}${featurename: -1}${i}_tex-${condition}c${cpus}g${gpus}_ps${playbackspeed} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize -c $cpus -g $gpus -p $playbackspeed $featurename $logdir $ssdflag $remaindereval $predictionsflag
            done
        done
    done

    condition="randomlazycspp"
    startwithfeatures=1
    explorer="coreset"
    playbackspeed=10000
    cpus=2
    gpus=2
    for splitidx in {0..9}
    do
        for featurename in "-f r3d_18_ap_mean_stride32_flatten" "-f clip_vitb32_embed_32fstride" "-f mvit_v1_b_16x2_stride32_getitem_1" "-f mvit_v1_b_16x2_random_stride32_getitem_1" "-f clip_vitb32_embed_16x2maxpool"
        do
            for predictionsflag in "--eval-on-trained"
            do
                sbatch exp-explore_strategies_many_features_k7m4-cpu.sbatch -k $nclasses -i $splitidx $splittype -e $explorer -b ${predictionsflag: -1}${featurename:0:1}${featurename: -1}${i}_tex-${condition}c${cpus}g${gpus}_ps${playbackspeed} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize -c $cpus -g $gpus -p $playbackspeed $featurename $logdir $ssdflag $remaindereval $predictionsflag
            done
        done
    done

    # VE-Full
    condition="VE-full"
    startwithfeatures=0
    explorer="randomifuniform"
    playbackspeed=3
    featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
    cpus=4
    gpus=4
    for splitidx in {0..9}
    do
        for predictionsflag in "--eval-on-trained"
        do
            sbatch exp-explore_strategies_many_features_k7m4-gpu.sbatch -k $nclasses -i $splitidx $splittype -e $explorer -b ${predictionsflag: -1}${featurenames:0:1}${featurenames: -1}${i}_tex-${condition}c${cpus}g${gpus}_ps${playbackspeed} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize -c $cpus -g $gpus -p $playbackspeed $featurenames $logdir $ssdflag $remaindereval --eager-feature-extraction-labeled --eager-model-training --async-bandit --use-priority --eager-feature-extraction-unlabeled --suspend-lowp $predictionsflag
        done
    done

    # VE-Lazy
    condition="VE-lazy-incremental"
    startwithfeatures=0
    explorer="randomifuniform"
    playbackspeed=10000
    featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
    gpus=2
    cpus=2
    for splitidx in {0..9}
    do
        for alX in 10 50 100
        do
            for predictionsflag in "--eval-on-trained"
            do
                sbatch exp-explore_strategies_many_features_k7m4-gpu.sbatch -k $nclasses -i $splitidx $splittype -e $explorer -b ${predictionsflag: -1}${alX}${featurenames:0:1}${featurenames: -1}${i}_tex-${condition}c${cpus}g${gpus}_ps${playbackspeed} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize -c $cpus -g $gpus -p $playbackspeed $featurenames $logdir $ssdflag $remaindereval $predictionsflag --al-vids-x $alX
            done
        done
    done

    # VE-lazy (preprocess)
    condition="VE-lazy-pp"
    startwithfeatures=1
    explorer="randomifuniform"
    playbackspeed=10000
    featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
    gpus=2
    cpus=2
    for splitidx in {0..9}
    do
        sbatch exp-explore_strategies_many_features_k7m4-cpu.sbatch -k $nclasses -i $splitidx $splittype -e $explorer -b ${featurenames:0:1}${featurenames: -1}${i}_tex-${condition}c${cpus}g${gpus}_ps${playbackspeed} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize -c $cpus -g $gpus -p $playbackspeed $featurenames $logdir $ssdflag $remaindereval --eval-on-trained
    done
done
