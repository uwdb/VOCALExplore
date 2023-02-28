#!/bin/bash

featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
nsteps=100
strategy="risingbandit"
labelt=1
startwithfeatures=1
banditmetric="ml_f1_score_macro"
logdir="-d many_features_bandit_deer"
validationsize=-1

### No smoothing.
bandittype="basic"
for explorer in "randomifuniform"
do
    for i in {1..2}
    do
        for splitidx in {0..4}
        do
            for C in 1 5 7
            do
                for T in 20 50
                do
                    echo $splitidx, $i, $labelt, $nsteps, $validationsize, $C, $T
                    model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}"
                    remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval testset --bandit-eval-metric $banditmetric"
                    sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b v${validationsize}-${featurenames:0:1}${featurenames: -1}${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize $featurenames $logdir $remainder
                done
            done
        done
    done
done

### Smoothing; hyperparameter exploration.
bandittype="exp"
explorer="randomifuniform"
for w in 3 5 7
do
    for C in 3 5 7
    do
        for T in 20 50
        do
            for i in {1..2}
            do
                for splitidx in {0..4}
                do
                    echo $splitidx, $i, $labelt, $nsteps, $validationsize, $C, $T
                    model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}"
                    remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval testset --bandit-eval-metric $banditmetric --bandit-window $w"
                    sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b v${validationsize}-${featurenames:0:1}${featurenames: -1}${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize $featurenames $logdir $remainder
                done
            done
        done
    done
done

### Smoothing; 3fold validation.
bandittype="exp"
explorer="randomifuniform"
C=5
banditeval="kfold"
for w in 5 7
do
    for i in {1..10}
    do
        for splitidx in {0..4}
        do
            for banditkfold in 3
            do
                for T in 20 50
                do
                    echo $splitidx, $i, $labelt, $nsteps, $validationsize, $C, $T
                    model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}"
                    remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval $banditeval --bandit-kfold-k $banditkfold --bandit-eval-metric $banditmetric --bandit-window $w"
                    sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b bek${banditkfold}_v${validationsize}-${featurenames:0:1}${featurenames: -1}${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize $featurenames $logdir $remainder
                done
            done
        done
    done
done

### Repeat to get F1 measures for more steps at best configuration.
### Smoothing; 3fold validation.
bandittype="exp"
explorer="randomifuniform"
C=5
banditeval="kfold"
nsteps=100
for w in 5
do
    for i in {1..20}
    do
        for splitidx in {0..4}
        do
            for banditkfold in 3
            do
                for T in 50
                do
                    echo $splitidx, $i, $labelt, $nsteps, $validationsize, $C, $T
                    model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}"
                    remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval $banditeval --bandit-kfold-k $banditkfold --bandit-eval-metric $banditmetric --bandit-window $w"
                    sbatch exp-explore_strategies_many_features_deer.sbatch -e $explorer -b bek${banditkfold}_v${validationsize}-${featurenames:0:1}${featurenames: -1}${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -w $startwithfeatures -s $strategy -v $validationsize $featurenames $logdir $remainder
                done
            done
        done
    done
done
