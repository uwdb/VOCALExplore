#!/bin/bash

# nclasses=$1

# if [ -z $nclasses ]; then
#     echo "Error: nclasses must be specified"
#     exit 1
# fi

nclasses=20

featurenames="-f r3d_18_ap_mean_stride32_flatten -f clip_vitb32_embed_32fstride -f mvit_v1_b_16x2_stride32_getitem_1 -f mvit_v1_b_16x2_random_stride32_getitem_1 -f clip_vitb32_embed_16x2maxpool"
nsteps=100
strategy="risingbandit"
labelt=1
startwithfeatures=1
banditmetric="ml_f1_score_macro"
logdir="-d many_features_bandit_k7m4"
validationsize=-1

### No smoothing.
bandittype="basic"
for explorer in "randomifuniform"
do
    for C in 1 5 7
    do
        for T in 20 50
        do
            remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval testset --bandit-eval-metric $banditmetric"
            model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}"
            for i in 1
            do
                for splittype in  "zipf-N${nclasses}-s2.0"
                do
                    for splitidx in {0..9}
                    do
                        sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -t $splittype -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
                    done
                done
            done

            for i in {1..10}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
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
            remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval testset --bandit-eval-metric $banditmetric --bandit-window $w"
            model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}"
            for i in 1
            do
                for splittype in  "zipf-N${nclasses}-s2.0"
                do
                    for splitidx in {0..9}
                    do
                        sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -t $splittype -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
                    done
                done
            done

            for i in {1..10}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
            done
        done
    done
done

#### Smoothing; and 3-fold validation.
bandittype="exp"
banditkfold=3
C=5
banditeval="kfold"
for explorer in "randomifuniform"
do
    for w in 5 7
    do
        for T in 20 50
        do
            remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval $banditeval --bandit-kfold-k $banditkfold --bandit-eval-metric $banditmetric --bandit-window $w"
            model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}_bek${banditkfold}"
            for i in {1..4}
            do
                for splittype in "zipf-N${nclasses}-s2.0"
                do
                    for splitidx in {0..9}
                    do
                        sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -t $splittype -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
                    done
                done
            done

            for i in {1..40}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
            done
        done
    done
done

### Run for more steps at the best configuration to get F1 measures.
#### Smoothing; 3-fold validation.
bandittype="exp"
banditkfold=3
C=5
banditeval="kfold"
nsteps=100
for explorer in "randomifuniform"
do
    for w in 5
    do
        for T in 50
        do
            remainder="-- --bandit-type $bandittype --bandit-C $C --bandit-T $T --bandit-eval $banditeval --bandit-kfold-k $banditkfold --bandit-eval-metric $banditmetric --bandit-window $w"
            model_suffix="ek${banditmetric}_C${C}_T${T}_b${bandittype}_w${w}_bek${banditkfold}"
            for i in {1..4}
            do
                for splittype in "zipf-N${nclasses}-s2.0"
                do
                    for splitidx in {0..9}
                    do
                        sbatch exp-explore_strategies_many_features_k7m4-any-skewed.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -i $splitidx -l $labelt -n $nsteps -t $splittype -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
                    done
                done
            done

            for i in {1..40}
            do
                sbatch exp-explore_strategies_many_features_k7m4-any.sbatch -c $nclasses -e $explorer -b rb${i}bandit_${model_suffix} -l $labelt -n $nsteps -w $startwithfeatures -s $strategy $featurenames $logdir $remainder
            done
        done
    done
done
