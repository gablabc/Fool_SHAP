#!/bin/bash



# Adults
for model in "mlp" "rf" "xgb"
do
    for rseed in {0..4}
    do
        for bseed in {0..4}
        do 
            python3 compute_Phis.py --model=$model --background_seed=$s
            python3 compute_weights.py --model=$model --rseed=$rseed --background_seed=$bseed --background_size=2000 --min_log=-0.2 --max_log=2
        done
        python3 final_attack.py --model=$model --dataset=adult_income --rseed=$rseed --background_size=2000
    done
done


# RF on COMPAS
for rseed in {0..4}
do
    python3 compute_Phis.py --model=rf --dataset=compas  --rseed=$rseed
    python3 compute_weights.py --model=rf --dataset=compas --rseed=$rseed --min_log=1 --max_log=3
    python3 final_attack.py --model=rf --dataset=compas --rseed=$rseed
done

# MLP on COMPAS
for rseed in {0..4}
do
    python3 compute_Phis.py --model=mlp --dataset=compas  --rseed=$rseed
    python3 compute_weights.py --model=mlp --dataset=compas --rseed=$rseed --min_log=1 --max_log=3
    python3 final_attack.py --model=mlp --dataset=compas --rseed=$rseed
done

# XGB on COMPAS
for rseed in {0..4}
do
    python3 compute_Phis.py --model=xgb --dataset=compas  --rseed=$rseed
    python3 compute_weights.py --model=xgb --dataset=compas --rseed=$rseed --min_log=1 --max_log=3
    python3 final_attack.py --model=xgb --dataset=compas --rseed=$rseed
done