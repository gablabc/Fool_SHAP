#!/bin/bash

for dataset in {"adult_income","compas"}
do
    for model in {"rf","xgb","mlp"}
    do
        for rseed in {0..4}
        do
            python3 1_1_train.py --dataset=$dataset --model=$model --rseed=$rseed
            python3 1_2_test.py  --dataset=$dataset --model=$model --rseed=$rseed
        done
    done
done

for dataset in {"marketing","communities"}
do
    for model in {"rf","xgb"}
    do
        for rseed in {0..4}
        do
            python3 1_1_train.py --dataset=$dataset --model=$model --rseed=$rseed
            python3 1_2_test.py  --dataset=$dataset --model=$model --rseed=$rseed
        done
    done
done