#!/bin/bash

for dataset in {"adult_income","marketing","default_credit","compas"}
do
    for model in {"rf","gbt","mlp"}
    do
        for rseed in {0..4}
        do
            python3 train.py --dataset=$dataset --model=$model --rseed=$rseed
            python3 test.py  --dataset=$dataset --model=$model --rseed=$rseed
        done
    done
done