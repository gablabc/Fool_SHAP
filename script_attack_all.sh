#!/bin/bash



# MLP Adults
for rseed in {0..4}
do
    for bseed in {0..4}
    do 
        python3 2_1_compute_Phis.py --model=mlp --background_seed=$bseed --rseed=$rseed --explainer=exact
        python3 2_2_compute_weights.py --model=mlp --dataset=adult_income --rseed=$rseed --background_seed=$bseed --background_size=2000 --min_log=-0.5 --max_log=2 --explainer=exact
    done
    python3 2_3_final_attack.py --model=mlp --dataset=adult_income --rseed=$rseed --background_size=2000 --loc="lower right" --explainer=exact --save
done

# XGB Adults
for rseed in {0..4}
do
    for bseed in {0..4}
    do 
        python3 2_1_compute_Phis.py --model=$model --background_seed=$bseed --rseed=$rseed --explainer=exact
        python3 2_2_compute_weights.py --model=xgb --dataset=adult_income --rseed=$rseed --background_seed=$bseed --background_size=2000 --min_log=-0.5 --max_log=2 --explainer=exact
    done
    python3 2_3_final_attack.py --model=xgb --dataset=adult_income --rseed=$rseed --background_size=2000 --loc="lower right" --explainer=exact --save
done

#RFs have smaller background size (because they are sooo slow)
for rseed in {0..4}
do
    for bseed in {0..4}
    do 
        python3 2_1_compute_Phis.py --model=rf --background_seed=$bseed --explainer=exact
        python3 2_2_compute_weights.py --model=rf --dataset=adult_income --rseed=$rseed --background_seed=$bseed --background_size=1000 --min_log=-0.5 --max_log=2 --explainer=exact
    done
    python3 2_3_final_attack.py --model=rf --dataset=adult_income --rseed=$rseed --background_size=1000 --loc="lower right" --explainer=exact --save
done



# MLP on COMPAS
for rseed in {0..4}
do
    python3 2_1_compute_Phis.py --model=mlp --dataset=compas  --rseed=$rseed --explainer=exact
    python3 2_2_compute_weights.py --model=mlp --dataset=compas --rseed=$rseed --min_log=1 --max_log=3 --background_size=-1 --explainer=exact
    python3 2_3_final_attack.py --model=mlp --dataset=compas --rseed=$rseed --loc="upper left" --explainer=exact --background_size=-1 --save
done

# RF on COMPAS
for rseed in {0..4}
do
    python3 2_1_compute_Phis.py --model=rf --dataset=compas  --rseed=$rseed --explainer=exact
    python3 2_2_compute_weights.py --model=rf --dataset=compas --rseed=$rseed --min_log=0.5 --max_log=3 --explainer=exact --background_size=-1
    python3 2_3_final_attack.py --model=rf --dataset=compas --rseed=$rseed --loc="upper left" --explainer=exact --background_size=-1 --save
done

# XGB on COMPAS
for rseed in {0..4}
do
    python3 2_1_compute_Phis.py --model=xgb --dataset=compas --rseed=$rseed --explainer=exact
    python3 2_2_compute_weights.py --model=xgb --dataset=compas --rseed=$rseed --min_log=-1.5 --max_log=2 --explainer=exact --background_size=-1
    python3 2_3_final_attack.py --model=xgb --dataset=compas --rseed=$rseed --loc="upper left" --explainer=exact --background_size=-1 --save
done



# RF Marketing
for rseed in {0..4}
do
    python3 2_1_compute_Phis.py --dataset=marketing --model=rf --explainer=tree --rseed=$rseed --background_size=-1
    python3 2_2_compute_weights.py --dataset=marketing --model=rf --explainer=tree --rseed=$rseed --min_log=1 --max_log=3 --step_log=10
    python3 2_3_final_attack.py --dataset=marketing --model=rf --explainer=tree --rseed=$rseed --loc="lower right" --save
done

# XGB Marketing
for rseed in {0..4}
do
    python3 2_1_compute_Phis.py --dataset=marketing --model=xgb --explainer=tree --rseed=$rseed --background_size=2000 --background_seed=1
    python3 2_2_compute_weights.py --dataset=marketing --model=xgb --explainer=tree --rseed=$rseed --min_log=-0.5 --max_log=2 --step_log=5
    python3 2_3_final_attack.py --dataset=marketing --model=xgb --explainer=tree --rseed=$rseed --loc="lower right" --save
done


# Communities
for model in {"xgb","rf"}
do
    for rseed in {0..4}
    do
        python3 2_1_compute_Phis.py --dataset=communities --model=$model --explainer=tree --rseed=$rseed
        python3 2_2_compute_weights.py --dataset=communities --model=$model --explainer=tree --rseed=$rseed --min_log=-0.5 --max_log=4 --step_log=40
        python3 2_3_final_attack.py --dataset=communities --model=$model --explainer=tree --rseed=$rseed --loc="lower right" --save
    done
done