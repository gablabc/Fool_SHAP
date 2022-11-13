""" 
The second part of the attack is to compute the non-uniform weights
over the background dataset D_1. To do so, a MCF is solved to minimize the
amplitude of the SHAP value of a sensitive attribute while ensuring that the
non-uniform distribution remains close to the uniform on D_1
"""
import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 21
mp.rcParams['font.family'] = 'serif'

# Local imports
from utils import get_data, get_foreground_background, load_model, SENSITIVE_ATTR
from stealth_sampling import brute_force


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='mlp', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--explainer', type=str, default='exact', help='exact or tree')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    parser.add_argument('--background_size', type=int, default=-1, help='Size of background minibatch, -1 means all')
    parser.add_argument('--time_multiplier', type=int, default=1, help='How much longer than computing the weights')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    D_0, D_1, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)

    # OHE+Ordinally encode B and F
    if ordinal_encoder is not None:
        D_0 = ordinal_encoder.transform(D_0)
        D_1 = ordinal_encoder.transform(D_1)
        # Permute features to match ordinal encoding
        numerical_features = ordinal_encoder.transformers_[0][2]
        categorical_features = ordinal_encoder.transformers_[1][2] 
        features = numerical_features + categorical_features
    if ohe_encoder is not None:
        D_0 = ohe_encoder.transform(D_0)
        D_1 = ohe_encoder.transform(D_1)

    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)

    # All background/foreground predictions
    if args.explainer == "tree" and args.model == "xgb":
        # When explaining Boosted trees with TreeSHAP, we explain the logit
        f_D_0 = model.predict(D_0, output_margin=True).reshape((-1, 1))
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict(D_1, output_margin=True).reshape((-1, 1))
        f_D_1_B = f_D_1[mini_batch_idx]
    else:
        # We explain the probability of class 1
        f_D_0 = model.predict_proba(D_0)[:, [1]]
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict_proba(D_1)[:, [1]]
        f_D_1_B = f_D_1[mini_batch_idx]

    # Get the sensitive feature
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])

    # Load the Phis
    phis_path = os.path.join("attacks", "Phis")
    tmp_filename = f"{args.explainer}_Phis_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"background_size_{args.background_size}_seed_{args.background_seed}.npy"

    load = np.load(os.path.join(phis_path, tmp_filename))
    minibatch_idx = load[:, 0].astype(int)
    Phi_S0_zj = load[:, 1:]


    ############################################################################################
    #                                       Experiment                                         #
    ############################################################################################

    # Get the time limit
    weights_path = os.path.join("attacks", "Weights")
    tmp_filename = f"{args.explainer}_Weights_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"B_size_{args.background_size}_seed_{args.background_seed}.txt"
    with open(os.path.join(weights_path, tmp_filename), "r") as file:
        time_limit = float(file.read())

    significance = 0.01
    S_1_p_idx = brute_force(f_D_0, f_S_0, f_D_1_B, Phi_S0_zj, s_idx, significance, args.time_multiplier * time_limit)
    S_1_p_idx = mini_batch_idx[S_1_p_idx]

    ############################################################################################
    #                                     Save Results                                         #
    ############################################################################################


    # Save the weights if success
    brute_path = os.path.join("attacks", "Brute")
    tmp_filename = f"{args.explainer}_Brute_{args.time_multiplier}_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"B_size_{args.background_size}_seed_{args.background_seed}"
    
    # Save the optimal weights
    np.save(os.path.join(brute_path, tmp_filename), S_1_p_idx)
    
