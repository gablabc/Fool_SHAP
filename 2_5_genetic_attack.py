""" 

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
import genetic
from utils import audit_detection


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, "xgb", rseed=args.rseed)

    # Get B and F
    D_0, D_1 = get_foreground_background(X_split, args.dataset)

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
    tmp_filename = f"xgb_{args.dataset}_{args.rseed}"
    model = load_model("xgb", "models", tmp_filename)

    # All background/foreground predictions
    M = 100
    f_D_0 = model.predict_proba(D_0)[:, [1]]
    S_0 = D_0[:M]
    f_S_0 = f_D_0[:M]
    f_D_1 = model.predict_proba(D_1)[:, [1]]
    S_1 = D_1[:M]

    # Get the sensitive feature
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])

    ############################################################################################
    #                                       Experiment                                         #
    ############################################################################################
    hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
    file_path = os.path.join("Images", args.dataset, "xgb")
    tmp_filename = f"genetic_{args.dataset}_rseed_{args.rseed}"

    explainer = genetic.Explainer(model)

    # Sensitive index
    s_idx = 7
    # Peturbate the background
    one_hot_columns = list(range(len(numerical_features), S_0.shape[1]))
    alg = genetic.GeneticAlgorithm(explainer, S_0, S_1, s_idx, constant=one_hot_columns, pop_count=25)
    detection = 0
    for i in range(1, 5):
        alg.fool_aim(max_iter=50, random_state=0)
        S_1_prime = alg.S_1_prime
        f_S_1 = model.predict_proba(S_1_prime)[:, [1]]

        # Detection
        detection = audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, significance=0.01)

        hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
        plt.figure()
        plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
        plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
        plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
        plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
        plt.xlabel("Output")
        plt.ylabel("CDF")
        plt.legend(framealpha=1, loc="lower right")
        plt.savefig(os.path.join(file_path, tmp_filename + f"_ite_{50*i:d}_detect_{detection}.pdf"), bbox_inches='tight')

    ############################################################################################
    #                                     Save Results                                         #
    ############################################################################################

    results_file = os.path.join("attacks", "Genetic", tmp_filename + ".csv")
    alg.iter_log.to_csv(results_file)
    
