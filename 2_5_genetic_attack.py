""" 

"""
import argparse
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

import os, sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")

# Local imports
from src.genetic import GeneticAlgorithm
from src.utils import get_data, get_foreground_background, load_model, SENSITIVE_ATTR


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='xgb', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--n_ite', type=int, default=100, help='Number of iterations')
    parser.add_argument('--n_pop', type=int, default=10, help='Size of the population')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    D_0, D_1 = get_foreground_background(X_split, args.dataset)
    N_0 = D_0.shape[0]
    N_1 = D_1.shape[0]

    # OHE+Ordinally encode B and F
    D_0 = ordinal_encoder.transform(D_0)
    D_1 = ordinal_encoder.transform(D_1)
    # Permute features to match ordinal encoding
    if isinstance(ordinal_encoder, ColumnTransformer):
        numerical_features = ordinal_encoder.transformers_[0][2]
        categorical_features = ordinal_encoder.transformers_[1][2] 
        features = numerical_features + categorical_features
    else:
        numerical_features = features

    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)


    # Generate a black box to explain
    if ohe_encoder is not None:
        mapper = ohe_encoder.transform
    else:
        mapper = lambda x: x
    if args.model == "xgb":
        # TreeSHAP on XGB explains the logit
        f_D_0 = model.predict(mapper(D_0), output_margin=True)
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict(mapper(D_1), output_margin=True)
    else:
        # We explain the probability of class 1
        f_D_0 = model.predict_proba(mapper(D_0))[:, 1]
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict_proba(mapper(D_1))[:, 1]

    # Get the sensitive feature
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])

    # Plot setup
    tmp_filename = f"{args.model}_{args.dataset}_rseed_{args.rseed}"

    alg = GeneticAlgorithm(model, D_0[:200], D_1[:200], f_S_0, f_D_0, f_D_1, s_idx, 
                            pop_count=args.n_pop, mutation_with_constraints=False, 
                            constant=list(range(len(numerical_features), n_features)),
                            ordinal_encoder=ordinal_encoder, ohe_encoder=ohe_encoder)
    
    alg.fool_aim(max_iter=args.n_ite, random_state=0)

    # Save logs
    results_file = os.path.join("attacks", "Genetic", tmp_filename + ".csv")
    pd.DataFrame(alg.iter_log).to_csv(results_file, index=False)
    
    # Save final dataset
    np.save(os.path.join("attacks", "Genetic", tmp_filename), 
                            alg.result_explanation['changed'])
