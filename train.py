import argparse
import pandas as pd
import numpy as np
import os

# Imports for data splitting
from sklearn.model_selection import StratifiedKFold

# Local imports
from utils import get_data, init_model
from utils import get_best_cv_model, get_hp_grid, save_model



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--k', type=int, default=5, help='Number of splits in cross-validation')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations of the RandomSearchCV')
    args = parser.parse_args()

    # Get the data
    X, y, _, _, cat_cols, num_cols = get_data(args.dataset, rseed=args.rseed, encoded=True)
    # Training set is the first index
    X = X[0]
    y = y[0]

    # Prepare the K-Fold cross-validation
    cross_validator = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    # Get the hyperparameter grid (specific to each model for now)
    hp_grid = get_hp_grid(os.path.join("hyper_params", f"{args.model}_grid.json"))
    
    # Load a un-trained (uninitialized) model
    model, hp_grid = init_model(args.model, hp_grid, cat_cols, num_cols)

    # Grid Search
    best_model, perfs = get_best_cv_model(X, y, model, hp_grid, 
                                          cross_validator, args.n_iter)
    
    # Save model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    save_model(args.model, best_model, "models", filename)
    