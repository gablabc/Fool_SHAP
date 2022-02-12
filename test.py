import argparse
import pandas as pd
import numpy as np
import os

# Imports for data splitting
from sklearn.metrics import roc_auc_score

# Local imports
from utils import get_data, load_model



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    args = parser.parse_args()

    # Get the data
    X, y, _, _, cat_cols, num_cols = get_data(args.dataset, rseed=args.rseed, encoded=True)
    # Test set is the second index
    X = X[1]
    y = y[1]
    
    # Load the model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", filename)
    model.set_params(predictor__n_jobs=-1)
    
    # Error on the test set
    proba = model.predict_proba(X)[:, 1]
    accuracy = 100 * np.mean( (proba>=0.5) == y)
    # TODO AUC
    AUC = roc_auc_score(y, proba)
    
    performance_file = os.path.join("models", "performance.csv")
    # Make the file if it does not exist
    if not os.path.exists(performance_file):
        with open(performance_file, 'w') as file:
            file.write("dataset,model,rseed,accuracy,AUC\n")
    # Append new results to the file
    with open(performance_file, 'a') as file:
        file.write(f"{args.dataset},{args.model},{args.rseed},{accuracy:.2f},{AUC:.2f}")
