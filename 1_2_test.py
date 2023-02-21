""" 
    Evaluate Models on the Test set and compute the Demographic Parity.
    All results are stored in models/performance.csv

"""
import argparse
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# Local imports
from src.utils import get_data, load_model, get_foreground_background



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    args = parser.parse_args()

    # Get the data
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)
    
    ############## Test Set Performance ##############
    
    # Test set
    X = X_split["test"]
    y = y_split["test"]

    # Process the data
    y = y.to_numpy()
    X = ordinal_encoder.transform(X)

    # Some model may perhaps not require OHE
    if ohe_encoder is not None:
        X = ohe_encoder.transform(X)

    # Load the model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", filename)
    
    # Error on the test set
    proba = model.predict_proba(X)[:, 1]
    accuracy = 100 * np.mean( (proba>=0.5) == y)
    AUC = roc_auc_score(y, proba)

    ############## Demographic Parity ##############

    # Get B and F
    D_0, D_1 = get_foreground_background(X_split, args.dataset)

    # Ordinally encode B and F
    D_0 = ordinal_encoder.transform(D_0)
    D_1 = ordinal_encoder.transform(D_1)
    
    if ohe_encoder is not None:
        D_0 = ohe_encoder.transform(D_0)
        D_1 = ohe_encoder.transform(D_1)

    # Fairness
    demographic_parity = model.predict_proba(D_0)[:, 1].mean() - \
                         model.predict_proba(D_1)[:, 1].mean()


    ############## Save Results ##############

    performance_file = os.path.join("models", "performance.csv")
    # Make the file if it does not exist
    if not os.path.exists(performance_file):
        with open(performance_file, 'w') as file:
            file.write("dataset,model,rseed,accuracy,AUC,DP\n")
    # Append new results to the file
    with open(performance_file, 'a') as file:
        file.write(f"{args.dataset},{args.model},{args.rseed},{accuracy:.2f},{AUC:.2f},{demographic_parity:.2f}\n")
