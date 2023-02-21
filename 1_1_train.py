""" Train models on the datasets with various train/test random splits """
import argparse
import os
import multiprocessing

# Imports for data splitting
from sklearn.model_selection import StratifiedShuffleSplit

# Local imports
from src.utils import get_data, MODELS
from src.utils import get_best_cv_model, get_hp_grid, save_model



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--k', type=int, default=5, help='Number of splits in cross-validation')
    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations of the RandomSearchCV')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs on which to run the search')
    args = parser.parse_args()

    # Get the data
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)
    # Training set is the first index
    X = X_split["train"]
    y = y_split["train"]

    # Process the data
    y = y.to_numpy()
    X = ordinal_encoder.transform(X)
    
    # Some model may perhaps not require OHE
    if ohe_encoder is not None:
        X = ohe_encoder.transform(X)

    # Prepare the K-Fold cross-validation
    cross_validator = StratifiedShuffleSplit(n_splits=args.k, random_state=42)
    # Get the hyperparameter grid (specific to each model for now)
    hp_grid = get_hp_grid(os.path.join("models", "hyper_params", f"{args.model}_grid.json"))
    
    # Load a un-trained (uninitialized) model
    model = MODELS[args.model]
    if args.model == "xgb":
        model.set_params(n_jobs=multiprocessing.cpu_count() // args.n_jobs)

    # Grid Search
    best_model, perfs = get_best_cv_model(X, y, model, hp_grid, cross_validator, args.n_iter, args.n_jobs)
    
    # Save model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    save_model(args.model, best_model, "models", filename)
    