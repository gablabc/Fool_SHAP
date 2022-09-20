""" 
The first part of the attack is to compute the coefficients
Phi(f, S_0', z^(j)) for all z^(j) in D_1. These coefficients
will be stored and used later in the MCF
"""

import shap
from shap.maskers import Independent
import argparse
import numpy as np
import os

# Local imports
from utils import get_data, get_foreground_background, load_model
from utils import tree_shap



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--explainer', type=str, default='tree', help='exact or tree')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_size', type=int, default=-1, help='Size of background minibatch, -1 means all')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    # Get the data
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    D_0, D_1, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)
    #print(len(mini_batch_idx))
    #print(mini_batch_idx)

    # Ordinally encoder
    if ordinal_encoder is not None:
        D_1 = ordinal_encoder.transform(D_1)
        D_0 = ordinal_encoder.transform(D_0)
    else:
        D_1 = D_1.to_numpy()
        D_0 = D_0.to_numpy()

    # Subsets of background and foreground
    M = 200
    D_1_B = D_1[mini_batch_idx]
    S_0 = D_0[:M]

    # Load the model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", filename)
    
    # Use Monkey-Patched implementation of SHAP
    if args.explainer == "exact":

        # Generate a black box to explain
        if ohe_encoder is not None:
            # Preprocessing converts to np.ndarray
            black_box = lambda x: model.predict_proba(ohe_encoder.transform(x))
        else:
            black_box = model.predict_proba

        mask = Independent(D_1_B, max_samples=len(mini_batch_idx))
        explainer = shap.explainers.Exact(black_box, mask)
        explainer(S_0)

        # Extract the LSV from the Hacked version of SHAP
        LSV = explainer.LSV
    
    # Use our custom TreeSHAP
    elif args.explainer == "tree":

        LSV = tree_shap(model, S_0, D_1_B, ordinal_encoder, ohe_encoder)
    
    else:
        raise ValueError("Wrong type of explainer")

    # Phi(f, S_0', z^(j))
    Phi_S_0_zj = LSV.mean(1).T

    save_path = os.path.join("attacks", "Phis")
    filename = f"{args.explainer}_Phis_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    filename += f"background_size_{args.background_size}_seed_{args.background_seed}"

    # Save Phis locally
    print(Phi_S_0_zj.shape)
    np.save(os.path.join(save_path, filename), np.column_stack((mini_batch_idx, Phi_S_0_zj)) )
