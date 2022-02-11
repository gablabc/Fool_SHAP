# # `Exact` explainer

import shap
from shap.maskers import Independent

import argparse
import pandas as pd
import numpy as np
import os

# Local importspyt
from utils import get_data, get_foreground_background, load_model



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_size', type=int, default=200, help='Size of background minibatch')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    # Get the data
    X, y, features, encoder, cat_cols, num_cols = get_data(args.dataset, rseed=args.rseed, encoded=True)
    
    # Get B and F
    foreground, background = get_foreground_background(X, -1, 0,
                                                       args.background_size, args.background_seed)
    
    # Load the model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", filename)
    model.set_params(predictor__n_jobs=-1)

    black_box = model.predict_proba
    demographic_parity = black_box(foreground)[:, 1].mean() - \
                         black_box(background)[:, 1].mean()
    print(f"Demographic Parity : {demographic_parity:.3f}")
    
    # ## Tabular data with independent (Shapley value) masking
    mask = Independent(background, max_samples=args.background_size)
    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(model.predict_proba, mask)
    shap_values = explainer(foreground)[...,1].values

    # TODO Shapley values should sum to the Demographic Parity
    assert np.abs(demographic_parity - shap_values.mean(0).sum()) < 1e-14
    
    # 3d tensor with index (foreground_idx, feature_idx, background_idx)
    phi_z_i_x = np.stack(explainer.values_all_background)[:, :, 1, :]

    # Assert that we can to the attributions for
    # each background sample separately
    assert np.max(shap_values - phi_z_i_x.mean(-1)) < 1e-14

    # The Phis represent how a single background sample
    # affects the Global SHAP values Phi(f, F, B) in the paper
    Phis = phi_z_i_x.mean(0).T

    save_path = os.path.join("attacks", "Phis")
    filename = f"Phis_{args.model}_{args.dataset}_{args.rseed}_"
    filename += f"{args.background_size}_{args.background_seed}"

    # Save Phis locally
    print(Phis.shape)
    np.save(os.path.join(save_path, filename), Phis)
    