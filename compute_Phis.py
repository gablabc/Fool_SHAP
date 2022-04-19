# # `Exact` explainer

import shap
from shap.maskers import Independent
import argparse
import numpy as np
import os

# Local imports
from utils import get_data, get_foreground_background, load_model



if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_size', type=int, default=1000, help='Size of background minibatch')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    # Get the data
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    foreground, background, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)
    
    # Ordinally encode B and F
    background = ordinal_encoder.transform(background)
    foreground = ordinal_encoder.transform(foreground)
    # Permute features to match ordinal encoding
    features = ordinal_encoder.transformers_[0][2] + ordinal_encoder.transformers_[1][2] 

    # Subsets of background and foreground
    subset_background = background[mini_batch_idx]
    subset_foreground = foreground[:200]

    # Load the model
    filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", filename)

    # Generate a black box to explain
    if ohe_encoder is not None:
        # Preprocessing converts to np.ndarray
        black_box = lambda x: model.predict_proba(ohe_encoder.transform(x))
    else:
        black_box = model.predict_proba

    # Fairness
    demographic_parity = black_box(subset_foreground)[:, 1].mean() - \
                         black_box(subset_background)[:, 1].mean()
    print(f"Demographic Parity : {demographic_parity:.3f}")
    
    # ## Tabular data with independent (Shapley value) masking
    mask = Independent(subset_background, max_samples=args.background_size)
    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(black_box, mask)
    shap_values = explainer(subset_foreground)[...,1].values

    # Shapley values should sum to the Demographic Parity
    assert np.allclose(demographic_parity, shap_values.mean(0).sum())
    
    # 3d tensor with index (foreground_idx, feature_idx, background_idx)
    phi_x_i_z = np.stack(explainer.values_all_background)[:, :, 1, :]

    # Assert that we can do the attributions for
    # each background sample separately
    assert np.max(shap_values - phi_x_i_z.mean(-1)) < 1e-14

    # The Phis represent how a single background sample
    # affects the Global SHAP values Phi(f, F, B)
    Phis = phi_x_i_z.mean(0).T

    save_path = os.path.join("attacks", "Phis")
    filename = f"Phis_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    filename += f"background_size_{args.background_size}_seed_{args.background_seed}"

    # Save Phis locally
    print(Phis.shape)
    np.save(os.path.join(save_path, filename), np.column_stack((mini_batch_idx, Phis)) )
