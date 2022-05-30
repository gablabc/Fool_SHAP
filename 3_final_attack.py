import argparse
import glob
import numpy as np
import os
import shap
from shap.maskers import Independent
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})

# Local imports
from utils import get_data, get_foreground_background, load_model
from utils import audit_detection, SENSITIVE_ATTR


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='compas', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_size', type=int, default=-1, help='Size of background minibatch')
    args = parser.parse_args()
    np.random.seed(42)
    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    foreground, background = get_foreground_background(X_split, args.dataset)
    
    # Ordinally encode B and F
    background = ordinal_encoder.transform(background)
    foreground = ordinal_encoder.transform(foreground)
    # Permute features to match ordinal encoding
    numerical_features = ordinal_encoder.transformers_[0][2]
    categorical_features = ordinal_encoder.transformers_[1][2] 
    features = numerical_features + categorical_features
    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)
    
    # Generate a black box to explain
    if ohe_encoder is not None:
        # Preprocessing converts to np.ndarray
        black_box = lambda x: model.predict_proba(ohe_encoder.transform(x))
    else:
        black_box = model.predict_proba
    # All background/foreground predictions
    b_pred = black_box(background)[:, [1]]
    f_pred = black_box(foreground)[:, [1]]
    subset_f_pred = f_pred[:200]

    # Get the sensitive feature
    print(f"Features : {features}")
    print(f"Sensitive attribute : {SENSITIVE_ATTR[args.dataset]}")
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])
    not_s_idx = [i for i in range(n_features) if not i==s_idx]
    print(f"Index of sensitive feature : {s_idx}")
    
    ############################################################################################
    #                                         Unbiased                                         #
    ############################################################################################

    # Fairness
    demographic_parity = black_box(foreground)[:, 1].mean() - \
                         black_box(background)[:, 1].mean()
    print(f"Full Demographic Parity : {demographic_parity:.3f}")

    # ## Tabular data with independent (Shapley value) masking
    mask = Independent(background, max_samples=200)
    print(mask.data.shape)
    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(black_box, mask)
    shap_values = explainer(foreground[:200])[...,1].values.mean(0)

    init_rank = rankdata(shap_values)[s_idx]
    print(f"Rank before attack : {init_rank}")
    print(f"Subsampled Demographic Parity : {shap_values.sum():.3f}\n")
    #print(f"Average Background pred : {black_box(mask.data)[:, 1].mean():.3f}")

    ############################################################################################
    #                                          Biased                                          #
    ############################################################################################
    # Load the Weights
    weights = np.zeros(background.shape[0])
    weights_path = os.path.join("attacks", "Weights")

    tmp_filename = f"Weights_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"B_size_{args.background_size}_seed_*.npy"
    # Only one batch
    if args.background_size == -1:
        weight_file = glob.glob(os.path.join(weights_path, tmp_filename))[0]
        load = np.load(weight_file)
        weights[load[:, 0].astype(int)] += load[:, 1]
    # Many minibatches, se how they combine to reduce detection rate
    else:
        all_weight_files = glob.glob(os.path.join(weights_path, tmp_filename))
        detections = []
        for weight_file in all_weight_files:
            detections.append(0)
            load = np.load(weight_file)
            weights[load[:, 0].astype(int)] += load[:, 1]
            for _ in range(100):
                # Sample Background distribution
                biased_idx = np.random.choice(background.shape[0], 200, p=weights/np.sum(weights))
                subset_b_pred = b_pred[biased_idx]

                detections[-1] += audit_detection(b_pred, f_pred, 
                                                subset_b_pred, subset_f_pred, 0.01)
        #plt.figure()
        #plt.plot(range(len(detections)), detections, "b-o")
        #plt.ylabel("Detection")
        #plt.xlabel("Number of background minibatches")

        #plt.figure()
        #plt.hist(b_pred, bins=25, alpha=0.25, density=True)
        #plt.hist(subset_b_pred, bins=25, alpha=0.25, density=True)
        #plt.show()

    # The company submits the final dataset to the audit
    # Sample Background distribution
    biased_idx = np.random.choice(background.shape[0], 200, p=weights/np.sum(weights))
    subset_b_pred = b_pred[biased_idx]
    detection = audit_detection(b_pred, f_pred, 
                                subset_b_pred, subset_f_pred, 0.01)
    print(f"Detection fo the fraud : {detection == 1}")

    # ## Tabular data with independent (Shapley value) masking
    maskb = Independent(background[biased_idx], max_samples=200)
    print(maskb.data.shape)
    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(black_box, maskb)
    biased_shap_values = explainer(foreground[:200])[...,1].values.mean(0)

    final_rank = rankdata(biased_shap_values)[s_idx]
    print(f"Rank after attack : {final_rank}")
    print(f"Biased Subsampled Demographic Parity : {biased_shap_values.sum():.3f}\n")
    #print(f"Average Biased Background pred : {black_box(maskb.data)[:, 1].mean():.3f}")


    results_file = os.path.join("attacks", "results.csv")
    # Make the file if it does not exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as file:
            file.write("dataset,model,rseed,detection,init_rank,final_rank\n")
    # Append new results to the file
    with open(results_file, 'a') as file:
        file.write(f"{args.dataset},{args.model},{args.rseed},")
        file.write(f"{detection==1},{int(init_rank):d},{int(final_rank):d}\n")



    # Where to save figure
    figure_path = os.path.join(f"Images/{args.dataset}/{args.model}/")
    tmp_filename = f"rseed_{args.rseed}_B_size_{args.background_size}"

    # Sort the features
    sorted_features_idx = np.argsort(shap_values)

    # Plot results
    df = pd.DataFrame(np.column_stack((shap_values[sorted_features_idx], 
                                       biased_shap_values[sorted_features_idx])),
                      columns=["Original", "Manipulated"],
                      index=[features[i] for i in sorted_features_idx])
    df.plot.barh()
    plt.plot([0, 0], plt.gca().get_ylim(), "k-")
    plt.xlabel('Shap value')
    plt.savefig(os.path.join(figure_path, f"attack_{tmp_filename}.pdf"), bbox_inches='tight')
