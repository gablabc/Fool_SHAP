""" 
The final part of the attack is for the company to cherry-pick S_0' with the
non-uniform weights and provide this benchmark dataset to the audit. The audit,
on their side, will try to detect the fraud and then run SHAP on the cherry-picked
data. The resulting SHAP value of the sensitive attribute should be small. Results
are stored in attacks/results.csv
"""
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
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':10})

# Local imports
from src.utils import get_data, get_foreground_background, load_model, tree_shap
from src.utils import audit_detection, confidence_interval, SENSITIVE_ATTR


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--explainer', type=str, default='tree', help='exact or tree')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_size', type=int, default=1000, help='Size of background minibatch, -1 means all')
    parser.add_argument("--loc", type=str, default="best", help="Location of the Legend in CDFs plot")
    parser.add_argument("--save", action='store_true', help="Save results in csv")
    args = parser.parse_args()
    np.random.seed(42)
    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get foreground and background
    D_0, D_1 = get_foreground_background(X_split, args.dataset)
    N_1 = D_1.shape[0]

    # Ordinal encoder
    if ordinal_encoder is not None:
        D_0 = ordinal_encoder.transform(D_0)
        D_1 = ordinal_encoder.transform(D_1)
        # Permute features to match ordinal encoding
        numerical_features = ordinal_encoder.transformers_[0][2]
        categorical_features = ordinal_encoder.transformers_[1][2] 
        features = numerical_features + categorical_features
    
    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)
    print(model)

    # Generate a black box to explain
    if ohe_encoder is not None:
        if args.explainer == "tree" and args.model == "xgb":
            black_box = lambda x:model.predict(ohe_encoder.transform(x), output_margin=True)
            f_D_0 = black_box(D_0).reshape((-1, 1))
            f_D_1 = black_box(D_1).reshape((-1, 1))
        else:
            # Preprocessing converts to np.ndarray
            black_box = lambda x: model.predict_proba(ohe_encoder.transform(x))
            # All background/foreground predictions
            f_D_0 = black_box(D_0)[:, [1]]
            f_D_1 = black_box(D_1)[:, [1]]
    else:
        black_box = model.predict_proba
        raise NotImplementedError("This case is not yet handled")

    if ordinal_encoder is None:
        D_0 = D_0.to_numpy()
        D_1 = D_1.to_numpy()
    M = 200
    S_0 = D_0[np.arange(M)]
    f_S_0 = f_D_0[:M]

    # Get the sensitive feature
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])
    not_s_idx = [i for i in range(n_features) if not i==s_idx]
    
    ############################################################################################
    #                                         Unbiased                                         #
    ############################################################################################

    # # Fairness
    # demographic_parity = f_D_0.mean() - f_D_1.mean()
    # print(f"Full Demographic Parity : {demographic_parity:.3f}")

    # if args.explainer == "exact":
    #     # Choose the background uniformly at random
    #     mask = Independent(D_1, max_samples=M)
    #     explainer = shap.explainers.Exact(black_box, mask)
    #     explainer(S_0)

    #     # Local Shapley Values phi(f, x^(i), z^(j))
    #     LSV = explainer.LSV
    
    # # Use our custom TreeSHAP
    # elif args.explainer == "tree":

    #     # Subsample the background
    #     S_1 = D_1[np.random.choice(N_1, M)]
    #     LSV = tree_shap(model, S_0, S_1, ordinal_encoder, ohe_encoder)

    # else:
    #     raise ValueError("Wrong type of explainer")

    # # Choose a subset uniformly at random (to simulate a honest result)
    # honest_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
    # CI = confidence_interval(LSV, 0.05)

    # init_rank = rankdata(honest_shap_values)[s_idx]
    # init_abs = np.abs(honest_shap_values[s_idx])
    # print(f"Rank before attack : {init_rank}")
    # print(f"Abs value before attack : {init_abs}")
    # print(f"Subsampled Demographic Parity : {honest_shap_values.sum():.3f}\n")
    


    ############################################################################################
    #                                      Brute-Force                                         #
    ############################################################################################
    # # Load the Weights
    # brute_path = os.path.join("attacks", "Brute")
    # tmp_filename = f"{args.explainer}_Brute_1_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    # tmp_filename += f"B_size_{args.background_size}_seed_0.npy"
    # S_1_idx = load = np.load(os.path.join(brute_path, tmp_filename))
    # f_S_1 = f_D_1[S_1_idx]
    # S_1 = D_1[S_1_idx]

    # if args.explainer == "exact":
    #     maskb = Independent(S_1, max_samples=M)
    #     explainer = shap.explainers.Exact(black_box, maskb)
    #     explainer(S_0)

    #     # Local Shapley Values phi(f, x^(i), z^(j))
    #     LSV = explainer.LSV
    
    # # Use our custom TreeSHAP
    # elif args.explainer == "tree":
    #     # Subsample the background
    #     LSV = tree_shap(model, S_0, S_1, ordinal_encoder, ohe_encoder)

    # else:
    #     raise ValueError("Wrong type of explainer")


    # biased_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
    # brute_rank = rankdata(biased_shap_values)[s_idx]
    # brute_abs = np.abs(biased_shap_values[s_idx])
    # print(f"Rank after brute attack : {brute_rank}")
    # print(f"Abs value before brute attack : {brute_abs}")



############################################################################################
    #                                      Brute-Force    10                                     #
    ############################################################################################
    # Load the Weights
    brute_path = os.path.join("attacks", "Brute")
    tmp_filename = f"{args.explainer}_Brute_10_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"B_size_{args.background_size}_seed_0.npy"
    S_1_idx = load = np.load(os.path.join(brute_path, tmp_filename))
    f_S_1 = f_D_1[S_1_idx]
    S_1 = D_1[S_1_idx]

    if args.explainer == "exact":
        maskb = Independent(S_1, max_samples=M)
        explainer = shap.explainers.Exact(black_box, maskb)
        explainer(S_0)

        # Local Shapley Values phi(f, x^(i), z^(j))
        LSV = explainer.LSV
    
    # Use our custom TreeSHAP
    elif args.explainer == "tree":
        # Subsample the background
        LSV = tree_shap(model, S_0, S_1, ordinal_encoder, ohe_encoder)

    else:
        raise ValueError("Wrong type of explainer")


    biased_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
    brute_ten_rank = rankdata(biased_shap_values)[s_idx]
    brute_ten_abs = np.abs(biased_shap_values[s_idx])
    print(f"Rank after brute attack : {brute_ten_rank}")
    print(f"Abs value before brute attack : {brute_ten_abs}")



    ############################################################################################
    #                                       Fool SHAP                                          #
    ############################################################################################
    # # Load the Weights
    # weights = np.zeros(N_1)
    # weights_path = os.path.join("attacks", "Weights")

    # tmp_filename = f"{args.explainer}_Weights_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    # tmp_filename += f"B_size_{args.background_size}_seed_*.npy"
    # # Only one batch
    # if args.background_size == -1:
    #     weight_file = glob.glob(os.path.join(weights_path, tmp_filename))[0]
    #     load = np.load(weight_file)
    #     weights[load[:, 0].astype(int)] += load[:, 1]
    # # Many minibatches, see how they combine to reduce detection rate
    # else:
    #     all_weight_files = glob.glob(os.path.join(weights_path, tmp_filename))
    #     detections = []
    #     for weight_file in all_weight_files:
    #         detections.append(0)
    #         load = np.load(weight_file)
    #         weights[load[:, 0].astype(int)] += load[:, 1]
    #         # for _ in range(100):
    #         #     # Sample Background distribution
    #         #     biased_idx = np.random.choice(N_1, 200, p=weights/np.sum(weights))
    #         #     f_S_1 = f_D_1[biased_idx]

    #         #     detections[-1] += audit_detection(f_D_0, f_D_1,
    #         #                                       f_S_0, f_S_1, 0.01)
    #     # plt.figure()
    #     # plt.plot(range(len(detections)), detections, "b-o")
    #     # plt.ylabel("Detections")
    #     # plt.xlabel("Number of background minibatches")
    #     # plt.show()


    # # The company submits the final dataset to the audit
    # # Sample Background distribution
    # biased_idx = np.random.choice(N_1, M, p=weights/np.sum(weights))
    # f_S_1 = f_D_1[biased_idx]
    # detection = audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, 0.01)
    # print(f"Detection fo the fraud : {detection == 1}")


    # # Select the cherry-picked background
    # print(len(np.unique(biased_idx)))
    # S_1 = D_1[biased_idx]

    # if args.explainer == "exact":
    #     maskb = Independent(S_1, max_samples=M)
    #     explainer = shap.explainers.Exact(black_box, maskb)
    #     explainer(S_0)

    #     # Local Shapley Values phi(f, x^(i), z^(j))
    #     LSV = explainer.LSV
    
    # # Use our custom TreeSHAP
    # elif args.explainer == "tree":
    #     # Subsample the background
    #     LSV = tree_shap(model, S_0, S_1, ordinal_encoder, ohe_encoder)

    # else:
    #     raise ValueError("Wrong type of explainer")


    # biased_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
    # CI_b = confidence_interval(LSV, 0.01)


    # final_rank = rankdata(biased_shap_values)[s_idx]
    # final_abs = np.abs(biased_shap_values[s_idx])
    # print(f"Rank after attack : {final_rank}")
    # print(f"Abs value before attack : {final_abs}")
    # print(f"Biased Subsampled Demographic Parity : {biased_shap_values.sum():.3f}\n\n")


    # if args.save:
    #     results_file = os.path.join("attacks", "results.csv")
    #     # Make the file if it does not exist
    #     if not os.path.exists(results_file):
    #         with open(results_file, 'w') as file:
    #             file.write("dataset,model,rseed,detection,init_rank,brute_rank,final_rank,init_abs,brute_abs,final_abs\n")
    #     # Append new results to the file
    #     with open(results_file, 'a') as file:
    #         file.write(f"{args.dataset},{args.model},{args.rseed},")
    #         file.write(f"{detection==1},{int(init_rank):d},{int(brute_rank):d},{int(final_rank):d},")
    #         file.write(f"{init_abs:.6f},{brute_abs:.6f},{final_abs:.6f}\n")

    
    if args.save:
        results_file = os.path.join("attacks", "results_2.csv")
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                file.write("dataset,model,rseed,detection,init_rank,brute_rank,final_rank,init_abs,brute_abs,final_abs\n")
        # Append new results to the file
        with open(results_file, 'a') as file:
            file.write(f"{args.dataset},{args.model},{args.rseed},")
            file.write(f"{1==1},{int(1):d},{1:d},{1:d},")
            file.write(f"{1:.6f},{1:.6f},{brute_ten_abs:.6f}\n")


    # # Where to save figure
    # figure_path = os.path.join(f"Images/{args.dataset}/{args.model}/")
    # tmp_filename = f"rseed_{args.rseed}_B_size_{args.background_size}"

    # # Plot the CDFs
    # hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
    # plt.figure()
    # plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
    # plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
    # plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
    # plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
    # plt.xlabel("Output")
    # plt.ylabel("CDF")
    # plt.legend(framealpha=1, loc=args.loc)
    # plt.savefig(os.path.join(figure_path, f"CDFs_{tmp_filename}.pdf"), bbox_inches='tight')


    # # Sort the features
    # sorted_features_idx = np.argsort(honest_shap_values)

    # # Plot Feature Attributions
    # df = pd.DataFrame(np.column_stack((honest_shap_values[sorted_features_idx], 
    #                                    biased_shap_values[sorted_features_idx])),
    #                   columns=["Original", "Manipulated"],
    #                   index=[features[i] for i in sorted_features_idx])
    # if df.shape[0] > 22:
    #     df = df.iloc[:22]
    # df.plot.barh(xerr=np.column_stack((CI[sorted_features_idx[:22]],
    #                                    CI_b[sorted_features_idx[:22]])).T, 
    #              capsize=2, width=0.75)
    # plt.plot([0, 0], plt.gca().get_ylim(), "k-")
    # plt.xlabel('Shap value')
    # plt.savefig(os.path.join(figure_path, f"attack_{tmp_filename}.pdf"), bbox_inches='tight')
