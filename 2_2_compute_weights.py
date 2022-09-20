""" 
The second part of the attack is to compute the non-uniform weights
over the background dataset D_1. To do so, a MCF is solved to minimize the
amplitude of the SHAP value of a sensitive attribute while ensuring that the
non-uniform distribution remains close to the uniform on D_1
"""
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 21
mp.rcParams['font.family'] = 'serif'

# Local imports
from utils import get_data, get_foreground_background, load_model, SENSITIVE_ATTR
from stealth_sampling import explore_attack


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='communities', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--explainer', type=str, default='tree', help='exact or tree')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    parser.add_argument('--background_size', type=int, default=-1, help='Size of background minibatch, -1 means all')
    parser.add_argument('--min_log', type=float, default=2, help='Min of log space')
    parser.add_argument('--max_log', type=float, default=5, help='Max of log space')
    parser.add_argument('--step_log', type=int, default=40, help='Number of steps in log space')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    D_0, D_1, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)

    # OHE+Ordinally encode B and F
    if ordinal_encoder is not None:
        D_0 = ordinal_encoder.transform(D_0)
        D_1 = ordinal_encoder.transform(D_1)
        # Permute features to match ordinal encoding
        numerical_features = ordinal_encoder.transformers_[0][2]
        categorical_features = ordinal_encoder.transformers_[1][2] 
        features = numerical_features + categorical_features
    if ohe_encoder is not None:
        D_0 = ohe_encoder.transform(D_0)
        D_1 = ohe_encoder.transform(D_1)

    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)

    # All background/foreground predictions
    if args.explainer == "tree" and args.model == "xgb":
        # When explaining Boosted trees with TreeSHAP, we explain the logit
        f_D_0 = model.predict(D_0, output_margin=True).reshape((-1, 1))
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict(D_1, output_margin=True).reshape((-1, 1))
        f_D_1_B = f_D_1[mini_batch_idx]
    else:
        # We explain the probability of class 1
        f_D_0 = model.predict_proba(D_0)[:, [1]]
        f_S_0 = f_D_0[:200]
        f_D_1 = model.predict_proba(D_1)[:, [1]]
        f_D_1_B = f_D_1[mini_batch_idx]

    # Get the sensitive feature
    print(f"Features : {features}")
    print(f"Sensitive attribute : {SENSITIVE_ATTR[args.dataset]}")
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])
    not_s_idx = [i for i in range(n_features) if not i==s_idx]
    print(f"Index of sensitive feature : {s_idx}")

    # Load the Phis
    phis_path = os.path.join("attacks", "Phis")
    tmp_filename = f"{args.explainer}_Phis_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"background_size_{args.background_size}_seed_{args.background_seed}.npy"

    load = np.load(os.path.join(phis_path, tmp_filename))
    minibatch_idx = load[:, 0].astype(int)
    Phi_S0_zj = load[:, 1:]

    init_abs_val = np.abs(Phi_S0_zj[:, s_idx].mean())
    print(f"Abs value before attack : {init_abs_val}")


    significance = 0.01
    lambd_space, weights, biased_shaps, detections = \
                explore_attack(f_D_0, f_S_0, f_D_1_B, Phi_S0_zj, s_idx, 
                               args.min_log, args.max_log, args.step_log, significance)

    # Confidence Intervals CLT
    bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(100)
    # Confidence Intervals for Bernoulli variables
    bandDetec = norm.ppf(0.995) * np.sqrt(detections * (100 - detections)) / 1000


    # Choose the right value of lambda
    undetected_idx = np.where(detections < 10)[0]
    # The lowest absolute value while remaining undetected
    abs_Phi_s = np.abs(biased_shaps[:, :, s_idx].mean(1))
    lowest_abs_value = np.min(abs_Phi_s[undetected_idx])
    best_attack_idx = undetected_idx[np.argmin(abs_Phi_s[undetected_idx])]
    best_lambda = lambd_space[best_attack_idx]
    best_weights = weights[best_attack_idx]
    print(f"Absolute Value after attack : {lowest_abs_value:.5f}")
    success = lowest_abs_value < init_abs_val
    print(f"Reduction factor: {(init_abs_val - lowest_abs_value) / init_abs_val:.2f}")
    print(f"Success of the attack: {success}")

    # # Quantities to characterize the attack
    # weights = []
    # biased_shaps = []
    # biased_ranks = []
    # detections = []

    # # Parameters
    # significance = 0.01
    # lambda_space = np.logspace(args.min_log, args.max_log, args.step_log)


    ############################################################################################
    #                                       Experiment                                         #
    ############################################################################################


    # # Main experiment
    # for lamb in tqdm(lambda_space):
    #     detections.append(0)
    #     biased_shaps.append([])
    #     biased_ranks.append([])

    #     weights.append(attack_SHAP(f_D_1_B, -Phi_S_0_zj[:, s_idx], lamb))
    #     print(f"Sparsity of weights : {np.mean(weights[-1] == 0) * 100}%")

    #     # Repeat detection experiment
    #     for _ in range(100):
    #         biased_idx = np.random.choice(len(mini_batch_idx), 200, p=weights[-1]/np.sum(weights[-1]))
    #         f_S_1 = f_D_1_B[biased_idx]
            
    #         # New shap values
    #         shap = np.mean(Phi_S_0_zj[biased_idx], axis=0)
    #         biased_shaps[-1].append(shap)
    #         biased_ranks[-1].append(rankdata(shap)[s_idx])

    #         detections[-1] += audit_detection(f_D_0, f_D_1, 
    #                                           f_S_0, f_S_1, significance)



    # # Convert to arrays for plots
    # weights = np.array(weights)
    # biased_shaps = np.array(biased_shaps)
    # biased_ranks = np.array(biased_ranks)
    # detections  = np.array(detections)
    
    ############################################################################################
    #                                     Save Results                                         #
    ############################################################################################


    # Save the weights if success
    if success:
        weights_path = os.path.join("attacks", "Weights")
        tmp_filename = f"{args.explainer}_Weights_{args.model}_{args.dataset}_rseed_{args.rseed}_"
        tmp_filename += f"B_size_{args.background_size}_seed_{args.background_seed}"
        #print(best_weights.shape)
        #print(mini_batch_idx.shape)
        np.save(os.path.join(weights_path, tmp_filename), np.column_stack((mini_batch_idx, best_weights)) )

    # Where to save figures
    figure_path = os.path.join(f"Images/{args.dataset}/{args.model}/")
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    tmp_filename = f"rseed_{args.rseed}_B_size_{args.background_size}_seed_{args.background_seed}"


    # Curves of Shapley values
    plt.figure()
    # Sensitive attribute
    plt.plot(lambd_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive feature")
    plt.fill_between(lambd_space, biased_shaps.mean(1)[:, s_idx] + bandSHAP[:, s_idx], 
                        biased_shaps.mean(1)[:, s_idx] - bandSHAP[:, s_idx], color='r', alpha=0.2)
    # Other attributes
    lines = plt.plot(lambd_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
    plt.setp(lines[1:], label="_") 
    for i in not_s_idx:
        plt.fill_between(lambd_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                    biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
    plt.plot(best_lambda * np.ones(2), [biased_shaps.min(), biased_shaps.max()], 'k--')
    plt.text(best_lambda, biased_shaps.min(), r"$\lambda^\star$", ha='left', va='bottom')
    plt.xlim(lambd_space.min(), lambd_space.max())
    plt.ylim(biased_shaps.min(), biased_shaps.max())
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel("GSV")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join(figure_path, 
                f"shapley_curve_{tmp_filename}.pdf"), bbox_inches='tight')


    # Curves of the detection rates of both tests
    plt.figure()
    plt.plot(lambd_space, detections, 'b-')
    plt.fill_between(lambd_space, detections + bandDetec, detections - bandDetec,
                                                            color='b', alpha=0.2)
    plt.plot(lambd_space, 10 * np.ones(lambd_space.shape), 'k--')
    plt.plot(best_lambda * np.ones(2), [0, 101], 'k--')
    plt.text(best_lambda, plt.gca().get_ylim()[0], r"$\lambda^\star$", ha='center', va='bottom')
    plt.xlim(lambd_space.min(), lambd_space.max())
    plt.ylim(plt.gca().get_ylim()[0], 101)
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel(r"Detection Rate $(\%)$")
    plt.savefig(os.path.join(figure_path, 
                f"detection_{tmp_filename}.pdf"), bbox_inches='tight')
