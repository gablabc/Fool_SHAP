import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm, rankdata

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 21
mp.rcParams['font.family'] = 'serif'

# Local imports
from utils import get_data, get_foreground_background, load_model
from utils import audit_detection, SENSITIVE_ATTR
from stealth_sampling import attack_SHAP


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    parser.add_argument('--background_size', type=int, default=-1, help='Size of background minibatch')
    parser.add_argument('--min_log', type=float, default=-0.1, help='Min of log space')
    parser.add_argument('--max_log', type=float, default=1, help='Max of log space')
    parser.add_argument('--step_log', type=int, default=40, help='Number of steps in log space')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    foreground, background, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)

    # OHE+Ordinally encode B and F
    background = ohe_encoder.transform(ordinal_encoder.transform(background))
    foreground = ohe_encoder.transform(ordinal_encoder.transform(foreground))

    # Permute features to match ordinal encoding
    numerical_features = ordinal_encoder.transformers_[0][2]
    categorical_features = ordinal_encoder.transformers_[1][2] 
    features = numerical_features + categorical_features
    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)

    # All background/foreground predictions
    b_pred = model.predict_proba(background)[:, [1]]
    f_pred = model.predict_proba(foreground)[:, [1]]
    subset_f_pred = f_pred[:200]

    # Get the sensitive feature
    print(f"Features : {features}")
    print(f"Sensitive attribute : {SENSITIVE_ATTR[args.dataset]}")
    s_idx = features.index(SENSITIVE_ATTR[args.dataset])
    not_s_idx = [i for i in range(n_features) if not i==s_idx]
    print(f"Index of sensitive feature : {s_idx}")

    # Load the Phis
    phis_path = os.path.join("attacks", "Phis")
    tmp_filename = f"Phis_{args.model}_{args.dataset}_rseed_{args.rseed}_"
    tmp_filename += f"background_size_{args.background_size}_seed_{args.background_seed}.npy"

    load = np.load(os.path.join(phis_path, tmp_filename))
    minibatch_idx = load[:, 0].astype(int)
    Phis = load[:, 1:]
    init_rank = rankdata(Phis.mean(0))[s_idx]
    print(f"Rank before attack : {init_rank}")


    # Quantities to characterize the attack
    weights = []
    biased_shaps = []
    biased_ranks = []
    detections = []

    # parameters
    significance = 0.01
    n_repetitions = 200
    lambda_space = np.logspace(args.min_log, args.max_log, args.step_log)


    ############################################################################################
    #                                       Experiment                                         #
    ############################################################################################


    # Main experiment
    for lamb in tqdm(lambda_space):
        detections.append(0)
        biased_shaps.append([])
        biased_ranks.append([])

        # Attack !!!
        weights.append(attack_SHAP(b_pred[mini_batch_idx],
                                   -1 * Phis[:, s_idx], lamb))
        print(f"Spasity of weights : {np.mean(weights[-1] == 0) * 100}%")


        # Repeat detection experiment
        for _ in range(100):
            biased_idx = np.random.choice(len(mini_batch_idx), 200, p=weights[-1]/np.sum(weights[-1]))
            subset_b_pred = b_pred[mini_batch_idx[biased_idx]]
            
            # New shap values
            shap = np.mean(Phis[biased_idx], axis=0)
            biased_shaps[-1].append(shap)
            biased_ranks[-1].append(rankdata(shap)[s_idx])

            detections[-1] += audit_detection(b_pred, f_pred, 
                                              subset_b_pred, subset_f_pred, significance)



    # Convert to arrays for plots
    weights = np.array(weights)
    biased_shaps = np.array(biased_shaps)
    biased_ranks = np.array(biased_ranks)
    detections  = np.array(detections)

    # Confidence Intervals CLT
    bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(100)
    # Confidence Intervals for Bernoulli variables
    bandDetec = np.sqrt(detections * (100 - detections) / 100)


    # Choose the right value of lambda
    undetected_idx = np.where(detections < 10)[0]
    # The hightest rank while remaining undetected
    best_attack_rank = np.max(biased_ranks[undetected_idx].mean(1))
    best_attack_idx = undetected_idx[np.argmax(biased_ranks[undetected_idx].mean(1))]
    best_lambda = lambda_space[best_attack_idx]
    best_weights = weights[best_attack_idx]
    print(f"Mean Rank after attack : {best_attack_rank}")
    success = np.round(best_attack_rank) > init_rank
    print(f"Success of the attack: {success}")
    
    ############################################################################################
    #                                     Save Results                                         #
    ############################################################################################


    # Save the weights if success
    if success:
        weights_path = os.path.join("attacks", "Weights")
        tmp_filename = f"Weights_{args.model}_{args.dataset}_rseed_{args.rseed}_"
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
    plt.plot(lambda_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive feature")
    lines = plt.plot(lambda_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
    plt.setp(lines[1:], label="_") 
    plt.fill_between(lambda_space, biased_shaps.mean(1)[:, s_idx] + bandSHAP[:, s_idx], 
                                   biased_shaps.mean(1)[:, s_idx] - bandSHAP[:, s_idx], color='r', alpha=0.2)
    for i in not_s_idx:
        plt.fill_between(lambda_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                     biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
    plt.plot(best_lambda * np.ones(2), [biased_shaps.min(), biased_shaps.max()], 'k--')
    plt.text(best_lambda, biased_shaps.min(), r"$\lambda^\star$", ha='left', va='bottom')
    plt.xlim(lambda_space.min(), lambda_space.max())
    plt.ylim(biased_shaps.min(), biased_shaps.max())
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel("GSV")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join(figure_path, 
                f"shapley_curve_{tmp_filename}.pdf"), bbox_inches='tight')


    # Curves of the detection rates of both tests
    plt.figure()
    plt.plot(lambda_space, detections, 'b-')
    plt.fill_between(lambda_space, detections + bandDetec, detections - bandDetec,
                                                            color='b', alpha=0.2)
    plt.plot(lambda_space, 10 * np.ones(lambda_space.shape), 'k--')
    plt.plot(best_lambda * np.ones(2), [0, 101], 'k--')
    plt.text(best_lambda, plt.gca().get_ylim()[0], r"$\lambda^\star$", ha='center', va='bottom')
    plt.xlim(lambda_space.min(), lambda_space.max())
    plt.ylim(plt.gca().get_ylim()[0], 101)
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel(r"Detection Rate $(\%)$")
    plt.savefig(os.path.join(figure_path, 
                f"detection_{tmp_filename}.pdf"), bbox_inches='tight')
