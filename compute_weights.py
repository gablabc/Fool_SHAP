import argparse
import numpy as np
import os
from stealth_sampling import attack_SHAP
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm, ks_2samp, rankdata

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 12
mp.rcParams['font.family'] = 'serif'

# Local imports
from utils import get_data, get_foreground_background, load_model
from utils import SENSITIVE_ATTR


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--background_seed', type=int, default=0, help='Seed of background minibatch')
    parser.add_argument('--background_size', type=int, default=200, help='Size of background minibatch')
    parser.add_argument('--min_log', type=float, default=-4, help='Min of log space')
    parser.add_argument('--max_log', type=float, default=0, help='Max of log space')
    parser.add_argument('--step_log', type=float, default=40, help='Number of steps in log space')
    parser.add_argument('--attack', type=str, default="value_diff", help='Attack "value" or "value_diff"')
    parser.add_argument('--fool', type=str, default="KS", help='Fool which test? KS or WaldF')
    args = parser.parse_args()

    # Get the data
    filepath = os.path.join("datasets", "preprocessed", args.dataset)
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                        get_data(args.dataset, args.model, rseed=args.rseed)

    # Get B and F
    foreground, background, mini_batch_idx = get_foreground_background(X_split, args.dataset,
                                                    args.background_size, args.background_seed)
    #print(foreground.iloc[0])
    # OHE+Ordinally encode B and F
    background = ohe_encoder.transform(ordinal_encoder.transform(background))
    # Permute features to match ordinal encoding
    numerical_features = ordinal_encoder.transformers_[0][2]
    categorical_features = ordinal_encoder.transformers_[1][2] 
    features = numerical_features + categorical_features
    n_features = len(features)

    # Load the model
    tmp_filename = f"{args.model}_{args.dataset}_{args.rseed}"
    model = load_model(args.model, "models", tmp_filename)
    
    # Statistics of prediction over background
    mean_B_pred = model.predict_proba(background)[:, 1].mean()
    std_B_pred  = model.predict_proba(background)[:, 1].std()

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
    shap = Phis.mean(0)
    shap_var = Phis.std(0)
    #print(shap)
    #print(shap_var)
    ranks = rankdata(shap)
    init_rank = ranks[s_idx]
    print(f"Rank before attack : {init_rank}")
    #print(ranks)


    # Quantities to characterize the attack
    weights = []
    parity = []
    biased_parity = []
    biased_shaps = []
    biased_ranks = []
    biased_means = []
    detections_KS = []
    detections_Wald = []

    # parameters
    significance = 0.01
    n_repetitions = 200
    lamb_space = np.logspace(args.min_log, args.max_log, args.step_log)


    ############################################################################################
    #                                       Experiment                                         #
    ############################################################################################


    # Main experiment
    for lamb in tqdm(lamb_space):
        detections_KS.append(0)
        detections_Wald.append(0)
        parity.append([])
        biased_parity.append([])
        biased_shaps.append([])
        biased_ranks.append([])
        biased_means.append([])

        # Fool Wald or KS?
        if args.fool == "Wald":
            # compute biased weights
            ref = model.predict_proba(background)[:, [1]]
        elif args.fool == "KS":
            ref = background
        else:
            raise NotImplementedError()

        # Attack the SHAP value or its difference with others?
        if args.attack == "value":
            objective = -1 * Phis[:, s_idx]
        elif args.attack == "value_diff":
            candidates = np.where((shap < 0) & (ranks > init_rank))[0]
            features_to_perturb = candidates
            std = Phis[:, features_to_perturb].std(0)
            objective = np.average(Phis[:, features_to_perturb] - Phis[:, [s_idx]], axis=1, weights=std)
        # Attack !!!
        weights.append(attack_SHAP(ref, objective, lamb))
        print(f"Spasity of weights : {np.mean(weights[-1] == 0) * 100}%")


        # Repeat detection experiment
        for _ in range(n_repetitions):
            # Biased sampling
            uniform_idx = np.random.choice(args.background_size, 100)
            unbiased_samples = background[uniform_idx, :]
            biased_idx = np.random.choice(args.background_size, 100, p=weights[-1]/np.sum(weights[-1]))
            biased_samples = background[biased_idx, :]

            # Biased demographic parity
            parity[-1].append(np.sum(Phis[uniform_idx, :].mean(0)))
            biased_parity[-1].append(np.sum(Phis[biased_idx, :].mean(0)))
            
            # New shap values
            shap = np.mean(Phis[biased_idx], axis=0)
            biased_shaps[-1].append(shap)
            biased_ranks[-1].append(rankdata(shap)[s_idx])

            # Do one KS test per feature and apply Bonferri correction
            for f in range(len(numerical_features)):
                # Test
                _, p_val = ks_2samp(biased_samples[:, f], unbiased_samples[:, f])
                if p_val < significance / len(numerical_features):
                    detections_KS[-1] += 1
                    break
            
            ## Compute mean of E_B'[f]
            biased_means[-1].append(model.predict_proba(biased_samples)[:, 1].mean())
        

        # Count the detection rate
        detections_KS[-1] *= 100 / n_repetitions
        # Wald statistic
        W = np.sqrt(100) * (np.array(biased_means[-1]) - mean_B_pred) / std_B_pred
        # Wald detection
        detections_Wald[-1] = 100 * np.mean(np.abs(W) > -norm.ppf(significance/2) )


    # Convert to arrays for plots
    weights = np.array(weights)
    parity = np.array(parity)
    biased_parity = np.array(biased_parity)
    biased_shaps = np.array(biased_shaps)
    biased_ranks = np.array(biased_ranks)

    detections_KS  = np.array(detections_KS)
    detections_Wald = np.array(detections_Wald)

    # Confidence Intervals CLT
    bandpar  = norm.ppf(0.995) * np.std(parity, axis=1) / np.sqrt(n_repetitions)
    bandbpar = norm.ppf(0.995) * np.std(biased_parity, axis=1) / np.sqrt(n_repetitions)
    bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(n_repetitions)
    
    # Confidence Intervals for Bernoulli variables
    bandKS = np.sqrt(detections_KS * (100 - detections_KS) / n_repetitions)
    bandW  = np.sqrt(detections_Wald * (100 - detections_Wald) / n_repetitions)
    

    # Choose the right value of lambda
    detection = detections_KS if args.fool == "KS" else detections_Wald
    undetected_idx = np.where(detection < 10)[0]
    # The hightest rank while remaining undetected
    best_attack_rank = np.max(biased_ranks[undetected_idx].mean(1))
    best_attack_idx = undetected_idx[np.argmax(biased_ranks[undetected_idx].mean(1))]
    best_lambda = lamb_space[best_attack_idx]
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
        tmp_filename += f"_fool_{args.fool}.npy"
        print(best_weights.shape)
        print(mini_batch_idx.shape)
        np.save(os.path.join(weights_path, tmp_filename), np.column_stack((mini_batch_idx, best_weights)) )

    # Where to save figures
    figure_path = os.path.join(f"Images/{args.dataset}/{args.model}/")
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    tmp_filename = f"rseed_{args.rseed}_B_size_{args.background_size}_seed_{args.background_seed}"


    #plt.figure()
    #plt.plot(lamb_space, parity.mean(1), 'r-')
    #plt.fill_between(lamb_space, parity.mean(1) + bandpar, 
    #                             parity.mean(1) - bandpar, color='r', alpha=0.2)
    #plt.plot(lamb_space, biased_parity.mean(1), 'b-')
    #plt.fill_between(lamb_space, biased_parity.mean(1) + bandbpar, 
    #                             biased_parity.mean(1) - bandbpar, color='b', alpha=0.2)
    #plt.xlabel(r"$\lambda$")
    #plt.xscale('log')
    #plt.ylabel("Parity")
    #plt.savefig(os.path.join("Images", f"parity_curve_{filename}.pdf"), bbox_inches='tight')


    # Curves of Shapley values
    plt.figure()
    plt.plot(lamb_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive feature")
    lines = plt.plot(lamb_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
    plt.setp(lines[1:], label="_") 
    plt.fill_between(lamb_space, biased_shaps.mean(1)[:, s_idx] + bandSHAP[:, s_idx], 
                                 biased_shaps.mean(1)[:, s_idx] - bandSHAP[:, s_idx], color='r', alpha=0.2)
    for i in not_s_idx:
        plt.fill_between(lamb_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                     biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
    plt.plot(best_lambda * np.ones(2), [biased_shaps.min(), biased_shaps.max()], 'k--')
    plt.text(best_lambda, biased_shaps.min(), r"$\lambda^\star$", ha='left', va='bottom')
    plt.xlim(lamb_space.min(), lamb_space.max())
    plt.ylim(biased_shaps.min(), biased_shaps.max())
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel("Global Shapley values")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join(figure_path, 
                f"shapley_curve_{tmp_filename}_fool_{args.fool}_attack_{args.attack}.pdf"), bbox_inches='tight')


    # Curves of the detection rates of both tests
    plt.figure()
    plt.plot(lamb_space, detections_KS, 'r-', label="KS")
    plt.fill_between(lamb_space, detections_KS + bandKS, detections_KS - bandKS,
                                                            color='r', alpha=0.2)
    plt.plot(lamb_space, detections_Wald, 'b-', label="Wald")
    plt.fill_between(lamb_space, detections_Wald + bandW, detections_Wald - bandW,
                                                            color='b', alpha=0.2)
    plt.plot(lamb_space, 10 * np.ones(lamb_space.shape), 'k--')
    plt.plot(best_lambda * np.ones(2), [0, 101], 'k--')
    plt.text(best_lambda, plt.gca().get_ylim()[0], r"$\lambda^\star$", ha='center', va='bottom')
    plt.xlim(lamb_space.min(), lamb_space.max())
    plt.ylim(plt.gca().get_ylim()[0], 101)
    plt.xlabel(r"$\lambda$")
    plt.xscale('log')
    plt.ylabel(r"Detection Rate $(\%)$")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join(figure_path, 
                f"detection_{tmp_filename}_fool_{args.fool}_attack_{args.attack}.pdf"), bbox_inches='tight')
    