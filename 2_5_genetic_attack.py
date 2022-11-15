""" 

"""
import argparse
from functools import partial
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import xgboost

import sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
import shap

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 21
mp.rcParams['font.family'] = 'serif'

# Local imports
import genetic
from utils import audit_detection


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Bob')
    parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
    parser.add_argument('--model', type=str, default="rf", help='Model type')
    args = parser.parse_args()

    X,y = shap.datasets.adult()
    X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
                "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
                "HoursPerWeek", "Country"]
    features = X.columns
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.rseed)
    if args.model == "rf":
        model = RandomForestClassifier(random_state=10, n_estimators=50, max_depth=5, min_samples_leaf=50)
    elif args.model == "xgb":
        y = y.astype(int)
        model = xgboost.XGBClassifier(random_state=0, eval_metric="error", use_label_encoder=False)
    else:
        raise NotImplementedError()
    model.fit(X_train, y_train)

    # Reference datasets
    M = 100
    D_0 = shuffle(X[X["Sex"]==0], random_state=args.rseed)
    f_D_0 = model.predict_proba(D_0)[:, [1]]
    S_0 = D_0.iloc[:M]
    f_S_0 = f_D_0[:M]
    D_1 = shuffle(X[X["Sex"]==1], random_state=args.rseed)
    f_D_1 = model.predict_proba(D_1)[:, [1]]
    S_1 = D_1.iloc[:M]

    n_features = len(features)

    # Get the sensitive feature
    s_idx = 7

    # Plot setup
    hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
    file_path = os.path.join("Images", "adult_income", args.model)
    tmp_filename = f"genetic_rseed_{args.rseed}"

    # Shap explainer wrapper
    explainer = genetic.Explainer(model)
    # Wrapper for the detector
    def detector_wrapper(S_1, f_S_0, f_D_0, f_D_1, model):
        f_S_1 = model.predict_proba(S_1)[:, [1]]
        return audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, significance=0.01)
    detector = partial(detector_wrapper, f_S_0=f_S_0, f_D_0=f_D_0, f_D_1=f_D_1, model=model)

    # Peturbate the background
    constant_features = ["Workclass", "MaritalStatus", "Occupation", "Relationship", "Race", 
                         "Sex", "Country"]
    alg = genetic.GeneticAlgorithm(explainer, S_0, S_1, s_idx, detector=detector, 
                                    constant=constant_features, pop_count=25)
    
    detection = 0
    for i in range(1, 3):
        alg.fool_aim(max_iter=50, random_state=0)
        S_1_prime = alg.S_1_prime
        f_S_1 = model.predict_proba(S_1_prime)[:, [1]]

        hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
        plt.figure()
        plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
        plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
        plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
        plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
        plt.xlabel("Output")
        plt.ylabel("CDF")
        plt.legend(framealpha=1, loc="lower right")
        plt.savefig(os.path.join(file_path, tmp_filename + f"_ite_{50*i:d}.pdf"), bbox_inches='tight')

    # Save logs
    results_file = os.path.join("attacks", "Genetic", f"{args.model}_" + tmp_filename + ".csv")
    pd.DataFrame(alg.iter_log).to_csv(results_file, index=False)
    
