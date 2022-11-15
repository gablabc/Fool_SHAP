import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os
import subprocess
from scipy.stats import norm, ks_2samp

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from shap.explainers import Tree
from shap.maskers import Independent

# Import for Construct Defect Models (Classification)
from sklearn.ensemble import RandomForestClassifier # Random Forests
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosted Trees (GBT)
import xgboost as xgb # eXtreme Gradient Boosting Tree (xGBTree)


def get_encoders(df_X, model_name):
    """ Fit a ordinal and ohe encoders on the whole dataset """

    # Categorical features ?
    is_cat = np.array([dt.kind == 'O' for dt in df_X.dtypes])
    cat_cols = list(df_X.columns.values[is_cat])
    num_cols = list(df_X.columns.values[~is_cat])

    # Ordinal encoding is required for SHAP
    if not len(cat_cols) == 0:
        ordinal_encoder = \
                    ColumnTransformer([
                                ('identity', FunctionTransformer(), num_cols),
                                ('ordinal', OrdinalEncoder(), cat_cols)]
                    ).fit(df_X)
        X = ordinal_encoder.transform(df_X)
        # Reorganize num_cols cat_cols order
        n_num = len(num_cols)
        num_cols = list(range(n_num))
        cat_cols = [i + n_num for i in range(len(cat_cols))]
    else:
        ordinal_encoder = None
        X = df_X

    # Some models require rescaling numerical features
    if model_name == "mlp":
        scaler = StandardScaler()
    # Otherwise Identity map
    else:
        scaler = FunctionTransformer()
    
    # One Hot Encode features
    if not len(cat_cols) == 0:
        ohe = OneHotEncoder(sparse=False)
    # Or not ...
    else:
        ohe = FunctionTransformer()

    ohe_preprocessor = ColumnTransformer([
                                    ('scaler', scaler, num_cols),
                                    ('ohe', ohe, cat_cols)]).fit(X)
    
    return ordinal_encoder, ohe_preprocessor


def get_data(dataset, model_name, rseed):
    """ Load the data, split it, and get encoders """
    # Get the data
    filepath = os.path.join("datasets", "preprocessed")
    # Dataset
    df = pd.read_csv(os.path.join(filepath, f"{dataset}.csv"))
    # Split indices
    split_dict = json.load(open(os.path.join(filepath, f"{dataset}_split_rseed_{rseed}.json")))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    features = list(X.columns)
    ordinal_encoder, ohe_encoder = get_encoders(X, model_name)
    
    # Splits
    X_split = {key : X.iloc[split_dict[key]].reset_index(drop=True) for key in ["train", "test"]}
    y_split = {key : y.iloc[split_dict[key]].reset_index(drop=True) for key in ["train", "test"]}
    
    return X_split, y_split, features, ordinal_encoder, ohe_encoder


SENSITIVE_ATTR = {
    'adult_income' : 'gender',
    'compas' : 'race',
    'default_credit' : 'SEX',
    'marketing' : 'age',
    'communities': 'PctWhite>90'
}

PROTECTED_CLASS = {
    'adult_income' : 'Female',
    'compas' : 'African-American',
    'default_credit' : 'Female',
    'marketing' : 'age:30-60',
    'communities': 0
}


def get_foreground_background(X_split, dataset, background_size=None, background_seed=None):
    """ 
    Load foreground and background distributions (F and B in the paper) based
    on the sensitive attribute
    """
    # Training set is the first index
    X = pd.concat([X_split["train"], X_split["test"]])

    if background_seed is not None:
        np.random.seed(background_seed)
    # Subsample a portion of the Background
    background = X.loc[X[SENSITIVE_ATTR[dataset]] != PROTECTED_CLASS[dataset]]
    foreground = X.loc[X[SENSITIVE_ATTR[dataset]] == PROTECTED_CLASS[dataset]]
    #print(background.shape)
    
    # Dont return a minibatch
    if background_size == None:
        return foreground, background
    # Minibatch is all of background
    if background_size == -1:
        mini_batch_idx = np.arange(len(background)).astype(int)
    # Minibatch is subset of background
    else:
        mini_batch_idx = np.random.choice(range(background.shape[0]), background_size)
    return foreground, background, mini_batch_idx



###################################################################################



MODELS = { 
    'mlp' : MLPClassifier(random_state=1234, max_iter=500, early_stopping=True),
    'rf' : RandomForestClassifier(random_state=1234, n_jobs=-1),
    'gbt' : GradientBoostingClassifier(random_state=1234),
    'xgb' : xgb.XGBClassifier(random_state=1234, eval_metric='error', use_label_encoder=False)
}


def save_model(model_name, model, path, filename):
    if model_name == "xgb":
        # save in JSON format
        model.save_model(os.path.join(path, f"{filename}.json"))
    else:
        # Pickle the model
        from joblib import dump
        dump(model, os.path.join(path, f"{filename}.joblib"))


def load_model(model_name, path, filename):
    if model_name == "xgb":
        # Load the JSON format
        model = MODELS["xgb"]
        model.load_model(os.path.join(path, f"{filename}.json"))
    else:
        # Un-Pickle the model
        from joblib import load
        model = load(os.path.join(path, f"{filename}.joblib"))
    return model


def get_hp_grid(filename):
    """ Get the gp_grid from a json file """
    def to_eval(string):
        if type(string) == str:
            split = string.split("_")
            if len(split) == 2:
                return split[1]
            else:
                return None
        else:
            return None

    hp_dict = json.load(open(filename, "r"))
    for key, value in hp_dict.items():
        # Iterate over list
        if type(value) == list:
            for i, element in enumerate(value):
                str_to_eval = to_eval(element)
                if str_to_eval is not None:
                    value[i] = eval(str_to_eval)
        # Must be evaluated
        if type(value) == str:
            str_to_eval = to_eval(value)
            if str_to_eval is not None:
                hp_dict[key] = eval(str_to_eval)
    return hp_dict


def get_best_cv_scores(search, k):
    res = search.cv_results_
    # Get the top-1 hyperparameters
    top_one_idx = np.where(res["rank_test_score"]==1)[0]
    perfs = np.zeros(k)
    for i in range(k):
        perfs[i] = res[f"split{i}_test_score"][top_one_idx]
    return perfs


def get_best_cv_model(X, y, estimator, param_grid, cross_validator, n_iter, n_jobs):
    # Try out default HP
    best_cv_scores = cross_val_score(estimator, X, y, cv=cross_validator, scoring='roc_auc')
    print("bob")

    # Try out Random Search
    hp_search = RandomizedSearchCV(estimator, param_grid, scoring='roc_auc', n_jobs=n_jobs,
                                   cv=cross_validator, n_iter=n_iter, random_state=42, verbose=2)
    hp_search.fit(X, y)

    if np.max(hp_search.cv_results_['mean_test_score']) - 0.0005 > best_cv_scores.mean():
        print("Use fine-tuned values")
        best_cv_scores =  get_best_cv_scores(hp_search, cross_validator.n_splits)
        model = hp_search.best_estimator_
    else:
        print("Use default values")
        model = estimator.fit(X, y)
    return model, best_cv_scores


def multidim_KS(sample1, sample2, significance):
    detection = 0
    # Do one KS test per feature and apply Bonferri correction
    n_features = sample1.shape[1]
    for f in range(n_features):
        # Compute KS test
        _, p_val = ks_2samp(sample1[:, f], sample2[:, f])
        if p_val < significance / n_features:
            detection = 1
            break
    return detection


def audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, significance):
    # Tests conducted by the audit to see if the background and foreground provided
    # are indeed subsampled uniformly from the data.
    for distribution, samples in zip([f_D_0, f_D_1],
                                     [f_S_0, f_S_1]):
        n_samples = len(samples)
        # KS test
        for _ in range(10):
            unbiased_preds = distribution[np.random.choice(len(distribution), n_samples)]
            _, p_val = ks_2samp(samples.ravel(), unbiased_preds.ravel())
            if p_val < significance / 40:
                return 1

        # Wald test
        W = np.sqrt(n_samples) * (samples.mean() - distribution.mean()) / distribution.std()
        # Wald detection
        if np.abs(W) > -norm.ppf(significance/8):
            return 1
    return 0


def confidence_interval(LSV, significance):
    #assert LSV.shape[1] == LSV.shape[2]
    M = LSV.shape[1]
    alpha = norm.ppf(1 - significance/2)
    sigma = np.sqrt(0.5 * (np.var(np.mean(LSV, axis=1), axis=1) + \
                           np.var(np.mean(LSV, axis=2), axis=1)))
    return alpha * sigma / np.sqrt(2 * M)


def tree_shap(model, S_0, S_1, ordinal_encoder=None, ohe_encoder=None):
    
    # Find out which ohe columns correspond to which feature
    if ordinal_encoder is not None and ohe_encoder is not None:
        n_num_features = len(ordinal_encoder.transformers_[0][2])
        categorical_to_features = list(range(n_num_features))
        counter = n_num_features
        for idx in range(len(ordinal_encoder.transformers_[1][2])):
            # Associate the feature to its encoding columns
            for _ in ordinal_encoder.transformers_[1][1].categories_[idx]:
                categorical_to_features.append(idx + n_num_features)
                counter = counter + 1

        S_0 = ohe_encoder.transform(S_0)
        S_1 = ohe_encoder.transform(S_1)
    else:
        categorical_to_features = list(range(S_0.shape[1]))

    n_features = categorical_to_features[-1] + 1
    
    mask = Independent(S_1, max_samples=len(S_1))
    ensemble = Tree(model, data=mask).model


    # Save tree ensemble
    np.savetxt("categorical_to_features.txt", categorical_to_features, fmt="%d")
    np.savetxt("feature.txt", ensemble.features, fmt="%d")
    np.savetxt("value.txt", ensemble.values[..., -1], fmt="%.20e")
    np.savetxt("threshold.txt", ensemble.thresholds, fmt="%.20e")
    np.savetxt("left.txt", ensemble.children_left, fmt="%d")
    np.savetxt("right.txt", ensemble.children_right, fmt="%d")

    # Save data
    pd.DataFrame(S_1).to_csv("background.txt", sep=' ', index=False, header=False)
    pd.DataFrame(S_0).to_csv("foreground.txt", sep=' ', index=False, header=False)

    cmd = f"./tree_shap/main > output.txt"
    proc = subprocess.Popen('exec ' + cmd, shell=True)
    proc.wait()
    LSV = np.loadtxt('output.txt')

    # Clean up
    os.remove("background.txt")
    os.remove("foreground.txt")
    os.remove("categorical_to_features.txt")
    os.remove("feature.txt")
    os.remove("value.txt")
    os.remove("threshold.txt")
    os.remove("left.txt")
    os.remove("right.txt")
    os.remove("output.txt")

    return LSV.reshape((n_features, len(S_0), -1))



def plot_CDFs(f_D_0, f_D_1, f_S_0, f_S_1, legend_loc="lower right"):
    hist_kwargs = {'cumulative':True, 'histtype':'step', 'density':True}
    plt.figure()
    plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_kwargs)
    plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_kwargs)
    plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_kwargs)
    plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_kwargs)
    plt.xlabel("Output")
    plt.ylabel("CDF")
    plt.legend(framealpha=1, loc=legend_loc)


if __name__ == "__main__":
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                get_data("default_credit", "rf", 0)
    print("Done")