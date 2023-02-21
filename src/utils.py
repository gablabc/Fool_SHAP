import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import ctypes
import glob
from scipy.stats import norm, ks_2samp

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from shap.explainers import Tree
from shap.maskers import Independent

# Import for Construct Defect Models (Classification)
from sklearn.ensemble import RandomForestClassifier  # Random Forests
from sklearn.neural_network import MLPClassifier  # Neural Network
# Gradient Boosted Trees (GBT)
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb  # eXtreme Gradient Boosting Tree (xGBTree)


class DummyEncoder(TransformerMixin):
    """ 
    A dummy ordinal encoder that simply maps a 
    DataFrame to a C-Contiguous numpy array 
    """
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return np.ascontiguousarray(X)



def get_encoders(df_X, model_name):
    """ 
    Fit a ordinal and ohe encoders on the whole dataset 
    
    Any cat features?
        Will return two encoders
    
    No cat features?
        Will return a ordinal_encoder that simply
    
    """

    # Categorical features ?
    is_cat = np.array([dt.kind == 'O' for dt in df_X.dtypes])
    cat_cols = list(df_X.columns.values[is_cat])
    num_cols = list(df_X.columns.values[~is_cat])

    ###### Ordinal Encoding ######
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
        ordinal_encoder = DummyEncoder()
        X = ordinal_encoder.fit_transform(df_X)
    # At this point X is always a contiguous np.array
    assert type(X) == np.ndarray
    assert X.flags['C_CONTIGUOUS']


    ###### One-Hot-Encoding ######
    unused = 0
    # Some models require rescaling numerical features
    if model_name == "mlp":
        scaler = StandardScaler()
    # Otherwise Identity map
    else:
        scaler = FunctionTransformer()
        unused += 1

    # Any categorical features
    if not len(cat_cols) == 0:
        ohe = OneHotEncoder(sparse=False)
    # Or not ...
    else:
        ohe = FunctionTransformer()
        unused += 1

    if unused < 2:
        ohe_preprocessor = ColumnTransformer([
                    ('scaler', scaler, num_cols),
                    ('ohe', ohe, cat_cols)]).fit(X)
    else:
        ohe_preprocessor = None

    return ordinal_encoder, ohe_preprocessor



def get_data(dataset, model_name, rseed):
    """ Load the data, split it, and get encoders """
    # Get the data
    filepath = os.path.join("datasets", "preprocessed")
    # Dataset
    df = pd.read_csv(os.path.join(filepath, f"{dataset}.csv"))
    # Split indices
    split_dict = json.load(
        open(os.path.join(filepath, f"{dataset}_split_rseed_{rseed}.json")))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    features = list(X.columns)
    ordinal_encoder, ohe_encoder = get_encoders(X, model_name)

    # Splits
    X_split = {key: X.iloc[split_dict[key]].reset_index(drop=True) for key in [
                    "train", "test"]}
    y_split = {key: y.iloc[split_dict[key]].reset_index(drop=True) for key in [
                    "train", "test"]}

    return X_split, y_split, features, ordinal_encoder, ohe_encoder



SENSITIVE_ATTR = {
    'adult_income': 'gender',
    'compas': 'race',
    'default_credit': 'SEX',
    'marketing': 'age',
    'communities': 'PctWhite>90'
}

PROTECTED_CLASS = {
    'adult_income': 'Female',
    'compas': 'African-American',
    'default_credit': 'Female',
    'marketing': 'age:30-60',
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
    # print(background.shape)

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
    'mlp': MLPClassifier(random_state=1234, max_iter=500, early_stopping=True),
    'rf': RandomForestClassifier(random_state=1234, n_jobs=-1),
    'gbt': GradientBoostingClassifier(random_state=1234),
    'xgb': xgb.XGBClassifier(random_state=1234, eval_metric='error', use_label_encoder=False)
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
    top_one_idx = np.where(res["rank_test_score"] == 1)[0]
    perfs = np.zeros(k)
    for i in range(k):
        perfs[i] = res[f"split{i}_test_score"][top_one_idx]
    return perfs


def get_best_cv_model(X, y, estimator, param_grid, cross_validator, n_iter, n_jobs):
    # Try out default HP
    best_cv_scores = cross_val_score(estimator, X, y, cv=cross_validator, scoring='roc_auc')
    print("Done Evaluating Default Estimator")

    # Try out Random Search
    hp_search = RandomizedSearchCV(estimator, param_grid, scoring='roc_auc', n_jobs=n_jobs,
                                   cv=cross_validator, n_iter=n_iter, random_state=42, verbose=2)
    hp_search.fit(X, y)

    if np.max(hp_search.cv_results_['mean_test_score']) - 0.0005 > best_cv_scores.mean():
        print("Use fine-tuned values")
        best_cv_scores = get_best_cv_scores(hp_search, cross_validator.n_splits)
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



def audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, significance=0.05):
    """
    Statistical Test conducted by the Audit to see if the background and foreground 
    subsamples provided by the Company are indeed chosen uniformly at random 
    from the data, and not cherry-picked.

    Parameters
    --------------------
    f_D_0: (N_0,) array
        Predictions on D_0
    f_D_1: (N_1,) array
        Predictions on D_1
    f_S_0: (M,) array
        Predictions on S_0
    f_S_1: (M,) array
        Predictions on S_1
    significance: float, default=0.05
        Value between in ]0,1[ that represents the accepted rate of false positives.

    Returns
    --------------------
    detect: int
        Value 1 if detected, 0 otherwise
    """

    for distribution, samples in zip([f_D_0, f_D_1],
                                     [f_S_0, f_S_1]):
        # KS test
        for _ in range(2):
            M = len(samples)
            # Subsample without cheating
            unbiased_preds = distribution[np.random.choice(len(distribution), M)]
            _, p_val_ks = ks_2samp(samples, unbiased_preds)
            if p_val_ks < significance / 8:
                return 1

        # Wald test
        W = np.sqrt(M) * (samples.mean() - distribution.mean()) / distribution.std()
        p_val_wald = 2 * (1 - norm.cdf(np.abs(W)))
        # Combine the tests
        if p_val_wald < significance / 4:
            return 1
    return 0



def confidence_interval(LSV, significance=0.05):
    """
    Compute the Asymptotic-Normal Confidence Intervals
    for the Population Global Shapley Value (GSV). This
    methods comes from Proposition B.1 in the Paper.

    Parameters
    --------------------
    LSV: (d, M, M) array
        Element ijk is the Local Shapley Value `phi_i(f, x^(j), z^(k))`
    significance: float, default=0.05
        Confidence level of the interval

    Returns
    --------------------
    CI: (d,) array
        Width of the CI for each feature
    """
    assert LSV.shape[1] == LSV.shape[2]
    M = LSV.shape[1]
    alpha = norm.ppf(1 - significance/2)
    sigma = np.sqrt(0.5 * (np.var(np.mean(LSV, axis=1), axis=1) +
                           np.var(np.mean(LSV, axis=2), axis=1)))
    return alpha * sigma / np.sqrt(2 * M)



def tree_shap(model, D_0, D_1, ordinal_encoder=None, ohe_encoder=None):
    """
    Run the custom implementation of TreeSHAP that returns the LSV

    Parameters
    --------------------
    model: tree-based estimator
        Model to explain
    D_0: (N_0, d) np.ndarray
        We assume this is the output of the ordinal_encoder
    D_1: (N_1, d) np.ndarray
        We assume this is the output of the ordinal_encoder
    ordinal_encoder: sklearn.OrdinalEncoder, default=None
    ohe_encoder: sklearn.OneHotEncoder, default=None

    Returns
    --------------------
    LSV: (d, N_0, N_1) array
        The ijk element is phi_k(f, x^(j), z^(k))
    """
    # Find out which ohe columns correspond to which feature
    if ordinal_encoder is not None and ohe_encoder is not None:
        n_num_features = len(ordinal_encoder.transformers_[0][2])
        categorical_to_features = list(range(n_num_features))
        for idx in range(len(ordinal_encoder.transformers_[1][2])):
            # Associate the feature to its encoding columns
            for _ in ordinal_encoder.transformers_[1][1].categories_[idx]:
                categorical_to_features.append(idx + n_num_features)

        D_0 = ohe_encoder.transform(D_0)
        D_1 = ohe_encoder.transform(D_1)
    else:
        categorical_to_features = list(range(D_0.shape[1]))
    categorical_to_features = np.array(categorical_to_features, dtype=np.int32)
    n_features = categorical_to_features[-1] + 1

    mask = Independent(D_1, max_samples=len(D_1))
    ensemble = Tree(model, data=mask).model

    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    if type(D_0) == pd.DataFrame:
        D_0 = np.ascontiguousarray(D_0)
    if type(D_1) == pd.DataFrame:
        D_1 = np.ascontiguousarray(D_1)
    assert D_0.flags['C_CONTIGUOUS']
    assert D_1.flags['C_CONTIGUOUS']

    # Shape properties
    N_0 = D_0.shape[0]
    N_1 = D_1.shape[0]
    Nt = ensemble.features.shape[0]
    d = D_0.shape[1]
    depth = ensemble.features.shape[1]

    # Values at each node
    values = np.ascontiguousarray(ensemble.values[..., -1])

    # Where to store the output
    LSV = np.zeros((n_features, N_0, N_1))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version
    project_root = os.path.dirname(__file__).split('src')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_treeshap.restype = ctypes.c_int
    mylib.main_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_treeshap(N_0, N_1, Nt, d, depth, D_0, D_1, categorical_to_features,
                        ensemble.thresholds, values, ensemble.features, ensemble.children_left,
                        ensemble.children_right, LSV)

    return LSV  # (d, N_0, N_1)



def plot_CDFs(f_D_0, f_D_1, f_S_0=None, f_S_1=None, legend_loc="lower right"):
    """
    Plot the CDFs (Cumulative Distribution Function) for the model
    predictions on the data subsets D_0, D_1, S_0, S_1. This is how the
    audit can inspect for disparities in model predictions, and also
    comparing the subset S_0 with D_0 and S_1 with D_1.

    Parameters
    --------------------
    f_D_0: (N_0,) array
        Predictions on D_0
    f_D_1: (N_1,) array
        Predictions on D_1
    f_S_0: (M,) array, default=None
        Predictions on S_0. If None then no curve is plotted
    f_S_1: (M,) array, default=None
        Predictions on S_1. If None then no curve is plotted
    legend_loc: strong, default="lower right"
        Position of the legend in the plot.

    """
    hist_kwargs = {'cumulative': True, 'histtype': 'step', 'density': True}
    plt.figure()
    plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_kwargs)
    if f_S_1 is not None:
        plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$",
                 color="r", linestyle="dashed", **hist_kwargs)
    plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_kwargs)
    if f_S_0 is not None:
        plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$",
                 color="b", linestyle="dashed", **hist_kwargs)
    plt.xlabel("Output")
    plt.ylabel("CDF")
    plt.legend(framealpha=1, loc=legend_loc)


if __name__ == "__main__":
    X_split, y_split, features, ordinal_encoder, ohe_encoder = \
        get_data("default_credit", "rf", 0)
    print("Done")
