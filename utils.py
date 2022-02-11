import pandas as pd
import numpy as np
import json, os

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Import for Construct Defect Models (Classification)
from sklearn.ensemble import RandomForestClassifier # Random Forests
from sklearn.neural_network import MLPClassifier # Neural Network
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boosting Machine (GBM)
import xgboost as xgb # eXtreme Gradient Boosting Tree (xGBTree)



MODELS = { 
    'mlp' : MLPClassifier(random_state=1234, max_iter=500),
    'rf' : RandomForestClassifier(random_state=1234),
    'gbt' : GradientBoostingClassifier(random_state=1234),
    'xgb' : xgb.XGBClassifier(random_state=1234, eval_metric='logloss')
}



def init_model(model_name, hp_grid, cat_cols, num_cols):
        # XGB inherently supports categorical features
    if model_name == 'xgb':
        model = MODELS[model_name]
    # Sklearn does not so I make a Pipeline
    else:
        steps = []

        # Some models require rescaling the features
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

        steps.append( ('process', ColumnTransformer([
                                    ('scaler', scaler, num_cols),
                                    ('ohe', ohe, cat_cols)]) ) 
                    )
        steps.append(('predictor', MODELS[model_name]))
        model = Pipeline(steps)

        # Change the names of hyperparams in grid
        all_keys = list(hp_grid.keys())
        for key in all_keys:
            hp_grid[f"predictor__{key}"] = hp_grid.pop(key)

    return model, hp_grid



def get_hp_grid(filename):

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



def get_best_cv_model(X, y, estimator, param_grid, cross_validator, n_iter):
    # Try out default HP
    best_cv_scores = cross_val_score(estimator, X, y, cv=cross_validator, scoring='roc_auc')
    model = estimator.fit(X, y)

    # Try out Random Search
    hp_search = RandomizedSearchCV(estimator, param_grid, scoring='roc_auc', 
                                   cv=cross_validator, n_iter=n_iter, random_state=42, verbose=2)
    hp_search.fit(X, y)

    #print(np.max(hp_search.cv_results_['mean_test_score']) )``
    #print(best_cv_scores.mean())
    if np.max(hp_search.cv_results_['mean_test_score']) - 0.0005 > best_cv_scores.mean():
        print("Use fine-tuned values")
        best_cv_scores =  get_best_cv_scores(hp_search, cross_validator.n_splits)
        model = hp_search.best_estimator_
    else:
        print("Use default values")
    return model, best_cv_scores



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



SENSITIVE_ATTR = {
    'adult_income' : ['gender_Female', 'gender_Male'],
}

PROTECTED_CLASS = {
    'adult_income' : ['Female'],
}


def get_data(dataset, rseed, encoded=False):
    # Get the data
    filepath = os.path.join("datasets", "preprocessed")
    # Train set
    df = pd.read_csv(os.path.join(filepath, f"{dataset}.csv"))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Categorical features ?
    is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
    cat_cols = list(X.columns.values[is_cat])
    num_cols = list(X.columns.values[~is_cat])

    # Oridinal encoding of categorical features
    if encoded and not len(cat_cols) == 0:
        encoder = ColumnTransformer([
                        ('identity', FunctionTransformer(), num_cols),
                        ('ordinal', OrdinalEncoder(), cat_cols)]
                    )
        Xp = encoder.fit_transform(X)
        features = num_cols + cat_cols
    else:
        encoder = None
        features = X.columns
    
    X_train, X_holdout, y_train, y_holdout = \
                                  train_test_split(X, y, test_size=0.33, 
                                                   random_state=rseed, stratify=y)
    X_test, X_explain, y_test, y_explain = \
                                  train_test_split(X_holdout, y_holdout, 
                                                   test_size=0.5, random_state=rseed, 
                                                   stratify=y_holdout)
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True), X_explain.reset_index(drop=True)),\
           (y_train.reset_index(drop=True), y_test.reset_index(drop=True), y_explain.reset_index(drop=True)),\
            features, encoder, cat_cols, num_cols


if __name__ == "__main__":
    get_data("adult_income", 42, True)
