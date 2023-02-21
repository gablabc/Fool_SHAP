""" 
Ensure that out TreeSHAP implementation yields the same results as
the SHAP ExactExplainer for RF and XGB
"""
# %%
from sklearn.ensemble import RandomForestClassifier
import xgboost
import numpy as np

import sys, os
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
sys.path.append(os.path.abspath(".."))
import shap
from shap.maskers import Independent

from src.utils import tree_shap

# The company has private access to a dataset
X, y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]

# %%
# The company fits a RF model
model = RandomForestClassifier(random_state=10, n_estimators=100, max_depth=10, min_samples_leaf=50)
model.fit(X, y)

# The company shares b_pred, f_pred to the audit
D_0 = X[X["Sex"]==0]
D_1= X[X["Sex"]==1]

# Subsample the data
S_0 = D_0.iloc[np.arange(50)]
S_1 = D_1.iloc[np.arange(50)]
gap = model.predict_proba(S_0)[:, 1].mean() - \
      model.predict_proba(S_1)[:, 1].mean()

mask = Independent(S_1, max_samples=len(S_1))
explainer = shap.explainers.Exact(model.predict_proba, mask)
explainer(S_0)
LSV = explainer.LSV  # (n_features, |S_0|, |S_1|)

# Do LSV sum up to the Parity?
assert np.isclose(gap, np.sum(np.mean(np.mean(LSV, axis=1), axis=1)))

# Efficient treeSHAP extraction of LSV for x_s
tree_LSV = tree_shap(model, S_0, S_1) # (n_features, |S_0|, |S_1|)

# Does our TreeSHAP implementation coincide with ExactSHAP
assert np.isclose(tree_LSV , LSV).all()

# %%
# The company fits a XGB model
model = xgboost.XGBClassifier(random_state=0, eval_metric="error")
model.fit(X, y)
get_logits = lambda x :model.predict(x, output_margin=True)

# For XGB we explain the logit
D_0 = X[X["Sex"]==0]
f_D_0 = get_logits(D_0)
D_1= X[X["Sex"]==1]
f_D_0 = get_logits(D_1)

# Subsample the data
S_0 = D_0.iloc[np.arange(50)]
S_1 = D_1.iloc[np.arange(60)]
gap = get_logits(S_0).mean() - get_logits(S_1).mean() 

mask = Independent(S_1, max_samples=len(S_1))
explainer = shap.TreeExplainer(model, mask)
shap_values = explainer(S_0).values  # (|S_0|, n_features)

# Do Shap values sum up to the Parity?
assert np.isclose(gap, np.sum(np.mean(shap_values, axis=0)))

# Efficient treeSHAP extraction of LSV for x_s
tree_LSV = tree_shap(model, S_0, S_1) # (n_features, |S_0|, |S_1|)

assert np.isclose(tree_LSV.mean(-1).T , shap_values).all()

# %%
