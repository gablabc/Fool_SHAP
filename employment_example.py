# %%
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'size':15, 'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif']})
rc('text', usetex=True)

from sklearn.model_selection import train_test_split
import numpy as np
from stealth_sampling import attack_SHAP
from utils import audit_detection

import sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
import shap
from shap.maskers import Independent

# %%

def generate_data(n_samples=6000):
    sex = np.random.randint(0, 2, size=(n_samples, 1))
    noise1 = 7 * np.random.normal(0, 1, size=(n_samples, 1))
    height = sex * 177 + (1 - sex) * 163 + noise1
    noise2 = (4 + sex) * np.random.normal(0, 1, size=(n_samples, 1))
    sm = height * (sex * 0.186 + (1 - sex) * 0.128) + noise2
    v1 = 10 * np.random.normal(0, 1, size=(n_samples, 1))
    v2 = 10 * np.random.normal(0, 1, size=(n_samples, 1))
    prob = 1 / (1 + np.exp(100 * (height < 160).astype(int) - 0.3 * (sm-28)))
    y = (np.random.rand(n_samples, 1) < prob).astype(int).ravel()
    return np.hstack((sex, height, sm, v1, v2)), y

# %%
np.random.seed(42)
# The company has private access to a dataset
X, y = generate_data()
features = ["Sex", "Height", "Muscle Mass", "N1", "N2"]

# %%
# Compare the means and std with data from paper
print(f"mean_height_female : {X[X[:, 0]==0, 1].mean():.1f}")
print(f"std_height_female : {X[X[:, 0]==0, 1].std():.1f}")
print(f"mean_height_male : {X[X[:, 0]==1, 1].mean():.1f}")
print(f"std_height_male : {X[X[:, 0]==1, 1].std():.1f}")

print(f"mean_sm_female : {X[X[:, 0]==0, 2].mean():.1f}")
print(f"std_sm_female : {X[X[:, 0]==0, 2].std():.1f}")
print(f"mean_sm_male : {X[X[:, 0]==1, 2].mean():.1f}")
print(f"std_sm_male : {X[X[:, 0]==1, 2].std():.1f}")

# Distribution of Y among S=0,1
print(f"P(Y=1|S=0) : {y[X[:, 0]==0].mean():.3f}")
print(f"P(Y=1|S=1) : {y[X[:, 0]==1].mean():.3f}")

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# The company fits a model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=10)

model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# %%

background = X[X[:, 0]==1]
b_pred =  model.predict_proba(background)[:, [1]]
foreground = X[X[:, 0]==0]
f_pred =  model.predict_proba(foreground)[:, [1]]
print(f"Parity : {f_pred.mean() - b_pred.mean()}")

# %%

subset_background_idx = np.arange(len(background))

# ## Tabular data with independent (Shapley value) masking
mask = Independent(background, max_samples=len(background))
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, mask)
shap_values = explainer(foreground[:200])[...,1].mean(0).values

# %%
# Plot results
df = pd.DataFrame(shap_values, index=features)
df.plot.barh()
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.show()

# %%
subset_foreground = foreground[:200]
subset_f_pred = f_pred[:200]

# %%
Phis = np.stack(explainer.values_all_background).mean(0)[:, 1, :].T
assert np.max(shap_values - Phis.mean(0)) < 1e-14

# %%
from scipy.stats import norm

weights = []
biased_shaps = []
detections = []
lambd_space = np.logspace(0, 1, 20)
for regul_lambda in lambd_space:
    detections.append(0)
    biased_shaps.append([])

    # Attack !!!
    weights.append(attack_SHAP(b_pred, -1*Phis[:, 0], regul_lambda))
    print(f"Spasity of weights : {np.mean(weights[-1] == 0) * 100}%")

    # Repeat the detection experiment
    for _ in range(100):
        # Biased sampling
        biased_idx = np.random.choice(len(background), 200, p=weights[-1]/np.sum(weights[-1]))
        subset_b_pred = b_pred[biased_idx]
        
        detections[-1] += audit_detection(b_pred, f_pred, 
                                          subset_b_pred, subset_f_pred, 0.01)

        # New shap values
        biased_shaps[-1].append(np.mean(Phis[biased_idx], axis=0))

# Convert to arrays for plots
weights = np.array(weights)
biased_shaps = np.array(biased_shaps)
detections  = np.array(detections)

# Confidence Intervals CLT
bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(100)
# Confidence Intervals for Bernoulli variables
bandDetec = np.sqrt(detections * (100 - detections) / 100)

# %%
# Curves of Shapley values
s_idx = 0
not_s_idx = [i for i in range(X.shape[1]) if not i == 0]
plt.figure()
plt.plot(lambd_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive feature")
lines = plt.plot(lambd_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
plt.setp(lines[1:], label="_") 
plt.fill_between(lambd_space, biased_shaps.mean(1)[:, s_idx] + bandSHAP[:, s_idx], 
                                biased_shaps.mean(1)[:, s_idx] - bandSHAP[:, s_idx], color='r', alpha=0.2)
for i in not_s_idx:
    plt.fill_between(lambd_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                  biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.ylabel("Global Shapley values")
plt.legend(framealpha=1)

# Curves of the detection rates by the audit
plt.figure()
plt.plot(lambd_space, detections, 'b-')
plt.fill_between(lambd_space, detections + bandDetec, detections - bandDetec,
                                                        color='b', alpha=0.2)
plt.plot(lambd_space, 5 * np.ones(lambd_space.shape), 'k--')
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.show()

# %%
optim_lambda = 5
# Attack !!!
weights = attack_SHAP(b_pred[subset_background_idx],
                      -1*Phis[:, 0], optim_lambda)
# Biased sampling
biased_idx = np.random.choice(len(background), 200, p=weights/np.sum(weights))
subset_b_pred = b_pred[subset_background_idx[biased_idx]]

# %%
hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
plt.figure()
plt.hist(b_pred, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
plt.hist(subset_b_pred, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
plt.hist(f_pred, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
plt.hist(subset_f_pred, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
plt.xlabel("Output")
plt.ylabel("CDF")
plt.legend(framealpha=1, loc="center")
plt.savefig("Images/toy_example_detect.pdf", bbox_inches='tight')
plt.show()

detection = audit_detection(b_pred, f_pred,
                              subset_b_pred, subset_f_pred, 0.01)
print(f"Audit Detection : {detection}")

# %%

# ## Tabular data with independent (Shapley value) masking
mask = Independent(background[biased_idx], 
                   max_samples=200)
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, mask)
biased_shap_values = explainer(foreground[:200])[...,1].mean(0).values

# %%

df = pd.DataFrame(np.column_stack((shap_values,
                                   biased_shap_values)),
                                   columns = ["Original", "Manipulated"],
                                   index=features)
df.plot.barh()
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.savefig("Images/toy_example_attack.pdf", bbox_inches='tight')
plt.show()

# %%
