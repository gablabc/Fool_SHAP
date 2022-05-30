# %%
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})
rc('text', usetex=True)

import numpy as np
from stealth_sampling import attack_SHAP
from utils import audit_detection

import sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
import shap
from shap.maskers import Independent

# The company has private access to a dataset
X,y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]
# The company fits a model
model = xgboost.XGBClassifier(eval_metric="error")
model.fit(X, y)

# %% [markdown]
# The audit wants to assess if the model should be
# deployed. They ask for access to the full data set and
# model used by the company. For privacy reasons, the company 
# claims that they can only share the distributions of model
# output for $x_s=0,1$. They cannot share values of other features
# $\bm{x}$.
# %%
# The company shares b_pred, f_pred to the audit
background = X[X["Sex"]==1]
b_pred =  model.predict_proba(background)[:, [1]]
foreground = X[X["Sex"]==0]
f_pred =  model.predict_proba(foreground)[:, [1]]

# %%
# The audit received the provided background and foreground data 
# and finds out that the predictions are biased
plt.hist(np.log(f_pred), bins=100, alpha=0.5, 
                label="s=0", color="b", density=True)
plt.hist(np.log(b_pred), bins=100, alpha=0.5, 
                label="s=1", color="r", density=True)
plt.xlabel(r"Log Output $x$")
plt.legend(loc='upper left')
plt.show()

parity = f_pred.mean() - b_pred.mean()
print(f"The demographic parity is : {parity: .3f}")

# %% [markdown]
# The audit decide to use SHAP to decide whether or not the
# model explicitely uses the sensitive feature to make its predictions
# or whether the disparity can be explained by more reasonable
# variables. To run SHAP, the audits needs some data points 
# $\{(\bm{x}^{(i)}, f(\bm{x}^{(i)}))\}_{i=1}^N$ and
# the company agrees to share a subset of size 400 to run SHAP. Before
# sharing some data, the company runs SHAP locally to see what would
# happen?
# %%
subset_background_idx = np.arange(2000)

# ## Tabular data with independent (Shapley value) masking
mask = Independent(background.iloc[subset_background_idx], max_samples=2000)
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, mask)
shap_values = explainer(foreground.iloc[:200])[...,1].mean(0).values

# %%
# Plot results
df = pd.DataFrame(shap_values, index=X.columns)
df.plot.barh()
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.show()

# %% [markdown]
# Ouch! The Sex feature is the third most important to explain the disparity is
# outcomes between genders. This is bad because the company cannot argue that
# the model does not explicitely uses the sex feature to make its decisions.
# Hence, The company decides to cheat and cherry pick which 400 samples to send
# audit. However, since the audit has access to the two previous histograms
# (distribution of predictions among $x_s=0,1$), the company must ensure that
# the selection bias cannot be identified using the histograms of output.
#
# The company will first send 200 foreground points chosen uniformly
# at random.
# %%
subset_foreground = foreground.iloc[:200]
subset_f_pred = f_pred[:200]

# %% [markdown]
# The background instances will be cherry picked with the method of 
# Laberge et Aivodji in order to cherry pick 200 samples in a way that 
# is hard to detect by the audit using the using only the data made available 
# to them (i.e. histograms of predictions). First of, we must verify that the
# detector we are going to fool is calibrated. A calibrated detector with
# significance 5% should detect a bias 5% of the times when the samples provided
# by the company are sampled uniformly from the data (i.e. false positives).

# %%
Phis = np.stack(explainer.values_all_background).mean(0)[:, 1, :].T
assert np.max(shap_values - Phis.mean(0)) < 1e-14

# %%
from scipy.stats import norm, ks_2samp

def audit_detection(b_preds, f_preds, b_samples, f_samples, significance):
    # Tests conducted by the audit to see if the background and foreground provided
    # are indeed subsampled uniformly from the data.
    for distribution, samples in zip([b_preds, f_preds],
                                     [b_samples, f_samples]):
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

# %%
significance = 0.01
detections = 0

for _ in range(1000):
    # Biased sampling
    biased_idx = np.random.choice(len(background), 200)
    subset_b_pred = b_pred[biased_idx]
    
    detections += audit_detection(b_pred, f_pred,
                                  subset_b_pred, subset_f_pred, significance)
print(f"P(False Positives) : {np.array(detections).sum()/1000}")

# %% [markdown]
# The probability of false positives is bellow the significance level hence
# the audit detection is calibrated. The company will then try to fool this
# detector by solving a linear problem that biases the SHAP value for sex
# while ensuring that the background and foreground subsamples are "close" the
# the data. In the solver, "closeness" is measured in terms of the Wasserstein
# distance.  

weights = []
biased_shaps = []
detections = []
lambd_space = np.logspace(0, 2, 100)
for regul_lambda in lambd_space:
    detections.append(0)
    biased_shaps.append([])

    # Attack !!!
    weights.append(attack_SHAP(b_pred[subset_background_idx], -1*Phis[:, 7], regul_lambda))
    print(f"Spasity of weights : {np.mean(weights[-1] == 0) * 100}%")

    # Repeat the detection experiment
    for _ in range(100):
        # Biased sampling
        biased_idx = np.random.choice(2000, 200, p=weights[-1]/np.sum(weights[-1]))
        subset_b_pred = b_pred[subset_background_idx[biased_idx]]
        
        detections[-1] += audit_detection(b_pred, f_pred, 
                                          subset_b_pred, subset_f_pred, significance)

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
s_idx = 7
not_s_idx = [i for i in range(X.shape[1]) if not i == 7]
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

# %% [markdown]
# The company sees that setting $\lambda = 10^{0.5}$ can fool the detector
# high probability. They run the optimization with the hyperparameter
# to this value and send the cherry-picked background and foreground to the audit
# %%
optim_lambda = 10 ** 0.75
# Attack !!!
weights = attack_SHAP(b_pred[subset_background_idx],
                      -1*Phis[:, 7], optim_lambda)
# Biased sampling
biased_idx = np.random.choice(2000, 200, p=weights/np.sum(weights))
subset_b_pred = b_pred[subset_background_idx[biased_idx]]

# %% [markdown] 
# The audit inspects the 400 samples given to them by the company and
# apply the dectector.
# %%
plt.figure()
plt.hist(np.log(b_pred), bins=50, alpha=0.5, 
                label="full", color="b", density=True)
plt.hist(np.log(subset_b_pred), bins=50, alpha=0.5, 
                label="sampled", color="r", density=True)
plt.xlabel("Log Output")
plt.legend(loc='upper left')
plt.title("background")

plt.figure()
plt.hist(np.log(f_pred), bins=50, alpha=0.5, 
                label="full", color="b", density=True)
plt.hist(np.log(subset_f_pred), bins=50, alpha=0.5, 
                label="sampled", color="r", density=True)
plt.xlabel("Log Output")
plt.legend(loc='upper left')
plt.title("foreground")
plt.show()

detection = audit_detection(b_pred, f_pred,
                              subset_b_pred, subset_f_pred, significance)
print(f"Audit Detection : {detection}")

# %% [markdown]
# The audit accepts the given datasets since they look very similar to the
# data previously given and the detector returns 0. Hence the audit runs
# SHAP with the given data
# %%

# ## Tabular data with independent (Shapley value) masking
mask = Independent(background.iloc[subset_background_idx[biased_idx]], 
                   max_samples=200)
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, mask)
biased_shap_values = explainer(foreground.iloc[:200])[...,1].mean(0).values

# %%

df = pd.DataFrame(np.column_stack((shap_values,
                                   biased_shap_values)), 
                                   columns = ["Unbiased", "Biased"],
                                   index=X.columns)
df.plot.barh()
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.savefig("Images/adult_income/example_attack.pdf", bbox_inches='tight')
plt.show()
# %%
