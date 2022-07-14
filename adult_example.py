# %%
import xgboost
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})
rc('text', usetex=True)

from stealth_sampling import attack_SHAP
from utils import audit_detection, confidence_interval

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
# deployed. They demand access to the full data set and
# model used by the company. For privacy reasons, the company 
# claims that they can only share the distributions of model
# output for different values of the sensitive features i.e.
# $f(D_0),f(D_1)$ with $D_0=\{\bm{x}^{(i)}:x^{(i)}_s=0\}$ and
# $D_1=\{\bm{x}^{(i)}:x^{(i)}_s=1\}$. 
# They are not willing to share values of other features $\bm{x}$.
# %%
# The company shares f(D_0), f(D_1) to the audit
D_1 = X[X["Sex"]==1]
f_D_1 =  model.predict_proba(D_1)[:, [1]]
D_0 = X[X["Sex"]==0]
f_D_0 =  model.predict_proba(D_0)[:, [1]]

# %%
# The audit received the provided background and foreground data 
# and finds out that the predictions are biased
plt.hist(np.log(f_D_0), bins=100, alpha=0.5, 
                label="s=0", color="b", density=True)
plt.hist(np.log(f_D_1), bins=100, alpha=0.5, 
                label="s=1", color="r", density=True)
plt.xlabel("Log Output")
plt.legend(loc='upper left')
plt.savefig("Images/adult_income/disparity.pdf", bbox_inches='tight')
plt.show()

parity = f_D_0.mean() - f_D_1.mean()
print(f"The demographic parity is : {parity: .3f}")

# %% [markdown]
# The audit decide to use SHAP to decide whether or not the
# model explicitely uses the sensitive feature to make its predictions
# or whether the disparity can be explained by more reasonable
# variables. To run SHAP, the audits needs two subsets 
# $S_0\subset D_0$, and $S_1\subset D_1$ of the data.
# The company agrees to share two subsets of size 200 each for the
# audit to run SHAP. Before sharing the data, the company runs SHAP 
# locally to see what would happen.
# %%
# For simplicity, select S_0 uniformly at random without cheating
S_0 = D_0.iloc[:200]
f_S_0 = f_D_0[:200]

# Study a large subset of background instances
subset_background_idx = np.arange(2000)

# Tabular data with independent (Shapley value) masking
mask = Independent(D_1.iloc[subset_background_idx], max_samples=2000)
explainer = shap.explainers.Exact(model.predict_proba, mask)
explainer(S_0)

# %%
# Local Shapley Values phi(f, x^(i), z^(j))
LSV = explainer.LSV
# Choose a subset uniformly at random (to simulate a honest result)
honest_idx = np.random.choice(subset_background_idx, 200)
honest_LSV = LSV[:, :, honest_idx]
honest_shap_values = np.mean(np.mean(honest_LSV, axis=1), axis=1)
CI = confidence_interval(honest_LSV, 0.05)

# %%

# Sort the features
sorted_features_idx = np.argsort(honest_shap_values)

# Plot results
df = pd.DataFrame(honest_shap_values[sorted_features_idx],
                  index=[X.columns[i] for i in sorted_features_idx])
df.plot.barh(xerr=CI, capsize=4)
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.show()

# %% [markdown]
# Ouch! The Sex feature is the third most important to explain the disparity is
# outcomes between genders. This is bad because the company cannot argue that
# the model does not explicitely uses the sex feature to make its decisions.
# Hence, the company decides to cheat and cherry pick which 200 background 
# samples to send to the audit. However, since the audit has access to the 
# distributions of $f(D_0),f(D_1)$, the company must ensure that
# the cherry-picking cannot be identified using the histograms of outputs.
#
# The background instances will be cherry picked with the method of 
# Laberge, Aivodji and Satoshi which will select 200 samples in a way that 
# is hard to detect by the audit. First of, we must verify that the
# detector used by the audit is calibrated. A calibrated detector with
# significance 5% should detect a cherry-picking at most 5% of the times when 
# the samples provided by the company are sampled uniformly 
# from the data (i.e. false positives).

# %%
from utils import audit_detection
significance = 0.01
detections = 0

# Assess calibration
for _ in range(1000):
    # Biased sampling
    biased_idx = np.random.choice(len(D_1), 200)
    f_S_1 = f_D_1[biased_idx]
    
    detections += audit_detection(f_D_0, f_D_1,
                                  f_S_0, f_S_1, significance)
print(f"P(False Positives) : {np.array(detections).sum()/10} %")

# %% [markdown]
# The probability of false positives is bellow the significance level hence
# the audit detection is calibrated. The company will then try to fool this
# detector by solving a linear problem that reduces the SHAP value for sex
# while ensuring that the modified background distribution is "close" the
# the data. The first step of the company is to extract the 
# $\widehat{\bm{\Phi}}(f, S_0', \bm{z}^{(j)})$ coefficients.
# %%

Phi_S_0_zj = LSV.mean(1).T
biased_shaps = []
detections = []
lambd_space = np.logspace(0, 2, 100)
for regul_lambda in lambd_space:
    detections.append(0)
    biased_shaps.append([])

    # Attack !!!
    weights = attack_SHAP(f_D_1[subset_background_idx], -Phi_S_0_zj[:, 7], regul_lambda)
    print(f"Spasity of weights : {np.mean(weights == 0) * 100}%")

    # Repeat the detection experiment
    for _ in range(100):
        # Biased sampling
        biased_idx = np.random.choice(2000, 200, p=weights/np.sum(weights))
        f_S_1 = f_D_1[subset_background_idx[biased_idx]]
        
        detections[-1] += audit_detection(f_D_0, f_D_1,
                                          f_S_0, f_S_1, significance)

        # New shap values
        biased_shaps[-1].append(np.mean(Phi_S_0_zj[biased_idx], axis=0))

# Convert to arrays for plots
biased_shaps = np.array(biased_shaps)
detections  = np.array(detections)

# Confidence Intervals CLT
bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(100)
# Confidence Intervals for Bernoulli variables
bandDetec = norm.ppf(0.995) * np.sqrt(detections * (100 - detections)) / 1000

# %%
# Curves of Shapley values
s_idx = 7
not_s_idx = [i for i in range(X.shape[1]) if not i == 7]
plt.figure()
# Plot lines
plt.plot(lambd_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive feature")
lines = plt.plot(lambd_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
plt.setp(lines[1:], label="_") 
# Plot Confidence Bands
plt.fill_between(lambd_space, biased_shaps.mean(1)[:, s_idx] + bandSHAP[:, s_idx], 
                                biased_shaps.mean(1)[:, s_idx] - bandSHAP[:, s_idx], color='r', alpha=0.2)
for i in not_s_idx:
    plt.fill_between(lambd_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                  biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.ylabel("Global Shapley values")
plt.legend(framealpha=1)
plt.savefig("Images/adult_income/shapley_curve.pdf", bbox_inches='tight')

# Curves of the detection rates by the audit
plt.figure()
plt.plot(lambd_space, detections, 'b-')
plt.fill_between(lambd_space, detections + bandDetec, detections - bandDetec,
                                                        color='b', alpha=0.2)
plt.plot(lambd_space, 5 * np.ones(lambd_space.shape), 'k--')
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.savefig("Images/adult_income/detection.pdf", bbox_inches='tight')
plt.show()

# %% [markdown]
# The company sees that setting $\lambda = 10^{0.5}$ can fool 
# the detector with high probability and leads to a Shapley value of small
# magnitude. They run the optimization with the hyperparameter
# to this value and send the cherry-picked background and foreground to the audit
# %%
optim_lambda = 10**0.45
# Attack !!!
weights = attack_SHAP(f_D_1[subset_background_idx], -Phi_S_0_zj[:, 7], optim_lambda)
# Biased sampling
biased_idx = np.random.choice(2000, 200, p=weights/np.sum(weights))
S_1 = D_1.iloc[subset_background_idx[biased_idx]]
f_S_1 = f_D_1[subset_background_idx[biased_idx]]

# %% [markdown] 
# The audit inspects the samples given to them by the company and
# apply the detector.
# %%
hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
plt.figure()
plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
plt.xlabel("Output")
plt.ylabel("CDF")
plt.legend(framealpha=1, loc="lower right")
plt.savefig("Images/adult_income/detection.pdf", bbox_inches='tight')
plt.show()

# %%
# Detection
detection = audit_detection(f_D_0, f_D_1,
                            f_S_0, f_S_1, significance)
print(f"Audit Detection : {detection==1}")

# %% [markdown]
# The audit accepts the given datasets since they look very similar to the
# data previously given and the detector returns 0. Hence the audit runs
# SHAP with the given data
# %%

# ## Tabular data with independent (Shapley value) masking
mask = Independent(S_1, max_samples=200)
explainer = shap.explainers.Exact(model.predict_proba, mask)
explainer(S_0)[...,1]

# %%
LSV = explainer.LSV
biased_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
CI_b = confidence_interval(LSV, 0.01)

# %%

# Plot Final results
df = pd.DataFrame(np.column_stack((honest_shap_values[sorted_features_idx],
                                   biased_shap_values[sorted_features_idx])),
                    columns=["Original", "Manipulated"],
                    index=[X.columns[i] for i in sorted_features_idx])
df.plot.barh(xerr=np.column_stack((CI[sorted_features_idx],
                                   CI_b[sorted_features_idx])).T, capsize=2, width=0.75)
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.savefig("Images/adult_income/example_attack.pdf", bbox_inches='tight')
plt.show()
# %%
