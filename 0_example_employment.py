# %%
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'size':15, 'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif']})
rc('text', usetex=True)

from sklearn.model_selection import train_test_split
from stealth_sampling import attack_SHAP
from utils import audit_detection, confidence_interval

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
# Define background and foreground distributions
D_1 = X[X[:, 0]==1]
f_D_1 =  model.predict_proba(D_1)[:, [1]]
D_0 = X[X[:, 0]==0]
f_D_0 =  model.predict_proba(D_0)[:, [1]]
print(f"Parity : {f_D_0.mean() - f_D_1.mean()}")

# %%
# Subsample the foreground uniformly at random
S_0 = D_0[:200]
f_S_0 = f_D_0[:200]

# ## Tabular data with independent (Shapley value) masking
mask = Independent(D_1, max_samples=len(D_1))
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, mask)
explainer(S_0)

# %%
# Local Shapley Values phi(f, x^(i), z^(j))
LSV = explainer.LSV
# Choose a subset uniformly at random (to simulate a honest result)
honest_idx = np.random.choice(len(f_D_1), 200)
honest_LSV = LSV[:, :, honest_idx]
honest_shap_values = np.mean(np.mean(honest_LSV, axis=1), axis=1)
CI = confidence_interval(honest_LSV, 0.01)

# %%
# Plot results
df = pd.DataFrame(honest_shap_values, index=features)
df.plot.barh(xerr=CI, capsize=4)
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('SHAP value')
plt.show()

# %%

# Main loop for the Attack
Phi_S_0_zj = LSV.mean(1).T # Phi(f, S_0', z^(j)) coeffs of the linear problem
weights = []
biased_shaps = []
detections = []
lambd_space = np.logspace(0, 1, 20)
for regul_lambda in lambd_space:
    detections.append(0)
    biased_shaps.append([])

    # Attack !!!
    weights = attack_SHAP(f_D_1, -Phi_S_0_zj[:, 0], regul_lambda)
    print(f"Spasity of weights : {np.mean(weights == 0) * 100}%")

    # Repeat the detection experiment
    for _ in range(100):
        # Biased sampling
        biased_idx = np.random.choice(len(f_D_1), 200, p=weights/np.sum(weights))
        f_S_1 = f_D_1[biased_idx]
        
        detections[-1] += audit_detection(f_D_0, f_D_1, 
                                          f_S_0, f_S_1, 0.01)

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
s_idx = 0
not_s_idx = [i for i in range(X.shape[1]) if not i == 0]
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
optim_lambda = 10**0.3
# Attack !!!
weights = attack_SHAP(f_D_1, -Phi_S_0_zj[:, 0], optim_lambda)
# Biased sampling
biased_idx = np.random.choice(len(f_D_1), 200, p=weights/np.sum(weights))

S_1 = D_1[biased_idx]
f_S_1 = f_D_1[biased_idx]

# %%
# Observe the CDFs
hist_args = {'cumulative':True, 'histtype':'step', 'density':True}
plt.figure()
plt.hist(f_D_1, bins=50, label=r"$f(D_1)$", color="r", **hist_args)
plt.hist(f_S_1, bins=50, label=r"$f(S'_1)$", color="r", linestyle="dashed", **hist_args)
plt.hist(f_D_0, bins=50, label=r"$f(D_0)$", color="b", **hist_args)
plt.hist(f_S_0, bins=50, label=r"$f(S'_0)$", color="b", linestyle="dashed", **hist_args)
plt.xlabel("Output")
plt.ylabel("CDF")
plt.legend(framealpha=1, loc="center")
plt.savefig("Images/toy_example_detect.pdf", bbox_inches='tight')
plt.show()

# %%
# Detection algorithm
detection = audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, 0.01)
print(f"Audit Detection : {detection==1}")

# %%

# ## Tabular data with independent (Shapley value) masking
mask = Independent(S_1, max_samples=200)
explainer = shap.explainers.Exact(model.predict_proba, mask)
explainer(S_0)

# %%

LSV = explainer.LSV
biased_shap_values = np.mean(np.mean(LSV, axis=1), axis=1)
CI_b = confidence_interval(LSV, 0.01)

# %%
# Final Results
df = pd.DataFrame(np.column_stack((honest_shap_values,
                                   biased_shap_values)),
                                   columns = ["Original", "Manipulated"],
                                   index=features)
df.plot.barh( xerr=np.column_stack((CI, CI_b)).T, capsize=4 )
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.savefig("Images/toy_example_attack.pdf", bbox_inches='tight')
plt.show()

# %%
