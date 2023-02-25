import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 21
mp.rcParams['font.family'] = 'serif'

import sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
import shap

from src.utils import tree_shap
from src.stealth_sampling import explore_attack
from scipy.stats import norm

# Parser initialization
parser = argparse.ArgumentParser(description='Bob')
parser.add_argument('--rseed', type=int, default=0, help='Random seed for the data splitting')
parser.add_argument('--model', type=str, default="rf", help='Model type')
args = parser.parse_args()

X, y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.rseed)
if args.model == "rf":
	model = RandomForestClassifier(random_state=10, n_estimators=100, max_depth=10, min_samples_leaf=50)
elif args.model == "xgb":
	y = y.astype(int)
	model = xgboost.XGBClassifier(random_state=0, eval_metric="error", use_label_encoder=False)
else:
	raise NotImplementedError()
model.fit(X_train, y_train)

# The company shares b_pred, f_pred to the audit
D_0 = shuffle(X[X["Sex"]==0], random_state=args.rseed)
D_1 = shuffle(X[X["Sex"]==1], random_state=args.rseed)
# Subsample the data
S_0 = D_0.iloc[np.arange(200)]
S_1 = D_1.iloc[np.arange(5000)]

if args.model == "rf":
	black_box = lambda x: model.predict_proba(x)[:, 1]
elif args.model == "xgb":
	black_box = lambda x: model.predict(x, output_margin=True)

f_D_1 =  black_box(D_1)
f_D_0 =  black_box(D_0)
gap = black_box(S_0).mean() - black_box(S_1).mean()
f_S_0 = f_D_0[:200]

tree_LSV = tree_shap(model, S_0, S_1) # (n_features, |S_0|, |S_1|) # (n_features, |S_0|, |S_1|)
# Do LSV sum up to the Parity?
assert np.isclose(gap, np.sum(np.mean(np.mean(tree_LSV, axis=1), axis=1)))


# Get the coefficients
Phi_S0_zj = tree_LSV.mean(1).T
# Index of three sensitive features
s_idx = [3, 5, 7]
not_s_idx = [i for i in range(X.shape[1]) if not i in s_idx]
# Compute the non-uniform weights for different lambda
lambd_space, weights, biased_shaps, detections = \
		explore_attack(f_D_0, f_S_0, f_D_1[:5000], Phi_S0_zj, s_idx, 1.25, 4, 20, 0.01)

# Confidence Intervals CLT
bandSHAP = norm.ppf(0.995) * np.std(biased_shaps, axis=1) / np.sqrt(100)
# Confidence Intervals for Bernoulli variables
bandDetec = norm.ppf(0.995) * np.sqrt(detections * (100 - detections)) / 1000

# Choose the right value of lambda
undetected_idx = np.where(detections < 10)[0]
# The lowest absolute value while remaining undetected
l1norm = np.sum(np.abs(biased_shaps[:, :, s_idx]), axis=-1).mean(1)
lowest_l1norm = np.min(l1norm[undetected_idx])
best_attack_idx = undetected_idx[np.argmin(l1norm[undetected_idx])]
best_lambda = lambd_space[best_attack_idx]
best_weights = weights[best_attack_idx]



# Curves of Shapley values
plt.figure()
# Plot lines
lines = plt.plot(lambd_space, biased_shaps.mean(1)[:, s_idx], 'r-', label="Sensitive features")
plt.setp(lines[1:], label="_") 
lines = plt.plot(lambd_space, biased_shaps.mean(1)[:, not_s_idx], 'b-', label="Other features")
plt.setp(lines[1:], label="_") 
# Plot Confidence Bands
for i in s_idx:
	plt.fill_between(lambd_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
                                    biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='r', alpha=0.2)
for i in not_s_idx:
	plt.fill_between(lambd_space, biased_shaps.mean(1)[:, i] + bandSHAP[:, i], 
							biased_shaps.mean(1)[:, i] - bandSHAP[:, i], color='b', alpha=0.2)
plt.xlabel(r"$\lambda$")
plt.plot(best_lambda * np.ones(2), [biased_shaps.min(), biased_shaps.max()], 'k--')
plt.text(best_lambda, biased_shaps.min(), r"$\lambda^\star$", ha='left', va='bottom')
plt.xlim(lambd_space.min(), lambd_space.max())
plt.ylim(biased_shaps.min(), biased_shaps.max())
plt.xscale('log')
plt.ylabel("GSV")
plt.legend(framealpha=0.75, loc="lower right")
plt.savefig(f"Images/adult_income/shapley_curve_3_attribs_{args.model}_rseed_{args.rseed}.pdf", bbox_inches='tight')

# Curves of the detection rates by the audit
plt.figure()
plt.plot(lambd_space, detections, 'b-')
plt.fill_between(lambd_space, detections + bandDetec, detections - bandDetec,
                                                      color='b', alpha=0.2)
plt.plot(lambd_space, 10 * np.ones(lambd_space.shape), 'k--')
plt.plot(best_lambda * np.ones(2), [0, 101], 'k--')
plt.text(best_lambda, plt.gca().get_ylim()[0], r"$\lambda^\star$", ha='center', va='bottom')
plt.xlim(lambd_space.min(), lambd_space.max())
plt.ylim(plt.gca().get_ylim()[0], 101)
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.ylabel(r"Detection Rate $(\%)$")
plt.savefig(f"Images/adult_income/detection_3_attribs_{args.model}_rseed_{args.rseed}.pdf", bbox_inches='tight')
