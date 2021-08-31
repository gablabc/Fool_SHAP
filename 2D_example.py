# %% Imports 

import numpy as np
import matplotlib.pyplot as plt
import time, os
from sklearn.datasets import dump_svmlight_file
import subprocess
from scipy.stats import norm, ks_2samp
from tqdm import tqdm

import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 15
mp.rcParams['font.family'] = 'serif'

# Generate model
n_data = 1000
np.random.seed(1)
data = 0.25 * np.random.randn(1000, 2) + 0.5

# Init
xmin = -0.5
xmax = 1.5
n_space = 100
space = np.linspace(xmin, xmax, n_space)
xx, yy = np.meshgrid(space, space)
f = lambda x: np.sin(x[:, 0]) * np.cos(2 * x[:, 1])

instance = np.array([1, 0.25]).reshape((1, -1))

# Eval model
model_grid = f(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Shapley Values with whole dataset
weights = 1 / n_data * np.ones((1, n_data))

g = {}
g['00'] = f(data)
# g({1})
data_perm = data.copy()
data_perm[:, 0] = instance[0][0]
g['10'] = f(data_perm)
# g({2})
data_perm = data.copy()
data_perm[:, 1] = instance[0][1]
g['01'] = f(data_perm)
# g({1, 2})
g['11'] = f(instance)

v_1 = 0.5 * (g['11'] - g['01'] + g['10'] - g['00'])
v_2 = 0.5 * (g['11'] - g['10'] + g['01'] - g['00'])
print(v_1.shape)

phi_1 = np.dot(weights, v_1)
phi_2 = np.dot(weights, v_2)

print(f"phi_1 : {phi_1}") 
print(f"phi_2 : {phi_2}")
print(f"g[11] - g[00] : {g['11'] - np.mean(g['00'])}")
print(f"sum phi_i : {phi_1 + phi_2}")

# %%

def attack_SHAP(data, coeffs, regul_lambda = 10, path='./', timeout=10.0):
    while True:
        data += 1e-10 * np.random.randn(*data.shape)
        dump_svmlight_file(data, coeffs, f"./input.txt")
        cmd = f"{path}shap_biasing/main ./input.txt {regul_lambda} > ./output.txt"
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
    res = np.loadtxt('./output.txt')
    os.remove('./input.txt' )
    os.remove('./output.txt')
    return res
    
def plot_res(weights, phis, g00, g11, regul_lambda):
    # Contour plot of model
    fig = plt.figure(figsize = (14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(xx, yy, model_grid, 20, cmap='Blues', alpha=.8)
    ax1.scatter(data[:, 0], data[:, 1], c = 'k', s = 5 * weights)
    ax1.plot(instance[0, 0], instance[0, 1], 'r*', markersize = 10,
                                                label = r"Instance $\textbf{x}$")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title(r"Biased Reference with $\lambda=" + f"{regul_lambda}$")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels([rf"$\phi_1$={phi_1:.3f}", rf"$\phi_2$={phi_2:.3f}"])
    ax2.set_ylim(0.5, 2.5)
    ax2.set_xlim([min(g00, g11) - 0.1, max(g00, g11) + 0.1])

    ax2.plot([g00, g00], [0, 1.125], 'k--', alpha = 0.5)
    ax2.plot([g00 + phi_1, g00 + phi_1], [0.875, 2.125], 'k--', alpha = 0.5)
    ax2.plot([g11, g11], [1.875, 2.5], 'k--', alpha = 0.5)

    width = [0, 0]
    width[0] = phi_1
    width[1] = phi_2
    start = [0, 0]
    start[0] = g00
    start[1] = phi_1 + g00
    ax2.barh(y=[1, 2], width=width, left=start, height=0.25, alpha = 0.75)

    ax2.text(g00 + 0.01, 0.6, r"$E[f(x)]$")
    ax2.text(g11 + 0.01, 2.4, r"$f(x)$")
    
# %%

# Visualisations for different lambdas
for regul_lambda in [5, 20, 100]:
    weights = attack_SHAP(data, v_2 - v_1, regul_lambda)
    phi_1 = np.dot(weights, v_1) / n_data
    phi_2 = np.dot(weights, v_2) / n_data
    plot_res(weights, [phi_1, phi_2], 1 / n_data * np.dot(weights, g["00"]), g["11"], regul_lambda)

# Uniform weights for reference
weights = np.ones((n_data))
phi_1 = np.dot(weights, v_1) / n_data
phi_2 = np.dot(weights, v_2) / n_data
plot_res(weights, [phi_1, phi_2], 1 / n_data * np.dot(weights, g["00"]), g["11"], "\infty")


# %%
# Wald test for fraud detection
mean = f(data).mean(0)
std = f(data).std(0)
x = np.linspace(-3, 3, 100)

# Compare E[f(x)] samples
n_samples = 200
unbiased_means = []
# repeat experiment
for _ in tqdm(range(5000)):
    unbiased_samples = data[np.random.choice(n_data, n_samples), :]
    unbiased_means.append(f(unbiased_samples).mean())

# Illustrate how the test works
plt.figure()
plt.plot(x, norm.pdf(x))
plt.hist(np.sqrt(n_samples) * (unbiased_means - mean) / std, bins = 30,
                                                   alpha = 0.5, density = True)
plt.show()
del unbiased_means 

# %% Repeat Wald/KS tests to detect biased samples

# detections
detections_wald = []
detections_wald_ref = []
detections_KS = []

# parameters
significance = 0.01
n_repetitions = 500
n_samples = 200
lamb_space = np.logspace(np.log10(3), 3, 20)

# Experiment
for lamb in lamb_space:
    detections_KS.append(0)
    biased_means = []
    # compute biased weights
    weights = attack_SHAP(data, v_2 - v_1, lamb)
    # Repeat experiment
    for _ in tqdm(range(n_repetitions)):
        # biased sampling
        unbiased_samples = data[np.random.choice(n_data, n_samples), :]
        biased_samples = data[np.random.choice(n_data, n_samples, p=weights/n_data), :]
        
        # KS test statistics
        _, p_val_1 = ks_2samp(biased_samples[:, 0], unbiased_samples[:, 0])
        _, p_val_2 = ks_2samp(biased_samples[:, 1], unbiased_samples[:, 1])
        # KS detection
        detections_KS[-1] += int( (p_val_1) < significance/2 and \
                                  (p_val_2) < significance/2 )
        
        # Compute mean of E_nu[f]
        biased_means.append(f(biased_samples).mean())
    
    detections_KS[-1] *= 100 / n_repetitions
    # Wald statistic
    W = np.sqrt(n_samples) * (np.array(biased_means) - mean) / std
    # Wald detection
    detections_wald.append(100 * np.mean(np.abs(W) > -norm.ppf(significance/2) ) )

# %%
detections_wald = np.array(detections_wald)
bandW = np.sqrt(detections_wald * (100 - detections_wald) / n_repetitions)

detections_KS = np.array(detections_KS)
bandKS = np.sqrt(detections_KS * (100 - detections_KS) / n_repetitions)

# Illustrate how the test works
plt.figure()
plt.plot(lamb_space, detections_wald, 'b-o', label=r"Wald$(\nu)$")
plt.fill_between(lamb_space, detections_wald + bandW, detections_wald - bandW,
                                                          color='b', alpha=0.2)
plt.plot(lamb_space, detections_KS, 'r->', label="KS")
plt.fill_between(lamb_space, detections_KS + bandKS, detections_KS - bandKS,
                                                          color='r', alpha=0.2)
plt.xlabel(r"$\lambda$")
plt.xscale('log')
plt.ylabel("Detection Rate (\%)")
plt.legend()
plt.show()

