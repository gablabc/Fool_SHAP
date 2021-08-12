# %% Imports 

import numpy as np
import matplotlib.pyplot as plt
import time, os
from sklearn.datasets import dump_svmlight_file
import subprocess

import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 5
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
    fig = plt.figure(figsize = (9, 2.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contourf(xx, yy, model_grid, 20, cmap='Blues', alpha=.8)
    ax1.scatter(data[:, 0], data[:, 1], c = 'k', s = weights)
    ax1.plot(instance[0, 0], instance[0, 1], 'r*', markersize = 10,
                                                label = r"Instance $\textbf{x}$")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title(r"Biased Reference with $\lambda=" + f"{int(regul_lambda)}$")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels([rf"$\phi_1$={phi_1:.2f}", rf"$\phi_2$={phi_2:.2f}"])
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
    

for regul_lambda in [5, 20, 100]:
    weights = attack_SHAP(data, v_2 - v_1, regul_lambda)
    phi_1 = np.dot(weights, v_1) / n_data
    phi_2 = np.dot(weights, v_2) / n_data
    plot_res(weights, [phi_1, phi_2], 1 / n_data * np.dot(weights, g["00"]), g["11"], regul_lambda)


# Compare E[f(x)] samples
weights = attack_SHAP(data, v_2 - v_1, 20)
n_samples = 100
unbiased_mean = []
biased_mean = []
for _ in range(50):
    unbiased_samples = data[np.random.choice(n_data, n_samples), :]
    biased_samples = data[np.random.choice(n_data, n_samples, p=weights / n_data), :]
    unbiased_mean.append(f(unbiased_samples).mean())
    biased_mean.append(f(biased_samples).mean())

plt.figure(figsize=(4, 4))
plt.hist(unbiased_mean, alpha = 0.2, density = True)
plt.hist(biased_mean, alpha = 0.2, density = True)
plt.show()
