""" Test the GeneticAlgorithm with XGBoost on Adult """
# %% 
# Imports
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})
rc('text', usetex=True)
import os, sys
sys.path.append(os.path.abspath(".."))

from src.genetic import GeneticAlgorithm
from src.utils import audit_detection, plot_CDFs

import sys
sys.path.append("/home/gabriel/Desktop/POLY/PHD/Research/Repositories/shap")
import shap

# %%
# The company has private access to a dataset
X, y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]
categorical = ["Workclass", "MaritalStatus", "Occupation", "Relationship", "Race", 
               "Sex", "Country"]
# The company fits a model
model = xgboost.XGBClassifier(random_state=0, eval_metric="error")
model.fit(X, y)

# %%
# The company shares f(D_0), f(D_1) to the audit
D_1 = X[X["Sex"]==1]
f_D_1 = model.predict(D_1, output_margin=True)
D_0 = X[X["Sex"]==0]
f_D_0 = model.predict(D_0, output_margin=True)

# %%
parity = f_D_0.mean() - f_D_1.mean()
print(f"The demographic parity is : {parity: .3f}")

# %%

M = 100
# Subsample foreground honestly
S_0 = D_0.iloc[-M:]
f_S_0 = f_D_0[-M:]
# Subsample background honestly
S_1 = D_1.iloc[:M]
f_S_1 = f_D_1[:M]

# %% 

# Sensitive index
s_idx = 7
# Peturbate the background
alg = GeneticAlgorithm(model, S_0, S_1, f_S_0, f_D_0, f_D_1, s_idx, constant=categorical, 
                       pop_count=30, mutation_with_constraints=False)

# %%
alg.fool_aim(max_iter=200, random_state=0)
alg.plot_losses()
f_S_1 = model.predict(alg.S_1_prime, output_margin=True)

# %%

plot_CDFs(f_D_0, f_D_1, f_S_0, f_S_1)
plt.show()

# %%
# Detection
detection = audit_detection(f_D_0, f_D_1, f_S_0, f_S_1)
print(f"Audit Detection : {detection==1}")

# %%
