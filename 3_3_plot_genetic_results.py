import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 20
mp.rcParams['font.family'] = 'serif'
import seaborn as sns


if __name__ == "__main__":

    dfs = []
    for seed in range(10):
        df = pd.read_csv(os.path.join("attacks", "Genetic", f"xgb_genetic_rseed_{seed}.csv"))
        df["Relative Amplitude Decrease"] = 100 * (df.loc[0, 'loss'] - df['loss']) / df.loc[0, 'loss']
        df['seed'] = seed
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    dfs_mean = dfs.groupby('iter').mean()
    dfs_mean['detection'] *= 100
    dfs_std = dfs.groupby('iter').std()
    dfs_std['detection'] *= 100 / (np.sqrt(10))


    # Curves of Shapley values
    fig, ax = plt.subplots()
    iters = dfs_mean.index
    ax.plot(iters, dfs_mean["Relative Amplitude Decrease"], 'r-')
    ax.fill_between(iters, dfs_mean["Relative Amplitude Decrease"] + dfs_std["Relative Amplitude Decrease"], 
                            dfs_mean["Relative Amplitude Decrease"] - dfs_std["Relative Amplitude Decrease"], 
                            color='r', alpha=0.2)
    # set x-axis label
    ax.set_xlabel("Iterations")
    ax.set_xlim(0, 100)
    # set y-axis label
    ax.set_ylabel("Relative Amplitude Decrease", color="red")
    ax.tick_params(axis='y', labelcolor="red")
    ax2 = ax.twinx()
    ax2.plot(iters, dfs_mean["detection"], 'b-')
    ax2.fill_between(iters, dfs_mean["detection"] + dfs_std["detection"], 
                            dfs_mean["detection"] - dfs_std["detection"], 
                            color='b', alpha=0.2)
    ax2.set_ylabel("Detection Rate %", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    figure_path = os.path.join(f"Images", "adult_income", "xgb")
    plt.savefig(os.path.join(figure_path, f"genetic_iterations.pdf"), bbox_inches='tight')

