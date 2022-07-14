import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 20
mp.rcParams['font.family'] = 'serif'

#from matplotlib import font_manager
# My path to Times
#font_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
#font_manager.fontManager.addfont(font_path)
#prop = font_manager.FontProperties(fname=font_path)
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = prop.get_name()
#plt.rcParams['font.size'] = 20


if __name__ == "__main__":

    df = pd.read_csv(os.path.join("attacks", "results.csv"))
    df["rank_diff"] = df["final_rank"] - df["init_rank"]
    df["abs_diff"] = 100 * (df["init_abs"] - df["final_abs"]) / df["init_abs"]
    grouped = df.groupby('dataset')

    # Increases in Rank
    plt.figure()
    for i, (name, group) in enumerate(grouped):
        rank_diffs = group["rank_diff"].value_counts()
        width = 0.3
        plt.bar(rank_diffs.index + (i-0.5)*width, rank_diffs, width=width, label=name)

    plt.xticks(np.arange(np.max(df["rank_diff"])+1))
    plt.xlabel("Rank Increase")
    plt.ylabel("Number of Experiments")
    plt.legend(loc="best", fontsize=16)
    plt.savefig(os.path.join("Images", f"rank_results.pdf"), bbox_inches='tight')

    # Decreases in amplitude
    plt.figure()
    for i, (name, group) in enumerate(grouped):
        abs_diffs = group["abs_diff"]
        plt.hist(abs_diffs, bins=np.arange(0, 101, 10), label=name, alpha=0.4)

    plt.xlim(0, 100)
    plt.xlabel("Relative Amplitude Decrease (\%)")
    plt.ylabel("Number of Experiments")
    plt.legend(fontsize=16)
    plt.savefig(os.path.join("Images", f"abs_results.pdf"), bbox_inches='tight')
