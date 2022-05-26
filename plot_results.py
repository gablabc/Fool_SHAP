import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager
# My path to Times
font_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = prop.get_name()
plt.rcParams['font.size'] = 20


if __name__ == "__main__":

    df = pd.read_csv(os.path.join("attacks", "results.csv"))
    df["rank_diff"] = df["final_rank"] - df["init_rank"]
    grouped = df.groupby('dataset')

    plt.figure()
    for i, (name, group) in enumerate(grouped):
        rank_diffs = group["rank_diff"].value_counts()
        width = 0.3
        plt.bar(rank_diffs.index + (i-0.5)*width, rank_diffs, width=width, label=name)

    plt.xticks(np.arange(np.max(df["rank_diff"])+1))
    plt.xlabel("Rank difference")
    plt.ylabel("Number of Experiments")
    plt.legend(framealpha=1)
    plt.savefig(os.path.join("Images", f"results.pdf"), bbox_inches='tight')
