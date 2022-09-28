""" Plot the result of all attacks, i.e. Figure 5 in the paper """
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 20
mp.rcParams['font.family'] = 'serif'

#from matplotlib import font_manager
## My path to Times
#font_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
#font_manager.fontManager.addfont(font_path)
#prop = font_manager.FontProperties(fname=font_path)
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = prop.get_name()
#plt.rcParams['font.size'] = 20


if __name__ == "__main__":

    df = pd.read_csv(os.path.join("attacks", "results.csv"))
    df["rank_diff"] = df["final_rank"] - df["init_rank"]
    df.loc[df["rank_diff"] > 8, "rank_diff"] = 9
    #df["abs_diff"] = 100 * (df["init_abs"] - df["final_abs"]) / df["init_abs"]
    grouped = df.groupby('dataset')

    # ECDF
    # plt.figure()
    # for i, (name, group) in enumerate(grouped):
    #     rank_diffs = group["rank_diff"]
    #     plt.hist(rank_diffs, density=True, cumulative=True, histtype="step", linewidth=2, label=name)
    # #plt.xticks([0, 1, 2, 3, 4, 5, 6], ['0','1','2','3','4','5','$>$5'])
    # plt.yticks([0, 0.25, 0.5, 0.75, 1])
    # plt.grid('on', alpha=0.5)
    # plt.xlabel("Rank Increase")
    # plt.ylabel("Ratio of Attacks")
    # plt.legend(loc="lower right", fontsize=15, framealpha=1)
    # plt.savefig(os.path.join("Images", f"rank_results.pdf"), bbox_inches='tight')

    # Increases in Rank
    plt.figure()
    for i, (name, group) in enumerate(grouped):
        rank_diffs = group["rank_diff"].value_counts()
        width = 0.15
        plt.bar(rank_diffs.index + (i-1.5)*width, rank_diffs, width=width, label=name)

    plt.xticks([-1] + list(range(10)), ['-1','0','1','2','3','4','5','6','7','8','$>$8'])
    plt.xlabel("Rank Increase")
    plt.ylabel("Number of Attacks")
    plt.legend(loc="upper left", bbox_to_anchor=(0.53,1), fontsize=15, framealpha=0.75)
    plt.savefig(os.path.join("Images", f"rank_results.pdf"), bbox_inches='tight')

    # # Decreases in amplitude
    # plt.figure()
    # for i, (name, group) in enumerate(grouped):
    #     abs_diffs = group["abs_diff"]
    #     plt.hist(abs_diffs, bins=np.arange(0, 101, 10), label=name, alpha=0.4)

    # plt.xlim(0, 100)
    # plt.xlabel("Relative Amplitude Decrease (\%)")
    # plt.ylabel("Number of Experiments")
    # plt.legend(fontsize=15, framealpha=1)
    # plt.savefig(os.path.join("Images", f"abs_results.pdf"), bbox_inches='tight')
