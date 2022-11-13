""" Plot the result of all attacks, i.e. Figure 5 in the paper """
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 20
mp.rcParams['font.family'] = 'serif'
import seaborn as sns

#from matplotlib import font_manager
## My path to Times
#font_path = '/usr/share/fonts/truetype/msttcorefonts/times.ttf'
#font_manager.fontManager.addfont(font_path)
#prop = font_manager.FontProperties(fname=font_path)
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = prop.get_name()
#plt.rcParams['font.size'] = 20

def get_cdf(data):
    data = np.concatenate(([0], np.sort(data), [100]))
    cdf = np.concatenate((np.linspace(0, 1, data.shape[0]-1), [1]))
    return data, cdf


if __name__ == "__main__":

    df = pd.read_csv(os.path.join("attacks", "results.csv"))
    df["rank_diff"] = df["final_rank"] - df["init_rank"]
    df.loc[df["rank_diff"] > 8, "rank_diff"] = 9
    df["brute_abs_diff"] = 100 * (df["init_abs"] - df["brute_abs"]) / df["init_abs"]
    df["fool_abs_diff"] = 100 * (df["init_abs"] - df["final_abs"]) / df["init_abs"]
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

    # # Increases in Rank
    # plt.figure()
    # for i, (name, group) in enumerate(grouped):
    #     rank_diffs = group["rank_diff"].value_counts()
    #     width = 0.15
    #     plt.bar(rank_diffs.index + (i-1.5)*width, rank_diffs, width=width, label=name)

    # plt.xticks([-1] + list(range(10)), ['-1','0','1','2','3','4','5','6','7','8','$>$8'])
    # plt.xlabel("Rank Increase")
    # plt.ylabel("Number of Attacks")
    # plt.legend(loc="upper left", bbox_to_anchor=(0.53,1), fontsize=15, framealpha=0.75)
    # plt.savefig(os.path.join("Images", f"rank_results.pdf"), bbox_inches='tight')


    df_copy = [ pd.DataFrame(np.column_stack((df["dataset"], df["brute_abs_diff"], len(df)*["Brute"])),
                            columns=["dataset", "diff", "method"]),
                pd.DataFrame(np.column_stack((df["dataset"], df["fool_abs_diff"], len(df)*["Ours"])),
                            columns=["dataset", "diff", "method"]) ]
    df_copy = pd.concat(df_copy)

    # Decreases in relative amplitude
    order = ["COMPAS", "Adult", "Marketing" ,"C\&C"]
    sns.boxplot(x="dataset", y="diff", hue="method", data=df_copy, width=0.45, order=order)

    # plt.figure()
    # colors = ['b', 'orange', 'g', 'r']
    # for i, (name, group) in enumerate(grouped):
    #     plt.step(*get_cdf(group["brute_abs_diff"]), '--', linewidth=2, color=colors[i], where='post')
    #     plt.step(*get_cdf(group["fool_abs_diff"]), linewidth=2, color=colors[i], where='post', label=name)
    plt.xlabel("Dataset")
    plt.ylabel("Relative Amplitude Decrease (\%)")
    plt.legend(fontsize=15, framealpha=1, loc="upper right", bbox_to_anchor=(1,0.8))
    plt.savefig(os.path.join("Images", f"abs_results.pdf"), bbox_inches='tight')
