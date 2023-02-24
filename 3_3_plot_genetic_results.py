import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mp
mp.rcParams['text.usetex'] = True
mp.rcParams['font.size'] = 20
mp.rcParams['font.family'] = 'serif'
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#0000FF", "#FF0000"])

import argparse
from scipy.signal import savgol_filter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import get_foreground_background, get_data


def smooth(y, window):
    """ Smooth the signal """
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    N = len(y)
    box = np.arange(1, window // 2+1).astype(np.float64)
    box = np.concatenate( (box, box[::-1]))
    box /= box.sum()
    y_smooth = np.convolve(y, box, mode='full')[:N]
    return y_smooth

# def smooth(y, window):
#     return savgol_filter(y, window, 3)


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='marketing', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='xgb', help='Model: mlp, rf, gbt, xgb')
    parser.add_argument('--max_iter', type=int, default=400, help="Maximum number of iterations of genetic alg")
    parser.add_argument('--window_phi', type=int, default=10, help="Smoothing window of phi signal")
    parser.add_argument('--window_detect', type=int, default=10, help="Smoothing window of detect signal")
    args = parser.parse_args()


    ############# Plot the Iterations ###############
    dfs = []
    for seed in range(5):
        filename = os.path.join("attacks", "Genetic", f"{args.model}_{args.dataset}_rseed_{seed}.csv")
        if os.path.exists(filename):
            df = pd.read_csv(os.path.join("attacks", "Genetic", f"{args.model}_{args.dataset}_rseed_{seed}.csv"))
            if df.shape[0] <= args.max_iter:
                pad_iter = np.arange(df['iter'].iloc[-1]+1, args.max_iter+1)
                pad_loss = df['loss'].iloc[-1] * np.ones(len(pad_iter))
                pad_detect = df['detection'].iloc[-1] * np.ones(len(pad_iter))
                pad = pd.DataFrame(np.column_stack((pad_iter, pad_loss, pad_detect)), columns=df.columns)
                df = pd.concat((df, pad), axis=0, ignore_index=True)
            df["Relative Amplitude Decrease"] = 100 * (df.loc[0, 'loss'] - df['loss']) / df.loc[0, 'loss']
            df['seed'] = seed
            dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    dfs_mean = dfs.groupby('iter').mean()
    dfs_mean['detection'] *= 100
    dfs_std = dfs.groupby('iter').std()
    dfs_std['detection'] *= 100 / np.sqrt(10)

    # Curves of Shapley values
    fig, ax = plt.subplots()
    iters = dfs_mean.index
    mean_signal = smooth(dfs_mean["Relative Amplitude Decrease"], args.window_phi)
    std_signal = smooth(dfs_std["Relative Amplitude Decrease"], args.window_phi)
    ax.plot(iters, mean_signal, 'r-')
    ax.fill_between(iters, mean_signal+std_signal, mean_signal-std_signal, color='r', alpha=0.2)
    # set x-axis label
    ax.set_xlabel("Iterations")
    ax.set_xlim(0, args.max_iter)
    # set y-axis label
    ax.set_ylabel("Relative Amplitude Decrease", color="red")
    ax.set_ylim(0)
    ax.tick_params(axis='y', labelcolor="red")
    ax2 = ax.twinx()
    mean_signal = smooth(dfs_mean["detection"], args.window_detect)
    std_signal = smooth(dfs_std["detection"], args.window_detect)
    ax2.plot(iters, mean_signal, 'b-')
    ax2.fill_between(iters, mean_signal+std_signal, 
                            mean_signal-std_signal, color='b', alpha=0.2)
    ax2.set_ylabel("Detection Rate %", color="blue")
    ax2.set_ylim(0)
    ax2.tick_params(axis='y', labelcolor="blue")
    figure_path = os.path.join(f"Images", args.dataset, args.model)
    plt.savefig(os.path.join(figure_path, f"genetic_iterations.pdf"), bbox_inches='tight')



    ############# Plot the Fake Dataset ###############
    if args.model == "xgb":
        # Get the real data
        filepath = os.path.join("datasets", "preprocessed", args.dataset)
        X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                            get_data(args.dataset, args.model, rseed=0)

        # Get foreground and background
        _, D_1 = get_foreground_background(X_split, args.dataset)
        N_1 = D_1.shape[0]

        # OHE+Ordinally encode B and F
        D_1 = ordinal_encoder.transform(D_1)

        if isinstance(ordinal_encoder, ColumnTransformer):
            n_numerical_features = len(ordinal_encoder.transformers_[0][2])
        else:
            n_numerical_features = D_1.shape[1]

        filename = os.path.join("attacks", "Genetic", f"{args.model}_{args.dataset}_rseed_0.npy")
        S_1 = np.load(filename)

        is_fake = np.concatenate( (np.zeros(N_1), np.ones(S_1.shape[0])) )
        D_1 = np.vstack((D_1, S_1))[:, :n_numerical_features]

        pca = Pipeline([('standard', StandardScaler()),
                        ('pca', PCA(n_components=2))])
        D_1_proj = pca.fit_transform(D_1)

        plt.figure()
        max_show = min(N_1, 1000)
        plt.scatter(D_1_proj[:max_show, 0], D_1_proj[:max_show, 1], c='blue',alpha=0.8)
        plt.scatter(D_1_proj[N_1:, 0], D_1_proj[N_1:, 1], c='red', alpha=0.8)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(os.path.join(figure_path, f"genetic_PCA.pdf"), bbox_inches='tight')