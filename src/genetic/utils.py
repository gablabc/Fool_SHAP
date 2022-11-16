import numpy as np


def check_early_stopping(df, epsilon, stop_iter=10):
    if len(df['loss']) >= stop_iter:
        relative_change = np.abs(df['loss'][-stop_iter] - df['loss'][-1] / np.mean(df['loss'][-stop_iter:]))
        return relative_change < epsilon
    else:
        return False