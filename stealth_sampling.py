import os
import numpy as np
import subprocess
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file
from utils import audit_detection
import time


def compute_weights(f_D_1, Phi_S0_zj, regul_lambda=10, epsilon=None, timeout=15.0):
    """
    Compute non-uniform weights for the background distribution

    Parameters
    --------------------
        f_D_1: (N_1, 1) array of predictions on D_1
        Phi_S0_zj: (N_1,) array of Shap coefficients for sensitive attribute. 
            If the shape is (N_1, s) with s>1 different sensitive attributes, 
            then the weights will be computed to reduce all of Shap values.
        regul_lambda: float that controls the trade-off between manipulation
            and proximity to the data

    Returns
    --------------------
        weights: (N_1, ) array of non-uniform weights on D_1
    """
    N_1 = len(f_D_1)
    while True:
        # Perturb the data a bit
        X = f_D_1 + 1e-5 * np.random.randn(*f_D_1.shape)
        
        # Compute bounds on weights
        bounds = N_1 * np.ones(N_1)
        # More than one sensitive attribute
        if Phi_S0_zj.ndim == 2:
            lin_coeffs = -Phi_S0_zj.mean(1)
            if epsilon is not None:
                unmanipulated_Phi = np.mean(Phi_S0_zj, 0)
                assert epsilon >= 0 and epsilon < N_1
                J = np.where( (Phi_S0_zj < unmanipulated_Phi).any(axis=1))[0]
                bounds[J] = epsilon
        # Only one sensitive attribute
        else:
            lin_coeffs = -Phi_S0_zj

        # Store files
        np.savetxt("bounds.txt", bounds, fmt="%d")
        dump_svmlight_file(X, lin_coeffs, f"input.txt")
        
        # Run the cpp program
        cmd = f"./shap_biasing/main input.txt bounds.txt {regul_lambda} > output.txt"
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
            print('killed')

    # Load results
    weights = np.loadtxt('./output.txt')
    os.remove('./input.txt' )
    os.remove('./bounds.txt' )
    os.remove('./output.txt')
    return weights



def explore_attack(f_D_0, f_S_0, f_D_1, Phi_S0_zj, s, lambda_min, lambda_max, 
                                        lambda_steps, significance, epsilon=None):
    """
    Searches the space of possible attacks

    Parameters
    --------------------
        f_D_0: (N_0, 1) array of predictions on D_0
        f_S_0: (M, 1) array of predictions on S_0
        f_D_1: (N_1, 1) array of predictions on D_1
        Phi_S0_zj: (N_1, d) array of Shap coefficients
        s: int sentitive attribute, List(int) sensitive attributes

    Returns
    --------------------
        lambd_space: array of all N_e lambda values explored
        weights: (N_e, N_1) array of non-uniform weights on D_1
        biased_shaps: (N_e, 100, d) array of Shapley values
        detections: (N_e,) array with number of detections out of 100 runs
    """
    weights = []
    biased_shaps = []
    detections = []
    N_1 = len(f_D_1)
    M = len(f_S_0)

    # Log-space search over possible attacls
    lambd_space = np.logspace(lambda_min, lambda_max, lambda_steps)
    for regul_lambda in tqdm(lambd_space):
        detections.append(0)
        biased_shaps.append([])

        # Attack !!!
        weights.append(compute_weights(f_D_1, Phi_S0_zj[:, s], regul_lambda, epsilon=epsilon))
        print(f"Spasity of weights : {np.mean(weights[-1] == 0) * 100}%")

        # Repeat the detection experiment
        for _ in range(100):
            # Biased sampling
            biased_idx = np.random.choice(N_1, M, p=weights[-1]/np.sum(weights[-1]))
            f_S_1 = f_D_1[biased_idx]
            
            detections[-1] += audit_detection(f_D_0, f_D_1, 
                                              f_S_0, f_S_1, significance)

            # New shap values
            biased_shaps[-1].append(np.mean(Phi_S0_zj[biased_idx], axis=0))

    # Convert to arrays for plots
    biased_shaps = np.array(biased_shaps)
    detections  = np.array(detections)
    weights = np.array(weights)

    return lambd_space, weights, biased_shaps, detections



def brute_force(f_D_0, f_S_0, f_D_1, Phi_S0_zj, s, significance, time_limit):
    """
    Searches the space of possible attacks via brute-force

    Parameters
    --------------------
        f_D_0: (N_0, 1) array of predictions on D_0
        f_S_0: (M, 1) array of predictions on S_0
        f_D_1: (N_1, 1) array of predictions on D_1
        Phi_S0_zj: (N_1, d) array of Shap coefficients
        s: int sentitive attribute, List(int) sensitive attributes
        time_limit: float time limit of the search in seconds

    Returns
    --------------------
        S_1: (M,) indices of instances in S_1'
    """
    
    N_1 = len(f_D_1)
    M = len(f_S_0)
    idx = np.arange(N_1)
    S_1 = np.random.choice(idx, M)
    min_abs_Phi_s = np.abs(Phi_S0_zj[S_1, s].mean())
    print(f"Init |Phi_s(f, S_0, S_1)| = {min_abs_Phi_s}")

    # Only search for a limited time
    start = time.time()
    step = 1
    while time.time() - start < time_limit:
        
        S_1_candidate = np.random.choice(idx, M)
        abs_Phi_s = np.abs(Phi_S0_zj[S_1_candidate, s].mean())
        f_S_1 = f_D_1[S_1_candidate]
        detection = audit_detection(f_D_0, f_D_1, f_S_0, f_S_1, significance)

        if abs_Phi_s < min_abs_Phi_s and not detection:
            S_1 = S_1_candidate
            min_abs_Phi_s = abs_Phi_s
        step +=1

    print(f"Searched for {step} steps")
    print(f"Smallest |Phi_s(f, S_0, S_1)| = {min_abs_Phi_s}\n")
    return S_1



def attack_SHAP_bootstrap(data, coeffs, regul_lambda=10, num_sample=10, num_process=2, seed=0, timeout=10.0):
    N = len(coeffs)
    assert N > 500
    c1 = 0
    c2 = 0
    weights = np.zeros(N)
    for _ in range(num_sample):
        while True:
            c2 += 1
            commands = []
            for p in range(num_process):
                prefix_p = f"{p:d}"
                np.random.seed(seed + c1)
                idx = np.random.permutation(N)[:500]
                np.random.seed(seed + c2)
                X = data[idx] + 1e-5 * np.random.randn(500)
                dump_svmlight_file(X, coeffs, f"./{prefix_p}_input.txt")
                cmd = f"./shap_biasing/main ./{prefix_p}_input.txt {regul_lambda} > ./{prefix_p}_output.txt"
                commands.append(cmd)
            procs = [subprocess.Popen('exec ' + cmd, shell=True) for cmd in commands]
            try:
                for p in procs:
                    p.wait(timeout)
                c1 += 1
                break
            except:
                for p in procs:
                    p.kill()
                print('killed')
        for p in range(num_process):
            prefix_p = f"{p:d}"
            dW = np.loadtxt(f"./{prefix_p}_output.txt")
            weights += dW
            os.remove(f"./{prefix_p}_input1.txt")
            os.remove(f"./{prefix_p}_output.txt")
    return weights



def stealth_sampling(X, K, path='./', prefix='tmp', timeout=10.0):
    assert len(X)==len(K)
    C = len(X)
    while True:
        Y = np.concatenate(X, axis=0)
        Y += 1e-10 * np.random.randn(*Y.shape)
        z = np.concatenate([i*np.ones(X[i].shape[0]) for i in range(C)])
        com = ' '.join('%d' % (k,) for k in K)
        dump_svmlight_file(Y, z, './%s_input.txt' % (prefix,), comment=com)
        cmd = '%sstealth-sampling/main ./%s_input.txt > %s_output.txt' % (path, prefix, prefix)
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
    res = np.loadtxt('./%s_output.txt' % (prefix,))
    p = np.split(res, np.cumsum([X[i].shape[0] for i in range(C)]))
    os.remove('./%s_input.txt' % (prefix,))
    os.remove('./%s_output.txt' % (prefix,))
    return np.array(p[:-1]) / (Y.shape[0] * sum(K))


def stealth_sampling_bootstrap(X, K, path='./', prefix='tmp', ratio=0.3, num_sample=10, num_process=2, seed=0, timeout=10.0):
    c = 0
    c2 = 0
    C = len(X)
    q = [1e-10*np.ones(X[j].shape[0]) for j in range(C)]
    for i in range(num_sample):
        while True:
            c2 += 1
            commands = []
            N = []
            M = []
            idx = []
            for p in range(num_process):
                prefix_p = '%s_p%05d' % (prefix, p)
                np.random.seed(seed+c)
                idx_p = []
                Ni = []
                Xi = []
                Ki = []
                for j in range(C):
                    nj = X[j].shape[0]
                    mj = int(np.round(ratio * nj))
                    idxj = np.random.permutation(nj)[:mj]
                    idx_p.append(idxj)
                    Ni.append(mj)
                    Xi.append(X[j][idxj, :])
                    Ki.append(int(np.round(ratio * K[j])))
                N.append(Ni)
                M.append(Ki)
                idx.append(idx_p)
                np.random.seed(seed+c2)
                Yi = np.concatenate(Xi, axis=0)
                Yi += 1e-10 * np.random.randn(*Yi.shape)
                zi = np.concatenate([j*np.ones(Xi[j].shape[0]) for j in range(C)])
                com = ' '.join('%d' % (k,) for k in Ki)
                dump_svmlight_file(Yi, zi, './%s_input.txt' % (prefix_p,), comment=com)
                cmd = '%sstealth-sampling/main ./%s_input.txt > %s_output.txt' % (path, prefix_p, prefix_p)
                commands.append(cmd)
            procs = [subprocess.Popen('exec ' + cmd, shell=True) for cmd in commands]
            try:
                for p in procs:
                    p.wait(timeout)
                c += 1
                break
            except:
                for p in procs:
                    p.kill()
                print('killed')
        for p in range(num_process):
            prefix_p = '%s_p%05d' % (prefix, p)
            res = np.loadtxt('./%s_output.txt' % (prefix_p,))
            res = np.split(res, np.cumsum([N[p][i] for i in range(C)]))[:-1]
            for j in range(C):
                q[j][idx[p][j]] += res[j] / (sum(N[p]) * sum(M[p]) * num_sample * num_process)
            os.remove('./%s_input.txt' % (prefix_p,))
            os.remove('./%s_output.txt' % (prefix_p,))
    qsum = np.sum(np.concatenate(q))
    q = [qq/qsum for qq in q]
    return q


def compute_wasserstein(X1, X2, path='./', prefix='tmp', timeout=10.0):
    assert X1.shape[1] == X2.shape[1]
    while True:
        dump_svmlight_file(X1+1e-10*np.random.randn(*X1.shape), np.zeros(X1.shape[0]), './%s_input1.txt' % (prefix,))
        dump_svmlight_file(X2+1e-10*np.random.randn(*X2.shape), np.zeros(X2.shape[0]), './%s_input2.txt' % (prefix,))
        cmd = '%swasserstein/main ./%s_input1.txt ./%s_input2.txt > %s_output.txt' % (path, prefix, prefix, prefix)
        proc = subprocess.Popen('exec ' + cmd, shell=True)
        try:
            proc.wait(timeout)
            break
        except:
            proc.kill()
            print('killed')
    
    d = np.loadtxt('./%s_output.txt' % (prefix,))
    os.remove('./%s_input1.txt' % (prefix,))
    os.remove('./%s_input2.txt' % (prefix,))
    os.remove('./%s_output.txt' % (prefix,))
    return d


def compute_wasserstein_bootstrap(X1, X2, n, path='./', prefix='tmp', num_sample=10, num_process=2, seed=0, timeout=10.0):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n = min([n, n1, n2])
    c = 0
    c2 = 0
    d = 0.0
    for i in range(num_sample):
        while True:
            c2 += 1
            commands = []
            for p in range(num_process):
                prefix_p = '%s_%05d' % (prefix, p)
                np.random.seed(seed+c)
                idx1 = np.random.permutation(n1)[:n]
                idx2 = np.random.permutation(n2)[:n]
                np.random.seed(seed+c2)
                dump_svmlight_file(X1[idx1, :]+1e-10*np.random.randn(idx1.size, X1.shape[1]), np.zeros(idx1.size), './%s_input1.txt' % (prefix_p,))
                dump_svmlight_file(X2[idx2, :]+1e-10*np.random.randn(idx2.size, X2.shape[1]), np.zeros(idx2.size), './%s_input2.txt' % (prefix_p,))
                cmd = '%swasserstein/main ./%s_input1.txt ./%s_input2.txt > %s_output.txt' % (path, prefix_p, prefix_p, prefix_p)
                commands.append(cmd)
            procs = [subprocess.Popen('exec ' + cmd, shell=True) for cmd in commands]
            try:
                for p in procs:
                    p.wait(timeout)
                c += 1
                break
            except:
                for p in procs:
                    p.kill()
                print('killed')
        for p in range(num_process):
            prefix_p = '%s_%05d' % (prefix, p)
            dp = np.loadtxt('./%s_output.txt' % (prefix_p,))
            d += dp / (num_sample * num_process)
            os.remove('./%s_input1.txt' % (prefix_p,))
            os.remove('./%s_input2.txt' % (prefix_p,))
            os.remove('./%s_output.txt' % (prefix_p,))
    return d