# Fool SHAP by biasing the background distribution
## Description

Python code to fool SHAP, an extension of the technique introduced in 
* K. Fukuchi, S. Hara, T. Maehara, [Faking Fairness via Stealthily Biased Sampling](https://arxiv.org/abs/1901.08291). to appear in AAAI'20 Special Track on Artificial Intelligence for Social Impact (AISI).

### Requirements ###
- g++ (-std=c++11)
- [lemon](https://lemon.cs.elte.hu/trac/lemon/)

-----------------
## Installation

### 1. Install LEMON ###

```
sudo apt install g++ make cmake 
wget http://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
tar xvzf lemon-1.3.1.tar.gz
cd lemon-1.3.1
mkdir build
cd build
cmake ..
make
sudo make install
```

### 2. Make files ###

```
cd shap_biasing
make
```

### 3. Install SHAP Fork ###

This repository relies on a `hacked` version of SHAP available through
a forked repository.

```
git clone https://github.com/gablabc/shap.git
cd shap
git checkout biased_sampling
```

Now, you must make sure that the python interpreter is able to access the
forked SHAP repository. In Linux, this is done via the command
```
export PYTHONPATH=<path to shap>:${PYTHONPATH}
```
which can be added to the `.bashrc` file.

-----------------
## Experiments

### Data Preprocessing
Preprocess the data by running
```
cd datasets
./main.sh
cd ..
```

### Model Training
Train all models by running
```
./train_all.sh
```
This script can be made more efficient if its for-loops are parallelized between multiple machines.

### Conduct the attack
The attack is conducted in three steps. First of, the malicious company must extract Phi coefficients
by running a hacked version a `SHAP`. This is done via the script
```
    python3 1_compute_Phis.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom> --background_seed=<custom>
```
where the parameter ``background_size`` controls how many points are used when solving the MCF to compute non-uniform weights of
the background distribution. Setting it to `-1` means use all the background points. This script can be run in parralel different machines for different values of ``background_seed``.

The second step of the attack is to compute the weights by solving a MCF for different values of regularization parameter $\lambda=\lambda_\text{min},\ldots,\lambda_\text{max}$

```
    python3 2_compute_weights.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom> --background_seed=<custom> --min_log=-1 --max_log=2
```
The resulting weights for different values of ``background_seed`` will be averaged to provide the final weights. The final step is to sample the misleading subsets $S_0',S_1'$ and provide them to the audit

```
    python3 3_final_attack.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom>
```