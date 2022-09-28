# Fool SHAP by biasing the background distribution
## Description

Python code to manipulate `SHAP` feature attributions, an extension of the steahlity biased sampling technique introduced in 
* K. Fukuchi, S. Hara, T. Maehara, [Faking Fairness via Stealthily Biased Sampling](https://arxiv.org/abs/1901.08291). AAAI'20 Special Track on Artificial Intelligence for Social Impact (AISI).

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
Build the custom treeshap implementation that can extract the $\hat{\Phi}$ coefficients.
```
cd treeshap
make
cd ..
```
Then build the MCF algorithm to compute the non-uniform weights $\omega$ on the
background distribution
```
cd shap_biasing
make
cd ..
```
### 3. Install SHAP Fork ###

To attack the `ExactExplainer`, we use monkey-patching via afork of the `SHAP` repository
```
git clone https://github.com/gablabc/shap.git
cd shap
git checkout biased_sampling
cd ..
```
If you do not install `SHAP` in this directory, you must tell the Python interpreter
where to look for it. In Linux, this is done via the command
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
This script can be made more efficient if its for-loops are parallelized 
between multiple machines.

### Conduct the attack
The attack is conducted in three steps. First of, the malicious company must extract $\hat{\Phi}$ coefficients. This is done via the Python script
```
    python3 2_1_compute_Phis.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom> --background_seed=<custom>
```
where the parameter ``background_size`` controls how many points are used when solving the MCF to compute non-uniform weights of
the background distribution. Setting it to `-1` means use all the background points. This script can be run in parralel different machines for different values of ``background_seed``.

The second step of the attack is to compute the weights by solving a MCF for different values of regularization parameter $\lambda=\lambda_\text{min},\ldots,\lambda_\text{max}$

```
    python3 2_2_compute_weights.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom> --background_seed=<custom> --min_log=-1 --max_log=2
```
The resulting weights for different values of ``background_seed`` will be averaged to provide the final weights. The final step is to sample the misleading subsets $S_0',S_1'$ and provide them to the audit

```
    python3 2_3_final_attack.py --dataset=adult_income --model=xgb --rseed=0 --background_size=<custom> --save
```

### Simple Examples of Attacks

The scripts `0_example_adult.py` and `0_example_employment.py` show basic examples of the attack on adult-income
and a toy employment dataset. These scripts are meant as basic tutorials of the attack on `SHAP` feature
attributions.