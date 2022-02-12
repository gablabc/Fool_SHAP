# Fool SHAP by biasing the background distribution

Python code to fool SHAP, an extension of the technique introduced in 
* K. Fukuchi, S. Hara, T. Maehara, [Faking Fairness via Stealthily Biased Sampling](https://arxiv.org/abs/1901.08291). to appear in AAAI'20 Special Track on Artificial Intelligence for Social Impact (AISI).

### Requirements ###
- g++ (-std=c++11)
- [lemon](https://lemon.cs.elte.hu/trac/lemon/)

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
