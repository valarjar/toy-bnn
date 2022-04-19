# Installation and usage guide

```
git clone --recursive https://github.com/valarjar/bnn.git

conda create --name toy-bnn python=3.6.9

conda activate toy-bnn

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

pip install numpy==1.19.2 matplotlib==3.1.1 scipy==1.3.1 pandas==1.0.2 pyro-ppl==1.6.0 seaborn==0.10.0
```

```
conda create --name toy-bnn python=3.6.9

conda activate toy-bnn

pip install numpy==1.19.2

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

pip install matplotlib==3.1.1

pip install scipy==1.3.1

pip install pandas==1.0.2

pip install pyro-ppl==1.6.0

pip install seaborn==0.10.0
```

## Example 1: Bayesian MNIST

```
cd 1_bayesian_mnist

# Figure 2

python viExperiment.py --savedir ./pre_models --notrain
```

## Example 2: Sparse Measure

```
cd 2_sparse_measure

# Figure 5

python exploration.py   

# Figure 6 and Figure 7

python viExperiment.py --nepochs 200
```

## Example 3: Paperfold

MCMC-model
```
python mcmc_experiment.py --constrained  --samples 200 --warmup 500 --numchains 5  --o mcmc_samples.pkl

python plots.py mcmc_samples.pkl

```
vi-model

```
python vi_experiment.py --constrained --trainsteps 5000 --o vi_samples.pkl

python plots.py vi_samples.pkl

```
ensemble-model

```
python vi_experiment.py --constrained --pointestimate --samples 200 --trainsteps 5000 --numnetworks 5  --o ensemble_samples.pkl                                   

python plots.py ensemble_samples.pkl
```
vi+ensemble-model

```
python vi_experiment.py --constrained  --samples 200 --trainsteps 5000 --numnetworks 5  --o vi_ensemble_samples.pkl

python plots.py vi_ensemble_samples.pkl
```
