# Code base for Fairness constraint to bandit-based online Influence Maximization problem

This is the code reposiratory for paper: "Fairness constraint to bandit-based online Influence
Maximization problem".

## Experiments and previous plots:

alpha is 1.2 in these experiments.

We also tried alpha=0.5 in some case.

Experiments on NBA dataset:

seed size is 100, run 100 iteration,  regret plots can be found in ./presentation/NBA folder.

Initial propagation probability in edge allocated as 2.

Experiments on german dataset:

seed size is 300, run 100 iteration,  regret plots can be found in ./presentation/german folder

Initial propagation probability in edge allocated as 2.

Experiments on

seed size is 300, run 100 iteration, regret plots can be found in ./presentation/pokec folder

Initial propagation probability in edge allocated as 2.

## Brief explanation of code:

Code reposiratory is based on github repo:

Bandit repo: [Matrix-Factorization-Bandit/IMFB-KDD2019: Code for the experiments of Matrix Factorization Bandit (github.com)](https://github.com/Matrix-Factorization-Bandit/IMFB-KDD2019)

Offline Oracle repo (Frank-Wolfe): [bwilder0/fair_influmax_code_release: Code for the paper &#34;Group-Fairness in Influence Maximization&#34; (github.com)](https://github.com/bwilder0/fair_influmax_code_release)

Offline Oracle repo(Mixed Integer Programming):

## Python Environment:

python==3.6, see the requirements.txt

## How to run:

For parameter setting, refer to conf.py

For run on the configuration, refer to IMBandit.py
