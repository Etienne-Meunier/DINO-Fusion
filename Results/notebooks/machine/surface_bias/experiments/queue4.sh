#!/bin/zsh
# Fourth round : C7 (eta 100) + the TS constraint restricted to the surface T and S channels, at the TS settings
# (eta 0.1, lbd0 1, eps = max over the training states), always on or only for t < 500.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
python run.py --variant c7_zm0 --eta 100 0.1 --eps 8.6e-8 2.5e-4 --lbd0 1 1 --name c7zm0_eta100_0.1 > logs/c7zm0_eta100_0.1.log 2>&1
python run.py --variant c7_zm0 --eta 100 0.1 --eps 8.6e-8 2.5e-4 --lbd0 1 1 --t_on 1000 500 --name c7zm0_eta100_0.1_ton500 > logs/c7zm0_eta100_0.1_ton500.log 2>&1
python run.py --variant c7_zm0 --eta 100 1 --eps 8.6e-8 2.5e-4 --lbd0 1 1 --name c7zm0_eta100_1 > logs/c7zm0_eta100_1.log 2>&1
