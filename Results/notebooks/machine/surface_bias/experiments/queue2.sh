#!/bin/zsh
# Second round : surface-mean step size retuned, profile hinge combined with the cell hinges.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
python run.py --variant c7_surf --eta 100 1e-7 --eps 8.6e-8 9 --lbd0 1 0 --name c7surf_eta100_s1e-7 > logs/c7surf_eta100_s1e-7.log 2>&1
python run.py --variant c7_surf --eta 100 1e-8 --eps 8.6e-8 9 --lbd0 1 0 --name c7surf_eta100_s1e-8 > logs/c7surf_eta100_s1e-8.log 2>&1
python run.py --variant profile_topfree --eta 100 100 --eps 0 8.6e-8 --lbd0 1 1 --name proftop_eta100 > logs/proftop_eta100.log 2>&1
python run.py --variant profile_c7 --eta 100 30 --eps 0 8.6e-8 --lbd0 1 1 --name profc7_eta100_30 > logs/profc7_eta100_30.log 2>&1
