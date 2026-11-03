#!/bin/zsh
# Third round : surface-mean constraint in sqrt mode (bounded push), C7 raw.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
for s in 1e-4 1e-5 ; do
  python run.py --variant c7_surf --eta 100 $s --eps 8.6e-8 9 --lbd0 1 0 --mode raw sqrt --name c7surf_eta100_sqrt$s > logs/c7surf_eta100_sqrt$s.log 2>&1
done
