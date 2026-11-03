#!/bin/zsh
# First round of experimental runs, one at a time on the local GPU.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
mkdir -p logs
for spec in "profile 30 0 1 profile_eta30" "profile 100 0 1 profile_eta100" \
            "topfree 30 8.6e-8 1 topfree_eta30" "topfree 100 8.6e-8 1 topfree_eta100" ; do
  set -- ${=spec}
  python run.py --variant $1 --eta $2 --eps $3 --lbd0 $4 --name $5 > logs/$5.log 2>&1
done
python run.py --variant c7_surf --eta 100 1e-5 --eps 8.6e-8 9 --lbd0 1 0 --name c7surf_eta100_s1e-5 > logs/c7surf_eta100_s1e-5.log 2>&1
