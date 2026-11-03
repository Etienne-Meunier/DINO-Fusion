#!/bin/zsh
# Eleventh round : the profile + C7 stop-grad runs turn the surface salty as the C7 strength grows (S0 35.7 at 200,
# 36.0 at 300). Same constraints with the gradient through temperature only (salinity channels detached), still no
# data information ; C7 alone T-only for reference. Then b16 of the screened data-free settings.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
PT() { r proftopT_p$1_t$2 --variant proftop_T --eta $1 $2 --eps 0 8.6e-8 --lbd0 1 1 }
PT 100 100
PT 300 100
PT 100 200
PT 300 300
r c7T_c7050 --variant c7_T --eta 50 --eps 8.6e-8
r c7T_c7100 --variant c7_T --eta 100 --eps 8.6e-8
r b16_proftop_p300_t100 --batch 16 --variant profile_topfree --eta 300 100 --eps 0 8.6e-8 --lbd0 1 1
python evaluate.py 'proftop*' 'c7T*' > logs/evaluate11.log 2>&1
