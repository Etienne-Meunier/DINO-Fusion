#!/bin/zsh
# Tenth round : constraints with no data information (no surface statistics, no reference profile, only "density
# increases with depth"), screened at b8 seed 0 against the C7 eta curve. Profile hinge (on the sample's own mean
# profile) + C7 with the surface stop-grad, over both strengths ; profile hinge + C7 without the top pair.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
P() { r proftop_p$1_t$2 --variant profile_topfree --eta $1 $2 --eps 0 8.6e-8 --lbd0 1 1 }
Q() { r profskip_p$1_s$2 --variant profile_c7skip --eta $1 $2 --eps 0 8.6e-8 --lbd0 1 1 }
P 100 30
P 100 300
P 300 100
P 100 50
P 100 200
P 30 100
P 300 300
Q 100 100
Q 100 300
python evaluate.py 'proftop*' 'profskip*' 'profile_eta*' 'topfree*' > logs/evaluate10.log 2>&1
