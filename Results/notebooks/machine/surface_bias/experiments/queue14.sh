#!/bin/zsh
# Fourteenth round, centred sampler at b32 seed 0 for the season and spread statistics : unconstrained, C7 10, zero_mean
# (eta 1, the sweep saturates from 0.3), zero_mean + C7 10, C7 30 + surface T,S mean. --centre_grad is dropped : it
# removes the level-mean part of the C7 step, which is what lowers E (E 2.2-2.4 at every eta, none 2.42).
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name --batch 32 "$@" > logs/$name.log 2>&1 }
r centred_none_b32 --variant none --eta 0 --eps 0 --centred
r centred_zm_e1_b32 --variant zero_mean --eta 1 --eps 0.01 --centred
r centred_zm_e1_c7010_b32 --variant zero_mean_c7 --eta 1 10 --eps 0.01 8.6e-8 --lbd0 1 1 --centred
r centred_c7010_b32 --variant c7 --eta 10 --eps 8.6e-8 --centred
r centred_surf_c7030_b32 --variant c7_surf --eta 30 1e-4 --eps 8.6e-8 9 --lbd0 1 0 --mode raw sqrt --centred
python evaluate.py '*_b32' > logs/evaluate14.log 2>&1
