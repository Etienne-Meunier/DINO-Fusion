#!/bin/zsh
# Thirteenth round, centred sampler (run.py --centred), b8 seed 0. The zero_mean loss of centred_none is 0.09 (data
# 0.009, paper eps 0.01), mostly deep salinity : zero_mean eta sweep at eps 0.01, zero_mean + C7 10, old-sampler zero_mean
# at b8 for reference ; C7 + surface T,S mean ; C7 with the level mean removed from the guidance step (--centre_grad).
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
for e in 0.03 0.1 0.3 1 3; do r centred_zm_e$e --variant zero_mean --eta $e --eps 0.01 --centred; done
r zm_b8 --variant zero_mean --eta 0.1 --eps 0.01
for e in 0.1 1; do r centred_zm_e${e}_c7010 --variant zero_mean_c7 --eta $e 10 --eps 0.01 8.6e-8 --lbd0 1 1 --centred; done
for e in 10 30 100; do r centred_surf_c7$(printf %03d $e) --variant c7_surf --eta $e 1e-4 --eps 8.6e-8 9 --lbd0 1 0 --mode raw sqrt --centred; done
for e in 10 30 100; do r centred_cg_c7$(printf %03d $e) --variant c7 --eta $e --eps 8.6e-8 --centred --centre_grad; done
python evaluate.py 'centred*' 'zm_b8' > logs/evaluate13.log 2>&1
