#!/bin/zsh
# Phase 1 runs of the assimilation study on the standard sampler (1_generate_sst.py --sampler standard), sequential; a
# run with a generated.npy is skipped. Strengths, seed, batch and projections of the paper "optimal" runs
# (inference/optimal/{none,none_border0,negative_grad_density_huber_border0_30,zero_mean_border0}/*/args.json): none
# without any projection, B0 (no constraint), C7 and TS with BorderZero after every step. Logs in OUT/logs/<name>.log.
export OCEANDATA=/Volumes/LoCe/oceandata/
OUT=$OCEANDATA/models/dino-fusion/tav0h83b/inference/assimilation/phase1_standard
PY=/Users/emeunier/miniforge3/envs/diffusion/bin/python
BATCH=${BATCH:-16}
cd "${0:A:h}"
mkdir -p $OUT/logs
r() { local name=$1; shift; [[ -f $OUT/$name/generated.npy ]] && return; $PY 1_generate_sst.py --sampler standard --out $OUT --name $name --batch $BATCH --seed 0 "$@" > $OUT/logs/$name.log 2>&1 }
NONE=(--variant none --eta 0 --eps 0 --projection none)
B0=(--variant none --eta 0 --eps 0 --projection border_zero)
C7=(--variant c7 --eta 30 --eps 8.6e-8 --lbd0 1 --constraint_mode raw --projection border_zero)
TS=(--variant zero_mean --eta 0.1 --eps 0.01 --lbd0 1 --constraint_mode raw --projection border_zero)
for season in summer winter; do
  r ${season}_none --season $season $NONE
  r ${season}_C7 --season $season $C7
  r ${season}_TS --season $season $TS
done
for season in summer winter; do
  r ${season}_B0 --season $season $B0
done
