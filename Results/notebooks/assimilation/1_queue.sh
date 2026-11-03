#!/bin/zsh
# Phase 1 runs of the assimilation study, sequential, most informative first; a run with a generated.npy is skipped.
# Flags of the centred configs from machine/surface_bias/experiments/queue14.sh. Logs in OUT1/logs/<name>.log.
export OCEANDATA=/Volumes/LoCe/oceandata/
OUT1=$OCEANDATA/models/dino-fusion/tav0h83b/inference/assimilation/phase1
PY=/Users/emeunier/miniforge3/envs/diffusion/bin/python
BATCH=${BATCH:-16}
cd "${0:A:h}"
mkdir -p $OUT1/logs
r() { local name=$1; shift; [[ -f $OUT1/$name/generated.npy ]] && return; $PY 1_generate_sst.py --name $name --batch $BATCH "$@" > $OUT1/logs/$name.log 2>&1 }
r summer_none --season summer --variant none --eta 0 --eps 0
r summer_c7 --season summer --variant c7 --eta 10 --eps 8.6e-8
r summer_ts --season summer --variant zero_mean --eta 1 --eps 0.01
r winter_none --season winter --variant none --eta 0 --eps 0
r winter_c7 --season winter --variant c7 --eta 10 --eps 8.6e-8
r winter_ts --season winter --variant zero_mean --eta 1 --eps 0.01
