#!/bin/bash
# Submit a miniveros job.  usage:
#   jobs/submit.sh extract        [extra extract_data.py args]
#   jobs/submit.sh train          <train.py args, e.g. --preset full --set data_file=$MV_DATA run_dir=$MV_WORK/runs/x>
#   jobs/submit.sh generate_eval  <run_dir> [n_samples, default 32]
# Account / QoS / constraint / log paths come from jz_env.sh next to this file (never from the tracked files).
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/jz_env.sh" ] || { echo "missing $HERE/jz_env.sh - copy jz_env.example.sh and fill it in" >&2; exit 1; }
source "$HERE/jz_env.sh"
JOB=$1; shift || true
mkdir -p "$MV_WORK/logs" "$MV_WORK/runs" "$MV_WORK/data"
export MV_ARGS="$*"
export MV_GIT_HASH=$(git -C "$MV_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)   # compute nodes may lack git
case "$JOB" in
  extract)
    sbatch --account="$MV_ACCOUNT_CPU" --qos="$MV_QOS_CPU" --partition="$MV_CPU_PARTITION" \
           --output="$MV_WORK/logs/extract_%j.out" --error="$MV_WORK/logs/extract_%j.out" --export=ALL "$HERE/extract.sbatch" ;;
  train)
    sbatch --account="$MV_ACCOUNT_GPU" --qos="$MV_QOS_GPU" --constraint="$MV_GPU_CONSTRAINT" \
           --output="$MV_WORK/logs/train_%j.out" --error="$MV_WORK/logs/train_%j.out" --export=ALL "$HERE/train.sbatch" ;;
  generate_eval)
    export MV_RUN_DIR="$1"; export MV_NSAMPLES="${2:-32}"
    sbatch --account="$MV_ACCOUNT_GPU" --qos="$MV_QOS_GPU" --constraint="$MV_GPU_CONSTRAINT" \
           --output="$MV_WORK/logs/geneval_%j.out" --error="$MV_WORK/logs/geneval_%j.out" --export=ALL "$HERE/generate_eval.sbatch" ;;
  *) echo "unknown job '$JOB' (extract|train|generate_eval)" >&2; exit 1 ;;
esac
