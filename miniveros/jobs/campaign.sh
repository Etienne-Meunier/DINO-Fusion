#!/bin/bash
# Submit the whole miniveros campaign as a SLURM dependency chain (all dev QoS, every job < 2 h):
#     extract  ->  train (one job per normalisation mode)  ->  generate_eval (one per training run)
# usage:  jobs/campaign.sh [norm_mode ...]        default: 3-std   (modes are "<k>-std" variants)
# Preconditions checked here: all raw runs present on SCRATCH with one identical file size.
# Each job exits non-zero on failure (set -e), so dependants never start on garbage.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/jz_env.sh" ] || { echo "missing $HERE/jz_env.sh" >&2; exit 1; }
source "$HERE/jz_env.sh"
MODES=("$@"); [ ${#MODES[@]} -gt 0 ] || MODES=(3-std)

N=$(ls "$MV_RAW_DIR"/ck*_eps*.npz 2>/dev/null | wc -l)
NSIZES=$(stat -c %s "$MV_RAW_DIR"/ck*_eps*.npz 2>/dev/null | sort -u | wc -l)
echo "raw runs on SCRATCH: $N files, $NSIZES distinct size(s) (expected ${MV_EXPECTED_RUNS:-100}, 1)"
[ "$N" -eq "${MV_EXPECTED_RUNS:-100}" ] && [ "$NSIZES" -eq 1 ] || { echo "raw data incomplete; aborting" >&2; exit 1; }

export MV_GIT_HASH=$(git -C "$MV_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)
mkdir -p "$MV_WORK/logs" "$MV_WORK/runs" "$MV_WORK/data"
sb() { sbatch --parsable "$@" | cut -d';' -f1; }
GPU=(--account="$MV_ACCOUNT_GPU" --qos="$MV_QOS_GPU" --constraint="$MV_GPU_CONSTRAINT" --export=ALL)

export MV_ARGS=""
J_EXTRACT=$(sb --account="$MV_ACCOUNT_CPU" --qos="$MV_QOS_CPU" --partition="$MV_CPU_PARTITION" \
               --output="$MV_WORK/logs/extract_%j.out" --error="$MV_WORK/logs/extract_%j.out" --export=ALL "$HERE/extract.sbatch")
echo "extract: job $J_EXTRACT  (git $MV_GIT_HASH)"

for MODE in "${MODES[@]}"; do
  TAG=${MODE//-/}                       # 3-std -> 3std
  RUN="$MV_WORK/runs/full_$TAG"
  export MV_ARGS="--preset full --set data_file=$MV_DATA run_dir=$RUN norm_mode=$MODE"
  J_TRAIN=$(sb --dependency=afterok:"$J_EXTRACT" "${GPU[@]}" \
               --output="$MV_WORK/logs/train_${TAG}_%j.out" --error="$MV_WORK/logs/train_${TAG}_%j.out" "$HERE/train.sbatch")
  export MV_RUN_DIR="$RUN" MV_NSAMPLES=32
  J_EVAL=$(sb --dependency=afterok:"$J_TRAIN" "${GPU[@]}" \
              --output="$MV_WORK/logs/geneval_${TAG}_%j.out" --error="$MV_WORK/logs/geneval_${TAG}_%j.out" "$HERE/generate_eval.sbatch")
  echo "$MODE: train job $J_TRAIN -> generate_eval job $J_EVAL   run_dir $RUN"
done
echo "monitor: squeue -u \$USER ; sacct -X -o JobID,JobName,State,Elapsed -j <ids>"
