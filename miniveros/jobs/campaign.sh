#!/bin/bash
# Submit a miniveros campaign as a SLURM dependency chain (all dev QoS, every job < 2 h):
#     [extract]  ->  train (one job per normalisation mode)  ->  generate_eval (one per training run)
# usage:  jobs/campaign.sh [-s name] [-p "key=value ..."] [-n] [-x "extract args"] [norm_mode ...]   default mode: 3-std
#   -s NAME   run-dir prefix: runs/NAME_<mode>          (default: full)
#   -p PAIRS  extra --set pairs for train.py, e.g. "split_mode=rows split_rows=0.126,0.2,0.3175"
#   -n        do not (re)extract: the chain starts at training with the existing data file
#   -x ARGS   extra extraction arguments
# One data file serves every split (normalisation statistics on all runs); the split is a training-config choice:
#     jobs/campaign.sh    -s fs_scattered -p "split_mode=interior_random"
#     jobs/campaign.sh -n -s fs_band3     -p "split_mode=rows split_rows=0.126,0.2,0.3175"
#     jobs/campaign.sh -n -s fs_top       -p "split_mode=row_ck_max"
#     jobs/campaign.sh -n -s fs_block     -p "split_mode=block"                     (7 x 7 centre block held out)
# Each job exits non-zero on failure (set -e), so dependants never start on garbage.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/env.sh" ] || { echo "missing $HERE/env.sh - copy env.example.sh and fill it in" >&2; exit 1; }
source "$HERE/env.sh"
SPLIT=""; XARGS=""; PAIRS=""; NOEXTRACT=0
while getopts "s:x:p:n" opt; do case $opt in s) SPLIT=$OPTARG ;; x) XARGS=$OPTARG ;; p) PAIRS=$OPTARG ;; n) NOEXTRACT=1 ;; *) exit 1 ;; esac; done
shift $((OPTIND - 1))
MODES=("$@"); [ ${#MODES[@]} -gt 0 ] || MODES=(3-std)
PREFIX="${SPLIT:-full}_"
echo "runs: ${PREFIX}<mode>  data: $MV_DATA  train settings: ${PAIRS:-defaults}  extraction: $([ $NOEXTRACT = 1 ] && echo skipped || echo yes)"

N=$(ls "$MV_RAW_DIR"/ck*_eps*.npz 2>/dev/null | wc -l)
NSIZES=$(stat -c %s "$MV_RAW_DIR"/ck*_eps*.npz 2>/dev/null | sort -u | wc -l)
if [ $NOEXTRACT = 0 ]; then
  echo "raw runs on SCRATCH: $N files, $NSIZES distinct size(s) (expected ${MV_EXPECTED_RUNS:-100}, 1)"
  [ "$N" -eq "${MV_EXPECTED_RUNS:-100}" ] && [ "$NSIZES" -eq 1 ] || { echo "raw data incomplete; aborting" >&2; exit 1; }
fi

export MV_GIT_HASH=$(git -C "$MV_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)
mkdir -p "$MV_WORK/logs" "$MV_WORK/runs" "$MV_WORK/data"
sb() { sbatch --parsable "$@" | cut -d';' -f1; }
GPU=(--account="$MV_ACCOUNT_GPU" --qos="$MV_QOS_GPU" --constraint="$MV_GPU_CONSTRAINT" --export=ALL)

if [ $NOEXTRACT = 1 ]; then
  [ -f "$MV_DATA" ] || { echo "no data file $MV_DATA and extraction skipped" >&2; exit 1; }
  DEP=()
else
  export MV_ARGS="$XARGS"
  J_EXTRACT=$(sb --account="$MV_ACCOUNT_CPU" --qos="$MV_QOS_CPU" --partition="$MV_CPU_PARTITION" \
                 --output="$MV_WORK/logs/extract_%j.out" --error="$MV_WORK/logs/extract_%j.out" --export=ALL "$HERE/extract.sbatch")
  echo "extract: job $J_EXTRACT  (git $MV_GIT_HASH)"; DEP=(--dependency=afterok:"$J_EXTRACT")
fi

for MODE in "${MODES[@]}"; do
  TAG=${MODE//-/}                       # 3-std -> 3std
  RUN="$MV_WORK/runs/${PREFIX}$TAG"
  export MV_ARGS="--preset full --set data_file=$MV_DATA run_dir=$RUN norm_mode=$MODE $PAIRS"
  J_TRAIN=$(sb "${DEP[@]}" "${GPU[@]}" \
               --output="$MV_WORK/logs/train_${PREFIX}${TAG}_%j.out" --error="$MV_WORK/logs/train_${PREFIX}${TAG}_%j.out" "$HERE/train.sbatch")
  export MV_RUN_DIR="$RUN" MV_NSAMPLES=32
  J_EVAL=$(sb --dependency=afterok:"$J_TRAIN" "${GPU[@]}" \
              --output="$MV_WORK/logs/geneval_${PREFIX}${TAG}_%j.out" --error="$MV_WORK/logs/geneval_${PREFIX}${TAG}_%j.out" "$HERE/generate_eval.sbatch")
  echo "$MODE: train job $J_TRAIN -> generate_eval job $J_EVAL   run_dir $RUN"
done
echo "monitor: squeue -u \$USER ; sacct -X -o JobID,JobName,State,Elapsed -j <ids>"
