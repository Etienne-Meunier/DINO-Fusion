#!/bin/bash
#SBATCH --job-name=mv_geneval
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=1:59:00
set -eo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
eval "$MV_ENV_SETUP"
cd "$MV_CODE"
echo "job $SLURM_JOB_ID on $(hostname) | git ${MV_GIT_HASH:-?} | $(date)"
echo "run_dir: $MV_RUN_DIR | n_samples: $MV_NSAMPLES | tag: ${MV_TAG:-none} | set: ${MV_GEN_SET:-none}"
# MV_TAG names a sampling variant (samples/*_<tag>.npz, eval_<tag>/); MV_GEN_SET holds --set pairs for generate.py
python generate.py --run-dir "$MV_RUN_DIR" --holdout --n-samples "$MV_NSAMPLES" ${MV_TAG:+--tag "$MV_TAG"} ${MV_GEN_SET:+--set $MV_GEN_SET}
python generate.py --run-dir "$MV_RUN_DIR" --grid    --n-samples "$MV_NSAMPLES" ${MV_TAG:+--tag "$MV_TAG"} ${MV_GEN_SET:+--set $MV_GEN_SET}
HOLD=$(ls -t "$MV_RUN_DIR"/samples/holdout_*.npz | head -1); GRID=$(ls -t "$MV_RUN_DIR"/samples/grid_*.npz | head -1)
python evaluate.py --data-file "$MV_DATA" --samples "$HOLD" --grid-samples "$GRID" --out-dir "$MV_RUN_DIR/eval${MV_TAG:+_$MV_TAG}" --config "$MV_RUN_DIR/config.json"
echo "done $(date)"
