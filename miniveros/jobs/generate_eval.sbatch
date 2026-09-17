#!/bin/bash
#SBATCH --job-name=mv_geneval
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=1:59:00
set -eo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
eval "$MV_ENV_SETUP"
cd "$MV_CODE"
echo "job $SLURM_JOB_ID on $(hostname) | git ${MV_GIT_HASH:-?} | $(date)"
echo "run_dir: $MV_RUN_DIR | n_samples: $MV_NSAMPLES"
python generate.py --run-dir "$MV_RUN_DIR" --holdout --n-samples "$MV_NSAMPLES"
python generate.py --run-dir "$MV_RUN_DIR" --grid    --n-samples 4
HOLD=$(ls -t "$MV_RUN_DIR"/samples/holdout_*.npz | head -1); GRID=$(ls -t "$MV_RUN_DIR"/samples/grid_*.npz | head -1)
python evaluate.py --data-file "$MV_DATA" --samples "$HOLD" --grid-samples "$GRID" --out-dir "$MV_RUN_DIR/eval"
echo "done $(date)"
