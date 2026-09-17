#!/bin/bash
#SBATCH --job-name=mv_train
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=1:59:00
set -eo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
eval "$MV_ENV_SETUP"
cd "$MV_CODE"
echo "job $SLURM_JOB_ID on $(hostname) | git ${MV_GIT_HASH:-?} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "args: $MV_ARGS"
python train.py $MV_ARGS
echo "done $(date)"
