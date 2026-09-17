#!/bin/bash
#SBATCH --job-name=mv_extract
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=0:59:00
set -eo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
eval "$MV_ENV_SETUP"
cd "$MV_CODE"
echo "job $SLURM_JOB_ID on $(hostname) | git $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $(date)"
echo "raw: $MV_RAW_DIR -> out: $MV_DATA | extra args: $MV_ARGS"
python extract_data.py --raw-dir "$MV_RAW_DIR" --out "$MV_DATA" $MV_ARGS
echo "done $(date)"
