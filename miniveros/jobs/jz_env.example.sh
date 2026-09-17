# Copy this file to jz_env.sh (gitignored) ON THE CLUSTER and fill in the values.
# submit.sh sources it; nothing site-specific (account, paths, login) lives in tracked files.
export MV_WORK=/lustre/fswork/projects/rech/<proj>/<login>/miniveros      # permanent: code, data, runs, logs
export MV_SCRATCH=/lustre/fsn1/projects/rech/<proj>/<login>/miniveros     # purgeable: raw runs
export MV_RAW_DIR=$MV_SCRATCH/full_state/mnk965ig
export MV_CODE=$MV_WORK/code/miniveros                                     # the package inside the git clone
export MV_DATA=$MV_WORK/data/veros_acc_TS.npz
export MV_ACCOUNT_CPU=<proj>@cpu
export MV_ACCOUNT_GPU=<proj>@a100
export MV_QOS_CPU=qos_cpu-dev
export MV_QOS_GPU=qos_gpu_a100-dev
export MV_GPU_CONSTRAINT=a100
export MV_CPU_PARTITION=cpu_p1
# Sourcing the module init makes this work from any shell (a plain `ssh host cmd` has no `module` function);
# `conda shell.bash hook` defines conda without relying on the user's rc files.
export MV_ENV_SETUP='source /etc/profile.d/modules.sh 2>/dev/null; source /etc/profile.d/z_modules.sh 2>/dev/null; module purge; module load python; eval "$(conda shell.bash hook)"; conda activate MLenv'
