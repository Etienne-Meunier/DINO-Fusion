#!/bin/bash
#SBATCH --job-name=dino_diffusion
#SBATCH --output=dino_diffusion.out
#SBATCH --error=dino_diffusion.out
#SBATCH --gres=gpu:1  ######4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=99:59:00              # max20h on qos_gpu_t3, max 100h on qos_gpu-t4
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=10           #24 si partition big
#SBATCH --account=omr@v100
#SBATCH --constraint v100-32g #####gpu_p2s for more RAM CPU, --partition=gpu_p13 

## load Pytorch module
module purge
module load python
conda activate MLenv
export WANDB_MODE=offline          # run wandb sync offline-run-XXXXXXX/ when you are on the head nodes with internet
export OCEANDATA=/lustre/fswork/projects/rech/omr/ufk69pe/

## launch script on every node
set -x

# code execution
srun accelerate launch train.py #--num_processes 4