#!/bin/bash
#SBATCH --job-name=dino_diffusion
#SBATCH --output=dino_diffusion.out
#SBATCH --error=dino_diffusion.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # max20h on qos_gpu_t3
#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=24           #40
#SBATCH --account=omr@v100
#SBATCH --partition=gpu_p2s          #-C v100-32g

## load Pytorch module
module purge
module load python
conda activate MLenv
export WANDB_MODE=offline          # run wandb sync offline-run-XXXXXXX/ when you are on the head nodes with internet

## launch script on every node
set -x

# code execution
srun accelerate launch train.py --num_processes 4
