#!/bin/bash
#SBATCH --job-name=dino_diffusion
#SBATCH --output=dino_diffusion.out
#SBATCH --error=dino_diffusion.out
#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=40:00:00             

## load Pytorch module
module purge
#module load python
conda activate MLenv
export WANDB_MODE=offline          # run wandb sync offline-run-XXXXXXX/ when you are on the head nodes with internet

## launch script on every node
set -x

# code execution
srun accelerate launch train.py --num_processes 2
