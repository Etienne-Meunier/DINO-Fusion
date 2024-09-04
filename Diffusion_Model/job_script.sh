#!/bin/bash
#SBATCH --job-name=dino_diffusion
#SBATCH --output=/home/meunier/logs/%x_%j.out
#SBATCH --error=/home/meunier/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --partition=hard
#SBATCH --gpus-per-node=1
#SBATCH --time=40:00:00             

## load Pytorch module
module purge
#module load python
conda init
conda activate sc_dl
export WANDB_MODE=online          # run wandb sync offline-run-XXXXXXX/ when you are on the head nodes with internet

## launch script on every node
set -x

# code execution
#srun accelerate launch train.py --num_processes 1 #srun python tr  #
srun accelerate launch train.py 
