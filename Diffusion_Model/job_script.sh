#!/bin/bash
#SBATCH --job-name=dino_diffusion
#SBATCH --output=dino_diffusion.out
#SBATCH --error=dino_diffusion.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=32
#SBATCH --account=omr@v100
#SBATCH -C v100-32g

## load Pytorch module
module purge
module load python
conda activate MLenv

## launch script on every node
set -x

# code execution
srun accelerate launch train.py
