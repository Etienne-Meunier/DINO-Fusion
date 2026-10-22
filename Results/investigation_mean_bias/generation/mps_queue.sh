#!/bin/zsh
export OCEANDATA=/Volumes/LoCe/oceandata/ MPLBACKEND=Agg
cd /Users/emeunier/Desktop/Projets/DINO-Fusion/Results/investigation_mean_bias/generation
python gen_test.py --device mps --gen_device mps --seed 0 --name mps_s0_b16 > logs/mps_s0_b16.log 2>&1
python gen_test.py --device mps --gen_device cpu --seed 0 --name mps_cpugen_s0_b16 > logs/mps_cpugen_s0_b16.log 2>&1
python gen_test.py --device mps --gen_device mps --seed 1 --name mps_s1_b16 > logs/mps_s1_b16.log 2>&1
python gen_test.py --device mps --gen_device cpu --seed 1 --name mps_cpugen_s1_b16 > logs/mps_cpugen_s1_b16.log 2>&1
