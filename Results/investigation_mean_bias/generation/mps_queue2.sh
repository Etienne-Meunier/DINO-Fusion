#!/bin/zsh
export OCEANDATA=/Volumes/LoCe/oceandata/ MPLBACKEND=Agg
cd /Users/emeunier/Desktop/Projets/DINO-Fusion/Results/investigation_mean_bias/generation
while pgrep -f mps_queue.sh > /dev/null; do sleep 10; done
for s in 2 3; do
python gen_test.py --device mps --gen_device mps --seed $s --name mps_s${s}_b16 > logs/mps_s${s}_b16.log 2>&1
python gen_test.py --device mps --gen_device cpu --seed $s --name mps_cpugen_s${s}_b16 > logs/mps_cpugen_s${s}_b16.log 2>&1
done
