#!/bin/zsh
# Sampler ablation, unconstrained, b8 seed 0 : the final batches of traj_test.py for E, the box metrics and the profiles.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
for a in "" "--border0" "--border_noise"; do for c in none all; do
  python traj_test.py ${=a} --center $c > logs/traj${a// /}_$c.log 2>&1
done; done
