#!/bin/zsh
# Twelfth round : the surface mean of a generation is set by the spatial mean of the noise added in the first steps,
# and BorderZero at high noise cools it by ~2.5 C. Centred noise + noisy land cells (BorderZero at the end only) gives
# T0 20.11 +/- 0.12 unconstrained (traj_test). Full runs : unconstrained and C7 sweep with that sampler.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
r centred_none --variant none --eta 0 --eps 0 --centred
for e in 10 30 50 100; do r centred_c7$(printf %03d $e) --variant c7 --eta $e --eps 8.6e-8 --centred; done
python evaluate.py 'centred*' > logs/evaluate12.log 2>&1
