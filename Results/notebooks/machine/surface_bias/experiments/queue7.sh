#!/bin/zsh
# Seventh round : tolerance of the anchor (C7 without the top pair + surface anchor, sqrt 1e-4). eps in units of
# (0.1 C, 0.01 psu)^2 : 16, 100, 400 allow the surface to drift by about 0.4, 1, 2 C from the unconstrained run.
# Trade-off between "no worse than none" and the density error E.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
for e in 16 100 400 ; do
  r skipanchor_c7100_e1e-4_eps$e --variant c7skip_anchor --mode raw sqrt --lbd0 1 0 --eta 100 1e-4 --eps 8.6e-8 $e
done
python evaluate.py > logs/evaluate7.log 2>&1
