#!/bin/zsh
# Ninth round : batch 16, seed 0, of the leading options, for the score table next to the paper optimal runs (b16, seed 0).
#   surface data term : C7 eta 100 (best E), 30 and 10 (paper strengths) + SurfaceMean sqrt 1e-4, eps 9
#   profile hinge + C7 surface stop-grad (eta 100, 100)
#   C7 without top pair + anchor to the unconstrained run (sqrt 1e-4, eps 1)
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name --batch 16 "$@" > logs/$name.log 2>&1 }
r b16_surf_c7100 --variant c7_surf --mode raw sqrt --lbd0 1 0 --eta 100 1e-4 --eps 8.6e-8 9
r b16_proftop    --variant profile_topfree --eta 100 100 --eps 0 8.6e-8 --lbd0 1 1
r b16_skipanchor --variant c7skip_anchor --mode raw sqrt --lbd0 1 0 --eta 100 1e-4 --eps 8.6e-8 1
r b16_surf_c7030 --variant c7_surf --mode raw sqrt --lbd0 1 0 --eta 30 1e-4 --eps 8.6e-8 9
r b16_surf_c7010 --variant c7_surf --mode raw sqrt --lbd0 1 0 --eta 10 1e-4 --eps 8.6e-8 9
