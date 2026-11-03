#!/bin/zsh
# Eighth round : strength sweep of the surface data loss, as fig4 of the paper. C7 eta on the grid of the repo C7
# sweep (10, 30, 50-100, 300, and 200), with the surface constraint fixed at sqrt eta 1e-4, eps 9. C7 alone at eta 200
# to complete the C7 curve.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
for e in 10 60 70 80 90 ; do
  r surf_c7$(printf %03d $e)_e1e-4 --variant c7_surf --mode raw sqrt --lbd0 1 0 --eta $e 1e-4 --eps 8.6e-8 9
done
r c7_c7200 --variant c7 --eta 200 --eps 8.6e-8
