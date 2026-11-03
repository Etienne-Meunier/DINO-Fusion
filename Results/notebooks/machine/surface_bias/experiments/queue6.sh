#!/bin/zsh
# Sixth round : anchor variants. queue5 showed the surface anchor holds level 0 at none but C7 then densifies the
# levels below (61-299 m +0.49) and E stays at 2.4-2.6. Tested here : C7 without the top pair (no fight with the
# anchor), anchor on T only, anchor on the top 12 / 36 level means, profile + C7 stop-grad + anchor ;
# and the stability limit of the surface data constraint (sqrt 3e-4).
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
A2() { r $1 --mode raw sqrt --lbd0 1 0 "${@:2}" }

A2 skipanchor_c7100_e1e-4      --variant c7skip_anchor --eta 100 1e-4 --eps 8.6e-8 1
r  skip_c7100                  --variant c7skip --eta 100 --eps 8.6e-8
A2 anchorT_c7100_e1e-4         --variant c7_anchor --dS 1000 --eta 100 1e-4 --eps 8.6e-8 1
A2 anchorprof12_c7100_e1e-4    --variant c7_anchor_prof --K 12 --eta 100 1e-4 --eps 8.6e-8 1
r  proftopanchor_e1e-4         --variant proftop_anchor --mode raw raw sqrt --lbd0 1 1 0 --eta 100 100 1e-4 --eps 0 8.6e-8 1
A2 skipanchor_c7100_e1e-5      --variant c7skip_anchor --eta 100 1e-5 --eps 8.6e-8 1
A2 anchorprof12_c7100_e1e-4_eps4 --variant c7_anchor_prof --K 12 --eta 100 1e-4 --eps 8.6e-8 4
A2 anchorprof36_c7100_e1e-4    --variant c7_anchor_prof --K 36 --eta 100 1e-4 --eps 8.6e-8 1
A2 surf_c7100_e3e-4            --variant c7_surf --eta 100 3e-4 --eps 8.6e-8 9
for s in 1 2 ; do
  A2 skipanchor_c7100_e1e-4_s$s --variant c7skip_anchor --eta 100 1e-4 --eps 8.6e-8 1 --seed $s
done
python evaluate.py > logs/evaluate6.log 2>&1
