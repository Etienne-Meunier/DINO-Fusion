#!/bin/zsh
# Fifth round : refinement of the two retained losses, seed 0 batch 8 unless stated, C7 always lbd0 1, eps 8.6e-8.
#   S (surface data target) : C7 + SurfaceMean in sqrt mode, eps 9, eta 1e-4 by default
#   A (anchor, no surface data) : C7 + SurfaceAnchor in sqrt mode, level-0 mean T and S kept at those of the
#     unconstrained trajectory of the same seed sampled alongside, units 0.1 C / 0.01 psu, eps 1, eta 1e-4 by default
# Sweeps interleaved so that partial results are already readable.
export OCEANDATA=/Volumes/LoCe/oceandata/
cd "${0:A:h}"
r() { local name=$1; shift; [[ -f runs/$name/generated.npy ]] && return; python run.py --name $name "$@" > logs/$name.log 2>&1 }
S() { r $1 --variant c7_surf --mode raw sqrt --lbd0 1 0 "${@:2}" }
A() { r $1 --variant c7_anchor --mode raw sqrt --lbd0 1 0 "${@:2}" }
R() { r $1 --variant c7_anchor_rho --mode raw sqrt --lbd0 1 0 "${@:2}" }

# anchor strength and surface data constraint vs C7 eta
A anchor_c7100_e1e-4      --eta 100 1e-4 --eps 8.6e-8 1
A anchor_c7100_e1e-3      --eta 100 1e-3 --eps 8.6e-8 1
S surf_c7300_e1e-4        --eta 300 1e-4 --eps 8.6e-8 9
A anchor_c7100_e1e-5      --eta 100 1e-5 --eps 8.6e-8 1
S surf_c7200_e1e-4        --eta 200 1e-4 --eps 8.6e-8 9
S surf_c7050_e1e-4        --eta 50 1e-4 --eps 8.6e-8 9
S surf_c7030_e1e-4        --eta 30 1e-4 --eps 8.6e-8 9
# anchor tolerance, one-sided density anchor, fixed final reference
A anchor_c7100_e1e-4_eps0.25 --eta 100 1e-4 --eps 8.6e-8 0.25
A anchor_c7100_e1e-4_eps4    --eta 100 1e-4 --eps 8.6e-8 4
R anchorrho_c7100_e1e-4_eps1    --eta 100 1e-4 --eps 8.6e-8 1
R anchorrho_c7100_e1e-4_eps0.01 --eta 100 1e-4 --eps 8.6e-8 0.01
A anchorfinal_c7100_e1e-4 --anchor final --eta 100 1e-4 --eps 8.6e-8 1
# surface data tolerance and strength
S surf_c7100_e1e-4_eps1   --eta 100 1e-4 --eps 8.6e-8 1
S surf_c7100_e1e-4_eps4   --eta 100 1e-4 --eps 8.6e-8 4
S surf_c7100_e1e-4_eps16  --eta 100 1e-4 --eps 8.6e-8 16
S surf_c7100_e1e-3        --eta 100 1e-3 --eps 8.6e-8 9
S surf_c7100_e1e-6        --eta 100 1e-6 --eps 8.6e-8 9
# anchor vs C7 eta
A anchor_c7300_e1e-4      --eta 300 1e-4 --eps 8.6e-8 1
A anchor_c7030_e1e-4      --eta 30 1e-4 --eps 8.6e-8 1
A anchor_c7200_e1e-4      --eta 200 1e-4 --eps 8.6e-8 1
# seeds 1, 2 of the defaults, paired with their own unconstrained run
for s in 1 2 ; do
  r none_s$s --variant none --eta 0 --eps 0 --seed $s
  r c7_c7100_s$s --variant c7 --eta 100 --eps 8.6e-8 --seed $s
  S surf_c7100_e1e-4_s$s --eta 100 1e-4 --eps 8.6e-8 9 --seed $s
  A anchor_c7100_e1e-4_s$s --eta 100 1e-4 --eps 8.6e-8 1 --seed $s
done
python evaluate.py > logs/evaluate5.log 2>&1
