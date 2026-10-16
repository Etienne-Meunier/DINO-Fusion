# Density inversion: work plan

Goal: one (max two) constraint/loss/projection that reduces the eval metric E = % ocean volume with rho[k+1] - rho[k] < 0 (`compute_density_error`, metrics.py ~l.337) better than `negative_grad_density` (NGD), with evidence. Post-hoc only, local CPU/MPS. Any new diffusion sampling needs user approval first.

Runs (R = `$OCEANDATA/models/dino-fusion/tav0h83b/inference/test_new_sampler_clip`, OCEANDATA=/Volumes/LoCe/oceandata/), all denormalised with the same helpers as `preliminaries/data_reference.py`, `local_metrics.py`:
- DATA: reference data set (same samples/count as in FINDINGS_short).
- NONE: `R/none`
- NGD: `R/negative_grad_density_sqrt_eta0.3_eps0`
- X64: `R/combo_iso5e-6_zm0.1_eps1.66e-1` (guided, no projection) and X64J: `R/combo_iso5e-6_zm0.1_eps1.66e-1_joint0.01` (+ joint)

Notation: g = rho[k+1] - rho[k] on wet pairs (both cells tmask = 1), w = volume weight as in the metric. E_bin = volume-% of cells with g in bin.

## Step 0 - Reproduce the metric (sanity, 5 min)
- Recompute E on the 5 sets with our own script and check it matches `compute_density_error` (and FINDINGS_short numbers: DATA 0.63 %, X64J ~1.5 %) to 0.01 %.
- Record which dtype the eval actually uses (xarray copy of `e3t_0` -> likely float64 density from float32 T,S) and which dtype the torch observer / joint projection uses (float32).
- Acceptance: match within 0.01 %. Otherwise stop and fix the reproduction before anything else.

## Step 1 - Binned diagnostic, no sampling (main step)
For each of DATA, NONE, NGD, X64, X64J, compute E split by |g| bins for g < 0: [0, 1.2e-4), [1.2e-4, 1e-3), [1e-3, 1e-2), >= 1e-2, plus the count of exact ties g == 0.
- Do it twice: density in float32 (torch EOS, as the observer) and float64 (numpy EOS, from the same T,S).
- Same table split by depth band: levels 0-9 (mixed layer / thermocline), 10-24, 25-36.
- Also report, for float64: fraction of X64J inversions located where float32 rho has a tie (g32 == 0) -> direct test of the "isotonic tie becomes negative at eval" mechanism.
- Output: `step1_bins.csv` + one small figure (stacked bars per set, two panels f32/f64).

## Gate - H1 vs H2 (decide on X64J, float64 = eval dtype)
Let s = share of E(X64J) in bins |g| < 1e-3 (first two bins).
- H1 (ties/noise) if s >= 0.7, or if E(X64J) changes by > 30 % between f32 and f64. -> Branch A.
- H2 (real inversions) if s <= 0.3, i.e. most of E is in |g| >= 1e-3. -> Branch B.
- In between: run Branch A first (cheap, post-hoc), then report how much of E remains and whether B is needed.
- Also note where NGD's 18.7 % lives in the bins: if mostly < 1e-3, NGD's failure is "hinge at 0 has no margin" (supports A/B with margin); if large, it is a guidance-strength failure.

## Branch A - Margin isotonic projection P_m (post-hoc, no sampling)
P_m(rho) = Iso(rho - m*k) + m*k (k = level index), i.e. projected profiles are strictly increasing with step >= m. Implement as a variant of `dist_isoton` target inside `joint_projection.py` logic (copy into this folder, do not edit the original), density computed in float64 (or float32 with m >> ulp).
- Choose m from DATA: q = 5 %, 10 %, 25 % quantile of g on deep stable data cells (levels >= 10, g > 0); cap at 1e-3. Report those values. Test m in {0, 1.2e-4 (ulp), q10, 1e-3}; m = 0 must reproduce X64J.
- Apply to X64 (same LBFGS budget, c = 0.01, then BorderZero) and, as control, to NONE.
- Metrics (same as FINDINGS_short table): E (f64), E > 0.01, distance to the unconstrained run (NONE, same seed), level-mean variance, anomaly variance, roughness, profile_rmse, mean_density; plus the Step-1 bins after projection.
- Acceptance: E(X64 + P_m) <= 1.0 % (target: <= 2 x DATA = 1.3 %) AND E > 0.01 not worse than X64J (<= 0.005 %) AND distance, roughness, level-mean variance within +10 % of X64J. Best m = smallest m meeting this.
- If E stays > 1.3 % with all m: the residual is not ties -> go to Branch B.
- Optional check (cheap): P_m on NONE alone -> is projection without guidance enough? Reported, not the proposal unless it meets the same criterion.

## Branch B - Volume-weighted soft hinge with margin (guidance loss)
L = sum_cells w * tau * softplus((-g + m) / tau) / sum w (penalises g < m smoothly; m from Branch A, tau ~ m/2).
- Post-hoc first (no sampling): use L as the objective of the final LBFGS (in place of / added to dist_isoton) on X64 and NONE; same metrics and acceptance as Branch A. Compare gradient sparsity vs NGD on the same x0 (fraction of cells with nonzero gradient, ratio of E reduction per unit distance).
- Diffusion-guided version (sqrt mode, 2 eta values, 100 steps, 2 seeds) only if post-hoc L beats P_m AND the gate pointed to H2: REQUIRES USER APPROVAL before running (new sampling).

## Step 3 - Report back to supervisor
- `RESULTS.md` (short): Step-1 table (5 sets x 4 bins x f32/f64), gate decision with numbers, Branch table (rows: X64J, X64+P_m for each m, NONE+P_m, [hinge]) with all metrics above, one figure of E vs distance.
- State explicitly: which hypothesis the bins support, chosen m and why, and remaining E split by bin.
- Hand to reviewer before sending; include reviewer comments verbatim in the message.

Constraints: all code and outputs in this folder; no edits to `Diffusion_Model/` or `preliminaries/`; no new sampling without approval; always BorderZero after any projection.
