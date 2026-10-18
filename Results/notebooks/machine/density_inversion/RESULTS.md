# Density inversion: results (Steps 0-1)

Scripts in this folder: `step1_bins.py` (Step 0 and Step 1 on DATA, NONE, NGD, X64, X64J), `step1_realism.py`, `step1_all_runs.py` (eval E + realism on every run of `test_new_sampler_clip`). Outputs: `step1_summary.csv`, `step1_bins.csv`, `step1_depth.csv`, `step1_bins.png`, `step1_realism.csv`, `step1_all_runs.csv`. DATA = 64 random training states (seed 0); generated sets are batches of 8.

## Step 0: reproduction of the eval metric

Our E (float64 EOS, volume of the lower cell, file levels from `Observer.unnormalized`) matches `compute_density_error` to all printed digits on the 5 sets. The eval is float64 (xarray copy of `e3t_0`); the torch observer and the joint projection compute rho in float32 (rho ~ 1027, ulp ~ 1.2e-4).

Important: the "inversions" column of FINDINGS_short (`pd_explore.inversion_fraction`) is NOT the eval metric. It is an unweighted fraction of vertical pairs on the 18 strided levels. The two disagree strongly:

| set | eval E (% volume) | FINDINGS pair fraction (%) | pair fraction > 0.01 (%) |
|---|---|---|---|
| DATA | 0.36 | 0.63 | 0 |
| NONE | 9.28 | 26.3 | 24.5 |
| NGD (sqrt eta 0.3) | 3.57 | 12.8 | 10.8 |
| X64 (iso + zm, guided) | 0.27 | 2.00 | 0.60 |
| X64J (+ joint 0.01) | 0.23 | 1.55 | 0.002 |

## Step 1: E split by inversion magnitude |g| (float64, % of ocean volume)

| set | E | < 1.2e-4 | 1.2e-4 to 1e-3 | 1e-3 to 1e-2 | >= 1e-2 | levels 0-9 | 10-24 | 25-35 |
|---|---|---|---|---|---|---|---|---|
| DATA | 0.360 | 0.360 | 0.000 | 0 | 0 | 0.349 | 0.010 | 0.000 |
| NONE | 9.28 | 0.078 | 0.559 | 3.81 | 4.83 | 1.74 | 2.21 | 5.34 |
| NGD | 3.57 | 0.047 | 0.310 | 1.42 | 1.79 | 0.55 | 1.67 | 1.35 |
| X64 | 0.271 | 0.012 | 0.068 | 0.172 | 0.019 | 0.047 | 0.138 | 0.086 |
| X64J | 0.232 | 0.062 | 0.139 | 0.032 | 0.000 | 0.115 | 0.032 | 0.085 |

- DATA: all of its E is below 1.2e-4 and in levels 0-9 (well-mixed columns, sign set by rounding). In float32 the data E falls to 0.07 % and 0.76 % of the volume is exact ties.
- Float32 vs float64 changes E by at most 12 % (X64J 0.207 vs 0.232); 11 % of X64J f64 inversions sit on f32 ties. Precision is a minor effect.
- Gate (PLAN.md) on X64J: s = share below 1e-3 = 0.86, formally H1. But X64J (0.23 %) is already below DATA (0.36 %), so the Branch A target (E <= 1 %) is already met without any new projection; the margin would only push below the data level.
- NGD: 90 % of its E is in |g| >= 1e-3 (1.42 + 1.79 of 3.57). Its failure is not a margin/tie problem: large inversions remain.

## All runs: eval E vs realism (extract of `step1_all_runs.csv`, 1000 steps)

| run | E | E >= 1e-3 | dist to NONE | level-mean var | anomaly var | roughness |
|---|---|---|---|---|---|---|
| DATA (FINDINGS ref.) | 0.36 | 0 | 0.044 (two states) | 1.15e-4 | 8.9e-4 | 1.2e-3 |
| iso + zm, zm eps 1.33 | 0.228 | 0.159 | 0.106 | 9.4e-5 | 1.87e-3 | 3.2e-3 |
| X64J | 0.232 | 0.032 | 0.105 | 2.6e-5 | 1.92e-3 | 3.3e-3 |
| zero_mean alone (eta 1) | 1.11 | 1.02 | 0.099 | 9.9e-6 | 1.99e-3 | 3.4e-3 |
| dist_isoton alone (sqrt 5e-6) | 1.71 | 1.36 | 0.143 | 1.9e-3 | 1.93e-3 | 2.9e-3 |
| NGD (sqrt 0.3) | 3.57 | 3.21 | 0.081 | 2.8e-3 | 2.0e-3 | 3.0e-3 |
| NGD (sqrt 0.1) | 4.02 | 3.59 | 0.059 | 2.7e-3 | 2.1e-3 | 3.1e-3 |
| NONE | 9.28 | 8.64 | 0 | 3.9e-3 | 2.2e-3 | 3.1e-3 |

- `zero_mean` alone, which has no density term, reduces the eval E more (1.11 %) than every density loss used alone (NGD 3.6-4.0 %, dist_isoton 1.7-3.5 %, mean_density 2.4-5.3 %) at a similar or smaller distance.
- The large inversions of NONE sit mostly in the deep levels (25-35: 5.3 of 9.3 %) and go with a level-mean variance 34x the data's. Most of the eval E is a profile (horizontal-mean) error, not a cell-scale one, so a cell-wise hinge such as NGD spends its budget badly.
- The `iso + zm` runs reach E below the data level; with zm eps 1.33 the level-mean variance (9.4e-5) is also close to the data (1.15e-4). The remaining realism gaps are the anomaly variance (2x the data) and the roughness (2.7x), which no density loss addresses.

Branch A and Branch B were not run: their acceptance target is already met by existing runs on the eval metric.

## Step 2: profile / anomaly split and seed spread (after review round 1)

`step2_swap.py`. Each state = level-mean profile of T and S (horizontal mean over ocean cells) + horizontal anomalies. E (float64, eval weighting) is recomputed on hybrids. "data profile" = mean over the 64 DATA states of their profiles.

| set | E | std over samples | data profile + own anomalies | own profile + data anomalies | own profile alone |
|---|---|---|---|---|---|
| NONE | 9.28 | 4.82 | 1.29 | 9.69 | 7.56 |
| NGD (sqrt 0.3) | 3.57 | 1.51 | 1.33 | 3.76 | 0.81 |
| ZM (eta 1) | 1.11 | 0.04 | 1.06 | 1.03 | 0.21 |
| ISO (sqrt 5e-6) | 1.71 | 0.59 | 1.21 | 2.04 | 0 |
| iso + zm, zm eps 1.33 | 0.228 | 0.032 | 1.20 | 0.243 | 0 |
| X64J | 0.232 | 0.033 | 0.68 | 0.65 | 0 |
| DATA | 0.36 | 0.05 | 0.69 | - | 0 |

- Placing the NONE anomalies on the data profile drops E from 9.28 to 1.29 %, and the NONE profile alone already has 7.56 %. 86 % of the unconstrained E comes from the level-mean profile, above the 70 % threshold the reviewer set.
- NGD removes most of the inversions of the profile itself (0.81 %), but its E stays at 3.57 %: the profile is left close to neutral, and the horizontal anomalies flip it. A cell-wise hinge at g = 0 removes inversions without restoring the stratification.
- Generated anomalies on the data profile give 1.2-1.3 % for every guided run except X64J (0.68 %), against 0.69 % for the data's own anomalies. The anomaly part is about 2x the data's, consistent with the 2.7x roughness, and no density loss changes it.
- iso + zm eps 1.33 reaches E 0.23 % with anomalies that would give 1.2 % on the data profile: its profile is more stratified than the data mean, which hides the excess anomaly inversions.
- X64J and iso + zm eps 1.33 cannot be told apart on E (0.232 +/- 0.033 vs 0.228 +/- 0.032, 8 samples each).

## Reviewer comments (round 2, verbatim summary)

1. Reproduction correct (post_processing uses the same `__uncall__`, xarray diff labels the lower cell), but Step 0 checks only the weighting; DATA also goes through stride + interpolation, so 0.36 % is "data as representable by the model".
2. Batches of 8, one seed: iso+zm variants are within noise; the claim "ZM beats every density loss at similar or smaller distance" is wrong for NGD (NGD at distance 0.08 / 0.06 vs ZM 0.099; NGD never pushed to about 0.1 at 1000 steps).
3. Skipping Branch A/B justified for the stated target.
4. "NGD fails because E is a profile error" not shown by a correlation; alternatives: NGD is unweighted (deep, thick cells under-weighted) and under-enforced (1.79 % at |g| >= 1e-2). Suggested the profile/anomaly swap (done above).
5. X64J over-flattens the level means (2.6e-5, 4x below data); iso + zm eps 1.33 without joint (9.4e-5) is the better and simpler proposal.
Proposed: (a) iso + zm with zm eps about 8x q0.5 (the 1.33 run), no joint projection, confirmed on 3 seeds; (b) one guided experiment, needing user approval: volume-weighted NGD, sqrt mode, eta tuned to distance about 0.10, alone and with zm, 2 seeds.

## Step 3: monotonicity-only losses (PLAN2), frontier, transfer, gates, robustness

Candidates (all on g = rho[k+1] - rho[k] from `observer.density`, w = lower-cell volume normalised to sum 1 over wet pairs, code in `frontier.py` `DensityLoss`): C0 = NGD of `losses.py`, C1 = sum w relu(-g)^2, C2 = softplus linear hinge (dropped: hidden margin, active on 27.5 % of data cells), C3 = sigmoid indicator (stalls), C4 = squared dist_isoton, C5 = hinge on the sample's own mean profile, C6 = non-squared distance to the isotonic cone, C7 = one-sided Huber hinge sum w h_tau(g) with h_tau(g) = 0 for g >= 0, g^2/(2 tau) on (-tau, 0), -g - tau/2 below -tau, tau = 1e-4.

Post-hoc frontier (LBFGS on mean_ocean((x - x0)^2) + lam L(x)/L(x0), 300 iterations, then BorderZero, 8 unconstrained states of `test_new_sampler_clip/none`, E 9.28): C7 E 4.79 at d 0.015 and E 2.61 at d 0.021; C1 E 3.55 at d 0.025; C0 E 1.60 at d 0.027 (1.26 in a second cold start under other run conditions, so C0 is not converged at 300 iterations and its frontier points are unreliable; C7 and C1 reproduce within 1 %; the PLAN2 3x-iteration convergence check was not run); C6 E 6.55 at d 0.023; C4 E 6.65 at d 0.025; C5 plateaus at E 7.7. Files: `frontier_*.csv`.

Transfer (`transfer2.py`, step of 0.005 RMS along the guidance gradient through the Tweedie estimate, t in 50..600): the C7 guided step removes 0.77-1.13 E points against 0.07-0.26 for C0, C1, C6, cosine with the clean gradient 0.63-0.69 for C7 against 0.70-0.85 for the others. Only one step size was tested and it overshoots for the concentrated gradients of C0, C1, C6 (a step along their own clean gradient raises E: dE_clean +0.07 / +0.10 / +0.007), so this shows that the C7 guided direction lowers E, not that C7 transfers better than the others. File: `transfer2.csv`.

Gates (`gates.py`, cold start at the lam of the band): on 16 data states the same lam moves them by at most 2.2e-5 and lowers E from 0.385 to 0.315; stable pairs have median shift 0 and negative mean shift (no hidden margin); level-mean variance C7 3.65e-3 / 3.53e-3 vs C0 3.42e-3. Files: `gates_*.csv`.

Robustness (`robust.py`, files `robust_*.csv`):

- eps scale: q90 of L_C7(x0_hat) on 48 data states x 3 draws rises from 8.6e-8 (t = 0) to 1.2e-4 (t = 5), 3.6e-3 (t = 800), 0.11 (t = 999).
- Neutral pile-up: the near-neutral volume (0 <= g < 1e-3) sits mostly in levels 25-34: 3.8-5.0 % for C7, 4.1-4.4 % for C0 / C1, against 0.11 % in data. All data inversions are below 1.2e-4 (E_tiny = E = 0.361), i.e. rounding-level.
- Re-noise to t = 10 / 20 / 50 and one UNet x0_hat: data 0.36 -> 0.98 / 1.02 / 1.09, C7 (lam 3e-3) 2.63 -> 5.37 / 5.56 / 5.77, C7 (lam 1e-3) 4.80 -> 5.85 / 6.01 / 6.23, C1 3.57 -> 5.15 / 5.31 / 5.43, C0 1.26 -> 4.67 / 4.89 / 5.15, none 9.28 -> 7.85. Post-hoc gains are largely undone by one denoising pass, and C7 comes back slightly higher than C0 / C1 at comparable clean E: the deep neutral pile-up is fragile. In guided sampling the guidance keeps acting in the last steps, so this is an upper bound on the loss, but it suggests a floor of the guided E well above the post-hoc one.

User decisions (2026-10-18): C7 only; usual pipeline (raw mode, constant eps = clean-data q90 8.6e-8, lbd0 1, BorderZero, seed 0, 1000 steps), eta is a tuned hyperparameter as for mean_density / zero_mean in `inference/optimal/`; sweep eta in {30, 100, 300, 1000} at batch 8 (eta range from matching eta L |grad L| with the optimal runs), then one batch-16 run at the chosen eta. eps(t) is kept as a fallback only.

Inspector review (2026-10-18, verdict GO with fixes): the E reimplementation and the monotonicity-only rule check out. The eta grid {30..1000} was about 10x too high, because the loss on the Tweedie estimate is front-loaded (int L dt about 11 on data, 94 % from t >= 800) while the reference runs only take lbd from 1 to about 5; revised grid {10, 30, 100, 300}, run sequentially from 30. The band is anchored on the optimal runs, paired with `optimal/none_border0/151026-145216` (b8, seed 0, E 9.28): 0.36 <= E < 8.3, preferably below 5.5 (optimal mean_density, E 5.50 +/- 1.69; zero_mean 1.27). In float32, g is quantised to 6.1e-5 / 1.22e-4, so tau = 1e-4 is a no-op and C7 is in practice a volume-weighted linear hinge; about 1 point of E is invisible to the float32 loss. eps = 8.6e-8 never binds, so eta is the only control. Selection rule for the batch-16 run: largest E reduction with E >= 0.36, level-mean variance >= data, deep near0 no worse than post-hoc C7, distance <= mean_density, roughness within +10 % of the paired none; ties (within one std of the paired per-sample differences) go to the smaller eta.

## Step 4: guided sampling with C7 (`negative_grad_density_huber`), 2026-10-18

Runs of `Results/notebooks/generate/4_generate_huber_sweep.py` (raw, eps = eps_data = 8.6e-8, lbd0 1, BorderZero, seed 0, 1000 steps, about 8 min per b8 run on this Mac), metrics of `sweep_eval.py` (`sweep_eval.csv`), b8 runs paired with `optimal/none_border0/151026-145216` (E 9.28), the b16 run with `151026-150424` (E 8.31).

| run | E | dist | level_mean | rough | near0 deep | lbd_end |
|---|---|---|---|---|---|---|
| eta 10 b8 | 2.80 | 0.048 | 2.29e-3 | 3.17e-3 | 0.23 | 96 |
| eta 30 b8 | 1.74 | 0.068 | 2.22e-3 | 3.16e-3 | 0.13 | 186 |
| eta 100 b8 | 0.75 | 0.117 | 3.16e-3 | 3.06e-3 | 0.03 | 409 |
| eta 300 b8 | 0.29 | 0.219 | 5.33e-3 | 2.83e-3 | 0.008 | 841 |
| eta 100 b16 (`optimal/negative_grad_density_huber_border0`) | 0.75 | 0.121 | 3.67e-3 | 3.09e-3 | 0.05 | 412 |
| mean_density optimal b16 | 5.50 | 0.216 | 1.95e-3 | 3.32e-3 | 0.93 | |
| zero_mean optimal b16 | 1.27 | 0.098 | 2.8e-5 | 3.41e-3 | 0.10 | |
| data (16) | 0.385 | | 9.6e-5 | 1.22e-3 | 0.11 | |

Selection rule (inspector): eta 300 fails (E below data, distance above mean_density, deep near-neutral volume far below data; the paper box metrics show it warms the bottom and deep water, BWbox toce 5.3 and DWbox toce 3.4 against 4.8 and 2.6 in data). eta 100 is the strongest that passes and is not a tie with eta 30. The post-hoc deep neutral pile-up does not appear in guided sampling. Board with Table 1 and the zonal sections : `../huber_board.html` (built by `../build_huber_board.py`).

User choice (2026-10-18): the paper run is eta 10. `optimal/negative_grad_density_huber_border0/181026-172751` (eta 10, b16, E 2.79, dist 0.050); the eta 100 b16 run (181026-165339) moved to `sweep_density_huber_b16/`, the `prefix` of both args.json updated to their new folder.
