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
