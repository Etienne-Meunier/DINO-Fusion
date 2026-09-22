# Results (current design, 2026-09-21)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped at 3 sigma, 32 samples per hold-out run and per grid point. Code `5fdcf01`
(training at `2b542aa`). Runs: `fs_scattered_3std`, `fs_band3_3std`, `fs_top_3std`.

## Data: distribution of T per level

`data/T_distribution_per_level.png` (made with `tdist.py`, statistics in `data/T_per_level_stats.npz`): all 100 runs
x 241 snapshots, water cells, before and after the 3-std normalisation. Above 400 m the histogram is a mixture of
restoring values and interior and sigma_z is set by the meridional gradient, so mu +- 3 sigma is wider than the
level and the normalised values stay within [-0.85, 0.6] ([-0.7, 0.4] in the top 200 m); below 1400 m the
distribution is skewed with a warm tail
to x' = +2.7 (the corner runs). Run-to-run signal sigma_runs / 3 sigma_z: 0.7 % at the surface, 9 % at 650 m,
26 % at the bottom.

## Hold-out metrics (mean over the held-out runs, water cells, against the true 20-year time-mean)

| | scattered (10 interior points) | band of three rows, `c_k` 0.126, 0.2, 0.3175 (30 runs) | top row, `c_k` 0.8 (10 runs) |
|---|---|---|---|
| training runs | 90 | 70 | 90 |
| gap the training rows bridge | one step, all four sides | 0.0794 to 0.504, factor 6.3 | extrapolation, nothing above |
| diffusion ensemble mean of 32 / RMSE (K) | 0.152 ± 0.007 | 0.135 ± 0.019 | 0.173 ± 0.081 |
| diffusion single sample / RMSE (K) | 0.188 | 0.178 | 0.214 |
| neighbour average / RMSE (K) | 0.020 | 0.116 | 0.170 |
| nearest training run / RMSE (K) | 0.037 | 0.116 | 0.170 |
| training-set mean / RMSE (K) | 0.155 | 0.197 | 0.470 |
| diffusion / domain-mean bias (K) | -0.027 | -0.027 | -0.071 |
| ensemble spread / true spread (K) | 0.109 / 0.031 | 0.111 / 0.031 | 0.115 / 0.032 |
| T-inversion fraction, generated / truth | 9.6 % / 7.7 % | 10.7 % / 1.3 % | 12.1 % / 0.1 % |
| max salinity error (psu) | 0.011 | 0.013 | 0.018 |
| W1 profile, diffusion (K) | 0.127 | 0.111 | 0.143 |
| W1 profile, neighbour average (K) | 0.036 | 0.095 | 0.142 |
| W1 profile, nearest training run (K) | 0.044 | 0.095 | 0.142 |
| W1 profile, training-set mean (K) | 0.136 | 0.160 | 0.420 |
| grid W1, all / hold-out / training points (K) | 0.125 / 0.127 / 0.124 | 0.119 / 0.110 / 0.123 | 0.119 / 0.141 / 0.117 |

Band of three, by row (RMSE in K): diffusion 0.140 / 0.139 / 0.127 for `c_k` 0.126 / 0.2 / 0.3175, nearest row
0.063 / 0.132 / 0.152, training mean 0.149 / 0.187 / 0.256. Top row by `c_eps`: the model beats copying the row
below from 0.139 to 0.556 (0.12 to 0.23 K vs 0.18 to 0.31 K), loses at the corner (0.40 vs 0.35 K) and in the
flat regime (`c_eps` >= 0.88: 0.12 to 0.16 vs 0.03 to 0.11 K).

## W1 metric

Each state is reduced to its horizontal-mean temperature profile over water cells. Per level, W1 between the
32 generated values and all 241 true snapshots of the window (quantile form; a subsample of 32 would only add
noise); levels combined with thickness weights dz/H from the level midpoints (H = 2080 m). Point predictions
(baselines): W1 = mean |x - T_i|. Rewards a correct spread, does not reward collapse to the mean, ignores
horizontal structure (kept by the RMSE). Code in `wmetrics.py`.

## What the results say

1. **Conditioning is weak under per-level scaling.** The run-to-run signal is 0.7 % of the normalised range at
   the surface, 9 % at 650 m, 25 % at the bottom; the epsilon loss weights every cell equally, so the shared
   structure dominates. Scattered split: RMSE = mean-state baseline. Under W1 the training grid points score
   like the held-out ones (0.124 vs 0.127 K): the model does not fit the training conditions either.
2. **Model error is nearly split-independent (0.14 to 0.17 K) while the baselines degrade with the gap
   (0.02 to 0.17 K).** The model beats the nearest run in the upper band row, ties the middle one, loses the lower
   one; on the top row it ties copying the row below on average and beats it where there is a trend to continue.
3. **Depth structure.** Best method between 370 and 800 m in the band split (0.08 vs 0.14 K at 650 m) and
   between 260 and 1430 m in the top row; worse than the baselines in the top 250 m (the per-level scale turns
   small normalised noise into 0.1 K of scatter) and at the two bottom levels (skewed, drifting distribution).
4. **The sampler's clip is a regulariser.** Same trained models, sampling only: clip at 1 gives 0.152 / 0.173 K
   (scattered / top), at 1.5 it gives 0.194 / 0.183 K, at 3 it gives 0.302 / 0.277 K, although the final samples
   barely exceed |x'| = 1 (0.1 %). Clipping the early, inaccurate clean-state estimates keeps the chain on track.
   The price is a range limit for extrapolation: 12 % of the held-out top row's true bottom values lie above
   mu + 3 sigma and cannot be generated.
5. **Inversions.** The model generates 10 to 12 % of interfaces with temperature decreasing upward whatever the
   regime; the truth goes from 7.7 % (scattered set) to 1.3 % (band) to 0.1 % (top row). Not learned; this is
   the kind of constraint DINO-Fusion imposes at sampling time.
6. **Plumbing.** Salinity within 0.02 psu of 35 without any constraint, land exact, spread 3.5x the true
   within-window spread.

Previous design (statistics per split, 8 then 32 samples): 0.150 / 0.129 / 0.193 K; fixed statistics changed
the scattered and band numbers within noise and improved the top row from 0.193 to 0.173 K.

## Normalisation: per-level min-max (2026-09-22, runs `fs_*_minmax`, code `4fe5d51`)

`norm_mode=minmax`: each level's data range [min_z, max_z] (all runs, water cells) -> [-1, 1]; land and padding
hold the normalised level mean (within +-0.42 of the midpoint) instead of 0. Same splits, training and sampling
settings as the 3-std runs.

| | scattered | band of three | top row |
|---|---|---|---|
| diffusion ensemble mean of 32 / RMSE (K), minmax vs 3-std | 0.199 vs 0.152 | 0.233 vs 0.135 | 0.288 vs 0.173 |
| diffusion single sample / RMSE (K) | 0.205 vs 0.188 | 0.238 vs 0.178 | 0.293 vs 0.214 |
| ensemble spread (K) | 0.051 vs 0.109 | 0.056 vs 0.111 | 0.061 vs 0.115 |
| W1 profile, diffusion (K) | 0.165 vs 0.127 | 0.188 vs 0.111 | 0.241 vs 0.143 |
| T-inversion fraction, generated | 9.1 % | 9.9 % | 10.9 % |
| max salinity error (psu) | 0.004 | 0.004 | 0.005 |

- Worse on every split and at almost every level: the 100 to 400 m band grows 1.2 to 3.4x (0.13 -> 0.24 K at
  106 m, scattered; 0.06 -> 0.19 K at 374 m) and the bottom level goes 0.34 -> 0.43 / 0.28 -> 0.54 / 0.35 -> 0.66 K.
  In the scattered split 70 m improves (0.20 -> 0.15 K) and 490 m and 1000 to 1200 m are unchanged; in the band
  and top splits 1000 to 1200 m are 1.25 to 1.6x worse too.
- Worse than the mean training state in the interior splits (0.199 vs 0.155, 0.233 vs 0.197 K) and at the
  training grid points (grid W1 at training points 0.190 / 0.185 / 0.171 vs 0.124 / 0.123 / 0.117 K): the model
  fits the training conditions less, not only generalises less.
- A single sample scores like the ensemble mean (0.205 vs 0.199 K; RMSE_sample^2 = RMSE_mean^2 + spread^2 holds
  in both modes) and the domain-mean |bias| does not grow (0.048 / 0.034 / 0.065 vs 0.027 / 0.029 / 0.076 K): the
  loss is structured, level-dependent error, not a shift of the mean. The top-row grid map shows a warm bias of
  0.00 to 0.09 K over the flat regime and -0.19 K at the corner. The halved spread is mechanical: above 260 m
  the min-max scale is 0.50 to 0.63 of 3 sigma.
- The cause is not isolated. Two confounds come with the map: (i) the clip at |x'| <= 1 now sits at each
  level's data range, so the sampler loses the tighter 3-sigma regulariser (finding 4); with the top row held
  out, the bottom-level range is set by the held-out corner run (its warm tail reaches 6.45 C). (ii) Land and
  padding are reset to a clean fill of up to +-0.42 at every sampling step while training saw that fill under
  noise, a train/sampling mismatch of (1 - sqrt(alpha_bar_t)) x fill, largest at the levels that degraded
  (|fill| 0.3 to 0.4 above 260 m and at the bottom, about 0 at 650 to 1200 m where the scattered split is
  unchanged; the cold bottom bias has the sign of fill = -0.42). One training seed per configuration.
  Sampling-only tests would separate them: the min-max models with the 3-sigma bounds, and a fill re-imposed
  under the noise level of the step.
- The smaller salinity error (0.004 vs 0.011 to 0.018 psu) is the floor, not learning: the same 0.08 in
  normalised units; minmax floors the half-range at 0.05 where 3-std floors sigma at 0.05 and multiplies by 3.
- Band rows (RMSE, K): diffusion 0.217 / 0.229 / 0.252 vs nearest row 0.063 / 0.132 / 0.152. Top row: loses to
  copying the row below at every c_eps (0.20 to 0.50 vs 0.03 to 0.35 K).

## Cost

Extraction 2.5 min on 8 CPU cores (once). Training 17 to 22 min on one A100 per split. Generation of the 32-sample
hold-out and grid sets plus evaluation 14 to 18 min. A split costs under 40 GPU minutes end to end.

## Suggested next steps

- Stratification constraint at sampling time (DINO-Fusion's isotonic projection, on T since S is constant).
- Stronger conditioning: classifier-free guidance; the year as a third condition (drift of 0.45 K at the bottom
  inside the window).
- Evaluate against snapshots as well as the time mean (nearest-snapshot RMSE, spread-skill).

## Figures

Per run (`fs_scattered_3std/`, `fs_band3_3std/`, `fs_top_3std/`, and the min-max runs `fs_*_minmax/`): `config.json`, `git_hash.txt`, `train_log.csv`,
`samples_final.png`, `samples_levels.png` (true state and three random samples of T and S at three depths for one
hold-out condition, made with `plot_samples.py`), and `eval/` with `summary.txt`, `metrics.csv`, `grid_maps.png`
(+ `grid_domain_mean.csv`, `grid_w1.csv`) and `profiles.png` (+ `profiles.csv`).
`data/T_distribution_per_level.png`, `data/T_per_level_stats.npz`: per-level T distribution (`tdist.py compute` on the
cluster, `tdist.py plot` locally).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
