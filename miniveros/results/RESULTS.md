# Results (current design, 2026-09-22)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped at 3 sigma; land and padding re-imposed after every step at the noise level of the
step (`fill_mode=noised`, the default since `609febe`); 32 samples per hold-out run and per grid point. Training
at `2b542aa`, sampling and evaluation at `bfee32f`. Runs: `fs_scattered_3std`, `fs_band3_3std`, `fs_top_3std`.

## Hold-out metrics (mean over the held-out runs, water cells, against the true 20-year time-mean)

| | scattered (10 interior points) | band of three rows, `c_k` 0.126, 0.2, 0.3175 (30 runs) | top row, `c_k` 0.8 (10 runs) |
|---|---|---|---|
| training runs | 90 | 70 | 90 |
| gap the training rows bridge | one step, all four sides | 0.0794 to 0.504, factor 6.3 | extrapolation, nothing above |
| diffusion ensemble mean of 32 / RMSE (K) | 0.061 ± 0.013 | 0.076 ± 0.033 | 0.141 ± 0.108 |
| diffusion single sample / RMSE (K) | 0.147 | 0.160 | 0.208 |
| neighbour average / RMSE (K) | 0.020 | 0.116 | 0.170 |
| nearest training run / RMSE (K) | 0.037 | 0.116 | 0.170 |
| training-set mean / RMSE (K) | 0.155 | 0.197 | 0.470 |
| diffusion / domain-mean bias (K) | +0.042 | +0.020 | -0.016 |
| ensemble spread / true spread (K) | 0.128 / 0.031 | 0.132 / 0.031 | 0.132 / 0.032 |
| T-inversion fraction, generated / truth | 10.2 % / 7.7 % | 10.9 % / 1.3 % | 12.3 % / 0.1 % |
| max salinity error (psu) | 0.013 | 0.014 | 0.017 |
| W1 profile, diffusion (K) | 0.066 | 0.075 | 0.132 |
| W1 profile, neighbour average (K) | 0.036 | 0.095 | 0.142 |
| W1 profile, nearest training run (K) | 0.044 | 0.095 | 0.142 |
| W1 profile, training-set mean (K) | 0.136 | 0.160 | 0.420 |
| grid W1, all / hold-out / training points (K) | 0.082 / 0.071 / 0.083 | 0.079 / 0.076 / 0.080 | 0.079 / 0.133 / 0.073 |
| grid domain-mean T RMSE, all / hold-out (K) | 0.047 / 0.041 | 0.044 / 0.034 | 0.043 / 0.073 |
| same models, exact fill after every step (previous sampler) / RMSE (K) | 0.152 ± 0.007 | 0.135 ± 0.019 | 0.173 ± 0.081 |
| same models, exact fill / W1 (K) | 0.127 | 0.111 | 0.143 |

Band of three, by row (RMSE in K): diffusion 0.063 / 0.074 / 0.091 for `c_k` 0.126 / 0.2 / 0.3175, nearest row
0.063 / 0.132 / 0.152, training mean 0.149 / 0.187 / 0.256. Top row by `c_eps`: the model beats copying the row
below from 0.139 to 1.4 (0.05 to 0.26 K vs 0.08 to 0.31 K), ties at 2.222 (0.06 vs 0.06 K), loses at the corner
(0.41 vs 0.35 K) and in the flat regime (`c_eps` >= 3.5: 0.09 to 0.10 vs 0.03 to 0.08 K).

## W1 metric

Each state is reduced to its horizontal-mean temperature profile over water cells. Per level, W1 between the
32 generated values and all 241 true snapshots of the window (quantile form; a subsample of 32 would only add
noise); levels combined with thickness weights dz/H from the level midpoints (H = 2080 m). Point predictions
(baselines): W1 = mean |x - T_i|. Rewards a correct spread, does not reward collapse to the mean, ignores
horizontal structure (kept by the RMSE). Code in `wmetrics.py`.

## What the results say

1. **The land/padding constraint at sampling was the main error source.** Same models, sampling only: exact
   zeros re-imposed after every DDPM step (DINO's BorderZeroConstraint) give 0.152 / 0.135 / 0.173 K; zeros
   re-imposed at the noise level of the step (sqrt(alpha_bar) fill + sqrt(1 - alpha_bar) z, as those cells
   looked in training) give 0.061 / 0.076 / 0.141 K. At high noise levels the network had never seen a solid
   block of zeros around the domain. Gain largest at the bottom (0.34 -> 0.07 K, scattered; 0.28 -> 0.10 band);
   the earlier "mean-state collapse" was this artefact: 40 % of the mean-state error now, and grid W1 at the
   training points 0.124 -> 0.083 K. The ensemble spread grows (0.11 -> 0.13 K), the single-sample RMSE hardly
   moves (0.188 -> 0.147): the ensemble mean now averages real sampler spread instead of a shared bias.
2. **Model vs baselines.** Band: beats the nearest run at every level below 26 m and in all three rows. Top row:
   beats the nearest run at every level but the surface and the bottom, and copying the row below for
   `c_eps` 0.14 to 1.4; loses at the corner and in the flat regime. Scattered: loses to the one-step neighbours
   (0.061 vs 0.037 / 0.020 K) except at 490 to 650 m.
3. **Depth structure.** 0.04 to 0.08 K at every level in the scattered split; the bottom two levels are no longer
   special (0.07 K) except in the top row (0.29 K, the corner runs). Worst band: the top 100 m (0.07 to 0.12 K),
   where 3 sigma_z is about 12 K and sampler noise is amplified.
4. **The sampler's clip is a regulariser, not a range limit.** Exact fill, scattered / top: clip at 1 gives
   0.152 / 0.173 K, at 1.5 0.194 / 0.183, at 3 0.302 / 0.277, although the final samples barely exceed
   |x'| = 1 (0.1 %). Under min-max, where |x'| <= 1 is the data range, keeping the clip at mu +- 3 sigma instead
   (`clip_ref=3-std`, per-channel bounds) is worth 0.126 -> 0.059 K (scattered) and 0.129 -> 0.079 (band); top
   row unchanged (0.133 vs 0.137). The price is a range limit: 12 % of the held-out top row's true bottom values
   lie above mu + 3 sigma and cannot be generated; the corner run is the largest top-row error in every variant.
5. **Inversions.** 10 to 12 % of interfaces with temperature decreasing upward whatever the regime and the
   sampler; the truth goes from 7.7 % (scattered set) to 1.3 % (band) to 0.1 % (top row). Not learned; this is
   the kind of constraint DINO-Fusion imposes at sampling time.
6. **Plumbing.** Salinity within 0.02 psu of 35 without any constraint, land exact, spread 4x the true
   within-window spread.

## Normalisation: per-level min-max (runs `fs_*_minmax`, training at `4fe5d51`)

`norm_mode=minmax`: each level's data range [min_z, max_z] (all runs, water cells) -> [-1, 1]; land and padding
hold the normalised level mean (within +-0.42 of the midpoint) instead of 0. Same splits and training settings
as the 3-std runs; four samplers, sampling only (RMSE of the ensemble mean, K; W1 in brackets):

| sampler | scattered | band of three | top row |
|---|---|---|---|
| noised fill, clip at mu +- 3 sigma (`eval_nfc3`, the 3-std reference sampler) | 0.059 (0.072) | 0.079 (0.083) | 0.137 (0.133) |
| noised fill, clip at the data range (`eval`, default config) | 0.126 (0.105) | 0.129 (0.106) | 0.133 (0.104) |
| exact fill, clip at the data range (`eval_cleanfill`, the first result) | 0.199 (0.165) | 0.233 (0.188) | 0.288 (0.241) |
| exact fill, clip at mu +- 3 sigma (`eval_c3`) | 0.669 (0.361) | 0.701 (0.391) | 0.707 (0.423) |
| 3-std reference, noised fill, clip at 1 | 0.061 (0.066) | 0.076 (0.075) | 0.141 (0.132) |

- With the same sampler the two normalisations are on par: depth profiles within 0.02 K at every level except
  106 m (0.11 vs 0.07 K, band); band rows 0.070 / 0.076 / 0.091 vs 0.063 / 0.074 / 0.091; grid W1 at training
  points 0.085 vs 0.083. The normalisation is immaterial; the sampler is not.
- The first min-max result (0.199 / 0.233 / 0.288, "worse on every split") was two sampler confounds at once:
  the exact fill, up to +-0.42 there (0.199 -> 0.126 when noised), and the clip at the data range instead of
  mu +- 3 sigma (0.126 -> 0.059 with the 3-sigma bounds).
- Exact fill with the mu +- 3 sigma clip is pathological (0.67 to 0.71 K, |bias| 0.3 to 0.4 K): a fill the
  network never saw plus asymmetric per-channel bounds drive the chain to a wrong state.
- The smaller min-max salinity error (0.004 to 0.007 vs 0.013 to 0.017 psu) is the floor, not learning:
  minmax floors the half-range at 0.05 where 3-std floors sigma at 0.05 and multiplies by 3.

## Cost

Extraction 2.5 min on 8 CPU cores (once). Training 17 to 22 min on one A100 per split. Generation of the 32-sample
hold-out and grid sets plus evaluation 14 to 18 min. A split costs under 40 GPU minutes end to end; a sampling
variant 15 GPU minutes.

## Suggested next steps

- Stratification constraint at sampling time (DINO-Fusion's isotonic projection, on T since S is constant).
- Top-row corner: the clip's range limit is now the dominant extrapolation error; a clip that follows the
  conditioning (bounds from the nearest training rows) is the next sampling-only test.
- Stronger conditioning: classifier-free guidance; the year as a third condition (drift of 0.45 K at the bottom
  inside the window). A second training seed to size the seed noise.
- Evaluate against snapshots as well as the time mean (nearest-snapshot RMSE, spread-skill).

## Figures

Per run (`fs_scattered_3std/`, `fs_band3_3std/`, `fs_top_3std/`, and the min-max runs `fs_*_minmax/`): `config.json`, `git_hash.txt`, `train_log.csv`,
`samples_final.png`, `samples_levels.png` (true state and three random samples of T and S at three depths for one
hold-out condition, made with `plot_samples.py`), and `eval/` (default sampler: noised fill) with `summary.txt`, `metrics.csv`, `grid_maps.png`
(+ `grid_domain_mean.csv`, `grid_w1.csv`) and `profiles.png` (+ `profiles.csv`); `eval_cleanfill/` = the previous sampler (exact
fill); for the min-max runs also `eval_c3/` (exact fill, clip at mu +- 3 sigma) and `eval_nfc3/` (noised fill, clip at
mu +- 3 sigma, the sampler of the 3-std reference; the min-max figures of the report).
`data/T_distribution_per_level.png`, `data/T_per_level_stats.npz`: per-level T distribution (`tdist.py compute` on the
cluster, `tdist.py plot` locally).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
