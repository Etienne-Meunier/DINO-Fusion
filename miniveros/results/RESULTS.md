# Results (current design, 2026-09-22)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped at 3 sigma; land and padding re-imposed after every step at the noise level of the
step (`LandZero`); 32 samples per hold-out run and per grid point. Training
at `2b542aa`, sampling and evaluation at `bfee32f`. Runs: `fs_scattered_3std`, `fs_band3_3std`, `fs_top_3std` (training at `2b542aa`), `fs_block_3std` (`338997d`),
`fs_ring_3std` (`61d0dd2`), `fs_block5_3std` and `fs_ring5_3std` (`8f157d5`).

## Hold-out metrics (mean over the held-out runs, water cells, against the true 20-year time-mean)

Seven splits of the 10 x 10 grid. Interpolation: scattered (10 interior points, 90 training runs), band of three
rows (`c_k` 0.126, 0.2, 0.3175; 30 runs; training rows a factor 6.3 apart), centre 5 x 5 (`split_block=3,8,3,8`:
`c_k` 0.05 to 0.3175, `c_eps` 0.35 to 2.222; 25 runs; training on 75), centre 7 x 7 (`split_block=2,9,2,9`: `c_k`
0.0315 to 0.504, `c_eps` 0.2205 to 3.528; 49 runs; training on the outer rows and columns). Extrapolation: outer 51
(the complement of the 7 x 7 block, training on its centre), outer 75 (the complement of the 5 x 5 block, training
on 25 runs), top row (`c_k` 0.8; 10 runs). Four runs (`split_mode=points`, `split_points=1,1,1,8,8,1,8,8`): training
on `c_k` {0.0198, 0.504} x `c_eps` {0.139, 3.53}, one step in from each corner; 96 held out. Runs `fs_scattered_3std`,
`fs_band3_3std`, `fs_block5_3std`, `fs_block_3std`, `fs_ring_3std`, `fs_ring5_3std`, `fs_top_3std`, `fs_four_3std`
(training at `812ff96`), `fs_block5_v_3std` and `fs_ring5_v_3std` (`1601947`, v-prediction without clip).

| split | runs | RMSE mean of 32 | RMSE one sample | RMSE nearest run | RMSE neighbour avg | RMSE mean state | bias | spread (true) | W1 diffusion | W1 nearest | W1 mean state | inversions gen / truth | max S error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scattered | 10 | 0.061 ± 0.013 | 0.147 | 0.037 | 0.020 | 0.155 | +0.042 | 0.128 (0.031) | 0.066 | 0.044 | 0.136 | 10.2 % / 7.7 % | 0.013 |
| band of three rows | 30 | 0.076 ± 0.033 | 0.160 | 0.116 | 0.116 | 0.197 | +0.020 | 0.132 (0.031) | 0.075 | 0.095 | 0.160 | 10.9 % / 1.3 % | 0.014 |
| centre 5 x 5 | 25 | 0.061 ± 0.013 | 0.153 | 0.077 | 0.076 | 0.157 | +0.036 | 0.134 (0.031) | 0.070 | 0.069 | 0.133 | 10.4 % / 3.6 % | 0.013 |
| centre 7 x 7 | 49 | 0.064 ± 0.016 | 0.153 | 0.114 | 0.113 | 0.197 | +0.032 | 0.132 (0.031) | 0.073 | 0.095 | 0.166 | 10.4 % / 4.2 % | 0.013 |
| outer 51 | 51 | 0.110 ± 0.136 (median 0.061) | 0.185 | 0.131 | 0.131 | 0.258 | -0.002 | 0.124 (0.030) | 0.100 | 0.117 | 0.223 | 11.0 % / 6.6 % | 0.013 |
| outer 75 | 75 | 0.138 ± 0.197 (median 0.064) | 0.213 | 0.155 | 0.155 | 0.220 | -0.021 | 0.126 (0.030) | 0.118 | 0.137 | 0.189 | 10.6 % / 6.1 % | 0.012 |
| top row | 10 | 0.141 ± 0.108 | 0.208 | 0.170 | 0.170 | 0.470 | -0.016 | 0.132 (0.032) | 0.132 | 0.142 | 0.420 | 12.3 % / 0.1 % | 0.017 |
| four runs | 96 | 0.154 ± 0.113 (median 0.120) | 0.214 | 0.179 (median 0.070) | 0.179 | 0.254 | +0.054 | 0.125 (0.031) | 0.106 | 0.156 | 0.208 | 10.5 % / 5.4 % | 0.014 |
| centre 5 x 5, v-prediction, no clip | 25 | 0.044 ± 0.012 | 0.079 | 0.077 | 0.076 | 0.157 | +0.012 | 0.048 (0.031) | 0.027 | 0.069 | 0.133 | 9.7 % / 3.6 % | 0.046 |
| outer 75, v-prediction, no clip | 75 | 0.128 ± 0.195 (median 0.048) | 0.145 | 0.155 | 0.155 | 0.220 | -0.063 | 0.041 (0.030) | 0.101 | 0.137 | 0.189 | 10.2 % / 6.1 % | 0.028 |

| split | grid W1 all / hold-out / training (K) | grid domain-mean T RMSE all / hold-out (K) |
|---|---|---|
| scattered | 0.082 / 0.071 / 0.083 | 0.047 / 0.041 |
| band of three rows | 0.079 / 0.076 / 0.080 | 0.044 / 0.034 |
| centre 5 x 5 | 0.082 / 0.068 / 0.087 | 0.046 / 0.038 |
| centre 7 x 7 | 0.084 / 0.072 / 0.095 | 0.048 / 0.036 |
| outer 51 | 0.081 / 0.102 / 0.059 | 0.064 / 0.086 |
| outer 75 | 0.102 / 0.117 / 0.057 | 0.119 / 0.137 |
| top row | 0.079 / 0.133 / 0.073 | 0.043 / 0.073 |
| four runs | 0.106 / 0.107 / 0.084 | 0.105 / 0.107 |
| centre 5 x 5, v-prediction | 0.028 / 0.027 / 0.028 | 0.014 / 0.017 |
| outer 75, v-prediction | 0.083 / 0.101 / 0.028 | 0.136 / 0.157 |

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

1. **Interpolation: the model's error is gap-independent, the baselines' is not.** Scattered / band / centre
   5 x 5 / centre 7 x 7: model 0.061 / 0.076 / 0.061 / 0.064 K, nearest run 0.037 / 0.116 / 0.077 / 0.114 K. Band:
   beats the nearest run at every level below 26 m and in all three rows. Centre blocks: the error is flat over
   the block (0.04 to 0.12 K) while the nearest run reaches 0.18 K at the centre of either block, where the model
   stays at 0.05 K; 11 of 25 and 28 of 49 wins, the losses in the flat regime (low `c_k`, high `c_eps`) where a
   copied neighbour is within 0.05 K. Beats the nearest run from 182 m (5 x 5) and 26 m (7 x 7) to the bottom.
   Worst run of either block: the warm edge (ck0.504_eps0.2205 in the 7 x 7, 0.12 K).
2. **Extrapolation: the warm corner sets the mean, the rest behaves like interpolation.** Outer 51 / outer 75
   / top row: mean 0.110 / 0.138 / 0.141 K, median 0.061 / 0.064 / 0.11 K, nearest run 0.131 / 0.155 / 0.170 K. The
   corner run ck0.8_eps0.0875 errs by 0.77 / 1.10 / 0.41 K with a cold bias (-0.43 / -0.68 / -0.19 K): the model
   does not carry the warming beyond the last training row, and the error grows with the distance to it (two,
   three, one cells). Away from the corner: outer 51 without the top row 0.084 vs 0.100 K; outer 75 without the
   top row and the two ck0.504 corner runs 0.089 vs 0.095 K. Beats the nearest run from about 150 m to the
   bottom and at 24 / 51, 32 / 75, 7 / 10 runs; the nearest run is worse still at the corner (0.92 / 1.24 / 0.35
   K). Top row by `c_eps`: beats copying the row below from 0.139 to 1.4, ties at 2.222, loses at the corner and
   in the flat regime.
2c. **Four training runs** (one step in from each corner). Mean 0.154 vs nearest 0.179 K and mean state 0.254 K,
   W1 0.106 vs 0.156 K, but median 0.120 vs 0.070 K and 31 wins of 96. The model's error is a smooth field over
   the grid: 0.05 to 0.16 K in the flat regime, 0.2 to 0.45 K on the warm side with a warm bias up to +0.27 K
   between the two training rows, 0.70 K at the corner (cold). The nearest run is far better where the state
   hardly changes (0.01 to 0.07 K) and far worse where it does (0.5 to 0.85 K in the warm half of the interior):
   inside the training rectangle 0.159 vs 0.201 K, outside 0.146 vs 0.143 K. Beats the nearest run at every level
   below 250 m.
2d. **v-prediction removes the clip and halves the single-sample error** (runs `fs_block5_v_3std`, `fs_ring5_v_3std`,
   training at `1601947`, `prediction_type=v_prediction clip_sample=false`). Centre 5 x 5: 0.044 vs 0.061 K (eps +
   clip), single sample 0.079 vs 0.153, W1 0.027 vs 0.070 (nearest 0.069), spread 0.048 vs 0.134 (truth 0.031),
   wins 15 vs 11 of 25, beats the nearest run from 106 m down, grid W1 at the training points 0.028 vs 0.087,
   grid domain-mean RMSE 0.014 vs 0.046. Outer 75: 0.128 vs 0.138, single sample 0.145 vs 0.213, W1 0.101 vs
   0.118, wins 44 vs 32 of 75, flat regime 0.03 to 0.05 K; the warm corner is unchanged (1.03 vs 1.10 K at the
   corner run), so its error is the extrapolation, not the clip. Control (`eval_noclip` on the eps models): without
   the clip the eps chain diverges (RMSE 552 and 572 K), so under eps-prediction the clip is what keeps it finite.
   Caveat: every v sample carries a small spot of wrong values at the tip of the ridge (S error 0.046 vs 0.013 psu).
3. **Depth structure.** 0.03 to 0.08 K at every level in the interpolation splits; the bottom two levels are no longer
   special (0.07 K) except in the top row (0.29 K, the corner runs). Worst band: the top 100 m (0.07 to 0.12 K),
   where 3 sigma_z is about 12 K and sampler noise is amplified.
4. **Why the surface is worst in kelvin.** The per-level scale 3 sigma_z is 12 K at the surface and 1.6 K at the
   bottom, and the sampler's precision is roughly uniform in normalised units: surface RMSE 0.6 % of its scale
   (0.073 K, scattered) vs 1 % at 650 m and 4.5 % at the bottom. The run-to-run surface signal is 0.7 % of the
   scale (0.085 K), the same size, so the model cannot tell the runs apart there and matches the mean state
   (0.076 vs 0.073 K); the nearest run wins (0.026 K) only because neighbouring runs are nearly identical at the
   surface. What remains is a per-sample offset of the surface mean: W1 on the horizontal-mean profile is 0.12 K
   at 14 m, above the 0.07 K cell RMSE of the ensemble mean, so each sample carries a nearly uniform shift of
   about 0.1 K (0.01 normalised) that the average of 32 mostly removes.
5. **The sampler's clip trades regularisation for range.** Same models, sampling only, scattered / top row
   (`eval_c15`, `eval_c3` on the cluster): clip at |x'| <= 1 gives 0.061 / 0.141 K, at 1.5 0.061 / 0.108 K, at 3
   0.091 / 0.123 K; W1 0.066 / 0.132, 0.086 / 0.117, 0.147 / 0.154 K. Widening to 1.5 costs nothing in
   interpolation RMSE, widens the sample spread (W1) and frees the top-row corner (12 % of its true bottom values
   lie above mu + 3 sigma); at 3 the regularisation is lost.
6. **Inversions.** 10 to 12 % of interfaces with temperature decreasing upward whatever the regime and the
   sampler; the truth is 7.7 % (scattered), 6.6 and 6.1 % (outer 51 and 75), 4.2 and 3.6 % (centre 7 x 7 and 5 x 5),
   1.3 % (band), 0.1 % (top row). Not learned; this is
   the kind of constraint DINO-Fusion imposes at sampling time.
7. **Plumbing.** Salinity within 0.02 psu of 35 without any constraint, land exact, spread 4x the true
   within-window spread.

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

Per run (`fs_scattered_3std/`, `fs_band3_3std/`, `fs_block5_3std/`, `fs_block_3std/`, `fs_ring_3std/`, `fs_ring5_3std/`,
`fs_top_3std/`, `fs_four_3std/`, `fs_block5_v_3std/`, `fs_ring5_v_3std/`): `config.json`, `git_hash.txt`, `train_log.csv`,
`samples_final.png`, `samples_levels.png` (true state and three random samples of T and S at three depths for one
hold-out condition, made with `plot_samples.py`), and `eval/` (default sampler: noised fill) with `summary.txt`, `metrics.csv`, `grid_maps.png`
(+ `grid_domain_mean.csv`, `grid_w1.csv`) and `profiles.png` (+ `profiles.csv`).
`data/level_density_150m.png`: one T and S density per run at 182 m (`level_density.py`, reads the raw run files);
the narrow peaks of the T distribution are the zonally uniform southern rows y = 0 to 6 (restoring zone).
`data/T_distribution_per_level.png`, `data/T_per_level_stats.npz`: per-level T distribution (`tdist.py compute` on the
cluster, `tdist.py plot` locally).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
