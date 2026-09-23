# Results (current design, 2026-09-22)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped at 3 sigma; land and padding re-imposed after every step at the noise level of the
step (`fill_mode=noised`, the default since `609febe`); 32 samples per hold-out run and per grid point. Training
at `2b542aa`, sampling and evaluation at `bfee32f`. Runs: `fs_scattered_3std`, `fs_band3_3std`, `fs_block_3std` (training at `338997d`), `fs_top_3std`.

## Hold-out metrics (mean over the held-out runs, water cells, against the true 20-year time-mean)

| | scattered (10 interior points) | band of three rows, `c_k` 0.126, 0.2, 0.3175 (30 runs) | centre block, 7 x 7 (49 runs) | top row, `c_k` 0.8 (10 runs) |
|---|---|---|---|---|
| training runs | 90 | 70 | 51 (outer rows and columns) | 90 |
| gap the training rows bridge | one step, all four sides | 0.0794 to 0.504, factor 6.3 | `c_k` 0.0198 to 0.8 (x40), `c_eps` 0.139 to 5.6 (x40) | extrapolation, nothing above |
| diffusion ensemble mean of 32 / RMSE (K) | 0.061 ± 0.013 | 0.076 ± 0.033 | 0.064 ± 0.016 | 0.141 ± 0.108 |
| diffusion single sample / RMSE (K) | 0.147 | 0.160 | 0.153 | 0.208 |
| neighbour average / RMSE (K) | 0.020 | 0.116 | 0.113 | 0.170 |
| nearest training run / RMSE (K) | 0.037 | 0.116 | 0.114 | 0.170 |
| training-set mean / RMSE (K) | 0.155 | 0.197 | 0.197 | 0.470 |
| diffusion / domain-mean bias (K) | +0.042 | +0.020 | +0.032 | -0.016 |
| ensemble spread / true spread (K) | 0.128 / 0.031 | 0.132 / 0.031 | 0.132 / 0.031 | 0.132 / 0.032 |
| T-inversion fraction, generated / truth | 10.2 % / 7.7 % | 10.9 % / 1.3 % | 10.4 % / 4.2 % | 12.3 % / 0.1 % |
| max salinity error (psu) | 0.013 | 0.014 | 0.013 | 0.017 |
| W1 profile, diffusion (K) | 0.066 | 0.075 | 0.073 | 0.132 |
| W1 profile, neighbour average (K) | 0.036 | 0.095 | 0.095 | 0.142 |
| W1 profile, nearest training run (K) | 0.044 | 0.095 | 0.095 | 0.142 |
| W1 profile, training-set mean (K) | 0.136 | 0.160 | 0.166 | 0.420 |
| grid W1, all / hold-out / training points (K) | 0.082 / 0.071 / 0.083 | 0.079 / 0.076 / 0.080 | 0.084 / 0.072 / 0.095 | 0.079 / 0.133 / 0.073 |
| grid domain-mean T RMSE, all / hold-out (K) | 0.047 / 0.041 | 0.044 / 0.034 | 0.048 / 0.036 | 0.043 / 0.073 |
| same models, exact fill after every step (previous sampler) / RMSE (K) | 0.152 ± 0.007 | 0.135 ± 0.019 | 0.136 ± 0.018 | 0.173 ± 0.081 |
| same models, exact fill / W1 (K) | 0.127 | 0.111 | 0.122 | 0.143 |

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
   zeros re-imposed after every DDPM step (DINO's BorderZeroConstraint) give 0.152 / 0.135 / 0.136 / 0.173 K
   (scattered / band / block / top); zeros re-imposed at the noise level of the step (sqrt(1 - alpha_bar) z, as
   those cells looked in training) give 0.061 / 0.076 / 0.064 / 0.141 K. At high noise levels the network had never seen a solid
   block of zeros around the domain. Gain largest at the bottom (0.34 -> 0.07 K, scattered; 0.28 -> 0.10 band);
   the earlier "mean-state collapse" was this artefact: 40 % of the mean-state error now, and grid W1 at the
   training points 0.124 -> 0.083 K. The ensemble spread grows (0.11 -> 0.13 K), the single-sample RMSE hardly
   moves (0.188 -> 0.147): the ensemble mean now averages real sampler spread instead of a shared bias.
2. **Model vs baselines.** Band: beats the nearest run at every level below 26 m and in all three rows. Centre
   block (half the grid held out, training on the edge only): beats the nearest run at every level below the
   surface; the error is flat over the block (0.04 to 0.12 K; centre 3 x 3 0.068 K, rim 0.063 K) while the
   nearest run degrades from 0.10 K at the rim to 0.18 K at the centre. Wins at the 28 runs where the state
   changes with the parameters (nearest run 0.10 to 0.39 K), loses by at most 0.07 K at the 21 runs of the flat
   regime (low `c_k`, high `c_eps`), where a copied neighbour is already within 0.05 K. Worst run ck0.504_eps0.2205
   (0.12 K), the warm edge of the block. Top row:
   beats the nearest run at every level but the surface and the bottom, and copying the row below for
   `c_eps` 0.14 to 1.4; loses at the corner and in the flat regime. Scattered: loses to the one-step neighbours
   (0.061 vs 0.037 / 0.020 K) except at 490 to 650 m.
3. **Depth structure.** 0.04 to 0.08 K at every level in the scattered and block splits; the bottom two levels are no longer
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
5. **The sampler's clip is a regulariser, not a range limit.** Exact fill, scattered / top: clip at 1 gives
   0.152 / 0.173 K, at 1.5 0.194 / 0.183, at 3 0.302 / 0.277, although the final samples barely exceed
   |x'| = 1 (0.1 %). The price is a range limit: 12 % of the held-out top row's true bottom values
   lie above mu + 3 sigma and cannot be generated; the corner run is the largest top-row error in every variant.
6. **Inversions.** 10 to 12 % of interfaces with temperature decreasing upward whatever the regime and the
   sampler; the truth is 7.7 % (scattered set), 4.2 % (block), 1.3 % (band), 0.1 % (top row). Not learned; this is
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

Per run (`fs_scattered_3std/`, `fs_band3_3std/`, `fs_block_3std/`, `fs_top_3std/`): `config.json`, `git_hash.txt`, `train_log.csv`,
`samples_final.png`, `samples_levels.png` (true state and three random samples of T and S at three depths for one
hold-out condition, made with `plot_samples.py`), and `eval/` (default sampler: noised fill) with `summary.txt`, `metrics.csv`, `grid_maps.png`
(+ `grid_domain_mean.csv`, `grid_w1.csv`) and `profiles.png` (+ `profiles.csv`); `eval_cleanfill/` = the previous sampler (exact
zeros after every step).
`data/T_distribution_per_level.png`, `data/T_per_level_stats.npz`: per-level T distribution (`tdist.py compute` on the
cluster, `tdist.py plot` locally).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
