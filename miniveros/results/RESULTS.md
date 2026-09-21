# Results (current design, 2026-09-21)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped at 3 sigma, 32 samples per hold-out run and per grid point. Code `5fdcf01`
(training at `2b542aa`). Runs: `fs_scattered_3std`, `fs_band3_3std`, `fs_top_3std`.

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

## Cost

Extraction 2.5 min on 8 CPU cores (once). Training 17 to 22 min on one A100 per split. Generation of the 32-sample
hold-out and grid sets plus evaluation 14 to 18 min. A split costs under 40 GPU minutes end to end.

## Suggested next steps

- Stratification constraint at sampling time (DINO-Fusion's isotonic projection, on T since S is constant).
- Stronger conditioning: classifier-free guidance; the year as a third condition (drift of 0.45 K at the bottom
  inside the window).
- Evaluate against snapshots as well as the time mean (nearest-snapshot RMSE, spread-skill).

## Figures

Per run (`fs_scattered_3std/`, `fs_band3_3std/`, `fs_top_3std/`): `config.json`, `git_hash.txt`, `train_log.csv`,
`samples_final.png`, `samples_levels.png` (true state and three random samples of T and S at three depths for one
hold-out condition, made with `plot_samples.py`), and `eval/` with `summary.txt`, `metrics.csv`, `grid_maps.png`
(+ `grid_domain_mean.csv`, `grid_w1.csv`) and `profiles.png` (+ `profiles.csv`).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
