# Results (current design, 2026-09-21)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Data = last 20 years of the 100 runs (241 snapshots per run, 30-day output).
Normalisation per vertical level as in DINO-Fusion (`3-std`), with the statistics computed once on all 100 runs
and held fixed (a deliberate, mild leakage of 30 scaling constants); the hold-out split is a training-config
choice, so one data file serves every split. 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling with the
predicted clean state clipped per level at the observed data range of that level (+5 % margin, bounds within
[-1.30, 2.86] in normalised units), 32 samples per hold-out run and per grid point. Code `1a9470b` (training at
`2b542aa`; the scalar-clip rows below were sampled at `5fdcf01`). Runs: `fs_scattered_3std`, `fs_band3_3std`,
`fs_top_3std`.

## Hold-out metrics (mean over the held-out runs, water cells, against the true 20-year time-mean)

| | scattered (10 interior points) | band of three rows, `c_k` 0.126, 0.2, 0.3175 (30 runs) | top row, `c_k` 0.8 (10 runs) |
|---|---|---|---|
| training runs | 90 | 70 | 90 |
| gap the training rows bridge | one step, all four sides | 0.0794 to 0.504, factor 6.3 | extrapolation, nothing above |
| diffusion ensemble mean of 32 / RMSE (K) | 0.197 ± 0.009 | 0.194 ± 0.013 | 0.206 ± 0.028 |
| diffusion single sample / RMSE (K) | 0.213 | 0.212 | 0.227 |
| neighbour average / RMSE (K) | 0.020 | 0.116 | 0.170 |
| nearest training run / RMSE (K) | 0.037 | 0.116 | 0.170 |
| training-set mean / RMSE (K) | 0.155 | 0.197 | 0.470 |
| diffusion / domain-mean bias (K) | -0.011 | -0.010 | -0.032 |
| ensemble spread / true spread (K) | 0.088 / 0.031 | 0.088 / 0.031 | 0.099 / 0.032 |
| T-inversion fraction, generated / truth | 8.9 % / 7.7 % | 10.3 % / 1.3 % | 12.2 % / 0.1 % |
| max salinity error (psu) | 0.003 | 0.003 | 0.003 |
| W1 profile, diffusion (K) | 0.161 | 0.156 | 0.158 |
| W1 profile, neighbour average (K) | 0.036 | 0.095 | 0.142 |
| W1 profile, nearest training run (K) | 0.044 | 0.095 | 0.142 |
| W1 profile, training-set mean (K) | 0.136 | 0.160 | 0.420 |
| grid W1, all / hold-out / training points (K) | 0.158 / 0.162 / 0.158 | 0.152 / 0.154 / 0.151 | 0.152 / 0.151 / 0.152 |
| same models, scalar clip at 1 / RMSE (K) | 0.152 ± 0.007 | 0.135 ± 0.019 | 0.173 ± 0.081 |
| same models, scalar clip at 1 / W1 (K) | 0.127 | 0.111 | 0.143 |

Band of three, by row (RMSE in K): diffusion 0.197 / 0.202 / 0.183 for `c_k` 0.126 / 0.2 / 0.3175, nearest row
0.063 / 0.132 / 0.152, training mean 0.149 / 0.187 / 0.256. Top row by `c_eps`: the model beats copying the row
below on the warm side, `c_eps` <= 0.35 (0.19 to 0.29 K vs 0.22 to 0.35 K, corner included: 0.29 vs 0.35 K),
ties at 0.5556 (0.20 vs 0.18 K) and loses in the flat regime (`c_eps` >= 0.88: 0.19 to 0.20 vs 0.03 to 0.11 K).

## W1 metric

Each state is reduced to its horizontal-mean temperature profile over water cells. Per level, W1 between the
32 generated values and all 241 true snapshots of the window (quantile form; a subsample of 32 would only add
noise); levels combined with thickness weights dz/H from the level midpoints (H = 2080 m). Point predictions
(baselines): W1 = mean |x - T_i|. Rewards a correct spread, does not reward collapse to the mean, ignores
horizontal structure (kept by the RMSE). Code in `wmetrics.py`.

## What the results say

1. **Conditioning is weak under per-level scaling.** The run-to-run signal is 0.7 % of the normalised range at
   the surface, 9 % at 650 m, 25 % at the bottom; the epsilon loss weights every cell equally, so the shared
   structure dominates. Scattered split: RMSE >= mean-state baseline. Under W1 the training grid points score
   like the held-out ones (0.158 vs 0.162 K): the model does not fit the training conditions either.
2. **Model error is split-independent (0.19 to 0.21 K) while the baselines degrade with the gap
   (0.02 to 0.17 K).** The model loses to the nearest run in all three band rows; on the top row it beats copying
   the row below on the warm side (corner included) and loses in the flat regime.
3. **Depth structure.** Best method between 370 and 800 m in the band split (0.10 vs 0.14 K at 650 m) and
   between 260 and 1200 m in the top row; worse than the baselines in the top 250 m (the per-level scale turns
   small normalised noise into 0.1 K of scatter) and at the two bottom levels (skewed, drifting distribution,
   widest clip bounds).
4. **The sampler's clip is a regulariser.** Same trained models, sampling only (scattered / top row): scalar
   clip at |x'| <= 1 gives 0.152 / 0.173 K, at 1.5 it gives 0.194 / 0.183 K, at 3 it gives 0.302 / 0.277 K,
   although the final samples barely exceed |x'| = 1 (0.1 %). Clipping the early, inaccurate clean-state
   estimates keeps the chain on track. A scalar clip at 1 also caps the deep levels of the warm corner (12 % of
   the top row's true bottom values above mu + 3 sigma): a floor of 0.04 K on the row, 0.17 K at the corner run.
   The per-level clip (each channel at its own data range) removes that floor (corner run 0.29 vs 0.40 K,
   top-row bias -0.03 vs -0.07 K) and cuts the salinity error by four, but loosens the regulariser where the
   bounds are wide: bottom-level RMSE 0.34 -> 0.48 K (scattered), 0.28 -> 0.46 K (band), 0.35 -> 0.42 K (top),
   hence 0.197 / 0.194 / 0.206 K overall, worse than the scalar clip on every split. Above 1400 m the two clips
   are within 0.04 K of each other.
5. **Inversions.** The model generates 9 to 12 % of interfaces with temperature decreasing upward whatever the
   regime; the truth goes from 7.7 % (scattered set) to 1.3 % (band) to 0.1 % (top row). Not learned; this is
   the kind of constraint DINO-Fusion imposes at sampling time.
6. **Plumbing.** Salinity within 0.003 psu of 35 without any constraint, land exact, spread 3x the true
   within-window spread.

Previous design (statistics per split, 8 then 32 samples, scalar clip at 1): 0.150 / 0.129 / 0.193 K; fixed
statistics changed the scattered and band numbers within noise and improved the top row from 0.193 to 0.173 K.

## Cost

Extraction 2.5 min on 8 CPU cores (once). Training 17 to 22 min on one A100 per split. Generation of the 32-sample
hold-out and grid sets plus evaluation 14 to 18 min. A split costs under 40 GPU minutes end to end.

## Suggested next steps

- Clip: keep the scalar clip's hold at depth without the corner floor, e.g. per-level bounds from training
  quantiles (tight where the data are narrow) or the scalar clip loosened only on the levels the corner exceeds.
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
