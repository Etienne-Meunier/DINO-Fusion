# Results: first end-to-end run (2026-09-18)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Code at commit `f912e66`, data = last 20 years of the 100 runs (241 snapshots
per run, 30-day output), 90 training runs, 10 interior hold-out runs chosen at random before any statistics.
Normalisation per vertical level as in DINO-Fusion (`3-std`). 20,000 steps, batch 32, EMA, 1000 DDPM steps
at sampling, 8 samples per hold-out condition.

## Hold-out metrics (mean over the 10 hold-out runs, water cells, against the true 20-year time-mean)

| method / metric                                | full_3std |
|------------------------------------------------|-----------|
| diffusion ensemble mean / RMSE (K)             | 0.154 ± 0.018 |
| diffusion single sample / RMSE (K)             | 0.188 ± 0.015 |
| neighbour average / RMSE (K)                   | 0.020 ± 0.011 |
| nearest training run / RMSE (K)                | 0.037 ± 0.027 |
| training-set mean / RMSE (K)                   | 0.155 ± 0.030 |
| diffusion / domain-mean bias (K)               | -0.028 |
| diffusion / ensemble spread (K)                | 0.105 |
| truth / spread within window (K)               | 0.031 |
| diffusion / T-inversion fraction               | 9.5 % |
| truth / T-inversion fraction                   | 7.7 % |
| diffusion / max salinity error (psu)           | 0.010 |
| grid map, domain-mean T RMSE, hold-out (K)     | 0.030 |
| grid map, domain-mean T RMSE, all runs (K)     | 0.049 |

## What it shows

1. **The pipeline runs end to end**: extraction, training (22 min on one A100), conditional generation,
   evaluation with baselines and figures, all as one SLURM dependency chain under one GPU hour.
2. **Conditioning is not yet effective.** The ensemble-mean RMSE (0.154 K) equals the mean-training-state
   baseline (0.155 K): the model reproduces the shared structure of the ensemble and adds noise. The generated
   map of domain-mean temperature over the parameter grid shows only a faint response in the warm high-`c_k`,
   low-`c_eps` corner. The mechanism: under per-level scaling the run-to-run signal is 0.7 % of the normalised
   range at the surface, 9 % at 650 m and 25 % at the bottom, and the epsilon-prediction loss weights every
   cell equally, so the shared structure dominates training. Accordingly the model beats the mean-state
   baseline only between about 400 and 1400 m; it is worse than it in the top 250 m, where the signal is
   below one percent of the range, and at the two bottom levels, where the training distribution is
   broadened by the residual drift and its extremes are clipped.
3. **Spread.** Single samples are 0.188 K from the truth and the ensemble spread (0.105 K) is three times the
   true within-window spread (0.031 K).
4. **Plumbing checks pass.** Salinity comes back within 0.01 psu of 35 without any constraint, land is exact,
   and the fraction of interfaces with temperature decreasing upward (9.5 %) is close to the truth's (7.7 %).

Data range under this normalisation: with k = 3, 99.4 % of the training temperature values lie inside the
sampler's clip range [-1, 1]; the exceptions are the cold southern boundary between 650 and 1400 m and the
bottom two levels, whose extremes are clipped at sampling. Salinity is exactly 0 everywhere.

## Samples per condition: 8 versus 32 (same model, same hold-out runs)

| | 8 samples | 32 samples |
|---|---|---|
| diffusion ensemble mean / RMSE (K) | 0.154 ± 0.018 | 0.150 ± 0.007 |
| diffusion single sample / RMSE (K) | 0.188 | 0.186 |
| diffusion / ensemble spread (K)    | 0.105 | 0.110 |

More samples remove only the Monte-Carlo noise of the ensemble mean (the predicted 0.004 K), not the bias
or the collapse toward the mean state. Generating 320 samples took 86 s on one A100, so 32 samples per
condition is now the default (`n_samples` in the config, `submit.sh` and `campaign.sh`); the grid map keeps
4 samples per condition.

## Hold-out splits: scattered, band of three rows, top row (2026-09-21)

Same model family and settings (per-level normalisation, 20,000 steps, EMA, 32 samples per hold-out run),
three ways of choosing the hold-out runs. Statistics and training runs differ per split, so each split has
its own dataset file (`data/veros_acc_TS_<split>.npz`) and run (`runs/<split>_3std`).

| | scattered (10 interior points) | band of three rows, `c_k` 0.126, 0.2, 0.3175 (30 runs) | top row, `c_k` 0.8 (10 runs) |
|---|---|---|---|
| training runs | 90 | 70 | 90 |
| gap the training rows bridge | one step, all four sides | 0.0794 to 0.504, factor 6.3 | extrapolation, nothing above |
| diffusion ensemble mean / RMSE (K) | 0.150 | 0.129 | 0.193 |
| diffusion single sample / RMSE (K) | 0.186 | 0.175 | 0.232 |
| log-`c_k` interpolation / RMSE (K) | n/a (one step) | 0.083 | 0.170 (one-sided = row below) |
| neighbour average / RMSE (K) | 0.020 | 0.116 | 0.170 |
| nearest training run / RMSE (K) | 0.037 | 0.116 | 0.170 |
| training-set mean / RMSE (K) | 0.155 | 0.197 | 0.470 |
| diffusion / domain-mean bias (K) | -0.024 | -0.023 | -0.083 |
| ensemble spread / true spread (K) | 0.110 / 0.031 | 0.113 / 0.031 | 0.114 / 0.032 |
| T-inversion fraction, generated / truth | 9.6 % / 7.7 % | 11.0 % / 1.3 % | 12.0 % / 0.1 % |
| domain-mean T map RMSE, hold-out (K) | 0.030 | 0.045 | 0.101 |

Band of three, by row (RMSE in K, mean over the 10 runs of the row):

| row | diffusion mean | log-`c_k` interpolation | nearest row | training mean |
|---|---|---|---|---|
| `c_k` 0.126 | 0.133 | 0.066 | 0.063 | 0.149 |
| `c_k` 0.2 (middle) | 0.134 | 0.098 | 0.132 | 0.187 |
| `c_k` 0.3175 | 0.120 | 0.086 | 0.152 | 0.256 |

Top row, by `c_eps` (RMSE in K): the model beats "copy the row below" where there is a trend to
extrapolate (`c_eps` 0.22 to 0.88: 0.10 to 0.20 versus 0.11 to 0.27), loses at the extreme corner
(`c_eps` 0.0875: 0.48 versus 0.35) and in the flat regime (`c_eps` >= 1.4: 0.14 to 0.16 versus 0.03 to 0.08).

What the three splits say together:

1. **The model's error is nearly independent of the split** (0.13 to 0.19 K) while the baselines' error
   grows with the gap they must bridge (0.02 to 0.17 K). The model therefore catches up with the baselines
   as the split gets harder: it beats the nearest-row baseline in the two upper rows of the band and over the
   trending part of the top row, but never beats log-`c_k` interpolation over an interior gap.
2. **Where the parameter signal lives, the model is competitive or best.** In the band split it is the best
   method between about 650 and 800 m (0.08 K against 0.11 K for interpolation and 0.14 K for the nearest
   row). Its floor comes from the top 250 m, where the per-level scale turns small normalised noise into
   0.1 K of scatter, and from the two bottom levels, where the training distribution is skewed and drifting
   and the sampler clips the extremes.
3. **Extrapolation is limited by the clip range before the model.** Under the top split's own training
   statistics, 17 % of the true bottom-level values of the held-out row and 7 % at 1666 m lie beyond
   [-1, 1], so they cannot be generated; the generated corner is 0.2 K too cold and the bottom-level RMSE is
   0.41 K against 0.22 K for the row below. Under the band split the same fraction is below 2 %.
4. **A physical miss that grows with `c_k`.** The model generates 10 to 12 % of interfaces with temperature
   decreasing upward whatever the regime, the rate of the low-`c_k` rows that dominate the grid, whereas the
   truth goes from 7.7 % (scattered set) to 1.3 % (band) to 0.1 % (top row): the strongly eddying states are
   stratified everywhere and the model has not learned that dependence. This is exactly the kind of
   constraint DINO-Fusion imposes at sampling time (isotonic projection of the density profile), and the
   natural next lever here.

## Wasserstein-1 on stratification profiles (2026-09-21)

Each state is reduced to its horizontal-mean temperature profile over water cells. Per level, W1 between the
32 generated values and all 241 true snapshots of the window (quantile form, unequal sizes are fine; a
subsample of 32 would only add noise); levels combined with thickness weights dz/H from the level midpoints
(H = 2080 m). Point predictions (baselines): W1 = mean |x - T_i|. Floor: 32 random true snapshots vs all 241,
mean of 100 draws. Drift scale: first vs second half of the window. Rewards a correct spread, does not reward
collapse to the mean, ignores horizontal structure (kept by the RMSE).

| thickness-weighted W1 (K) | scattered | band of three rows | top row |
|---|---|---|---|
| diffusion, 32 samples | 0.125 | 0.109 | 0.161 |
| interpolation in log ck | 0.036 | 0.062 | 0.142 |
| neighbour average | 0.036 | 0.095 | 0.142 |
| nearest training run | 0.044 | 0.095 | 0.142 |
| training-set mean | 0.136 | 0.160 | 0.420 |
| floor (32 of 241 true) | 0.007 | 0.007 | 0.008 |
| drift (half vs half) | 0.066 | 0.067 | 0.073 |
| grid map: mean W1, all / hold-out / training points | 0.123 / 0.125 / 0.123 | 0.115 / 0.107 / 0.119 | 0.120 / 0.160 / 0.116 |

Same ranking as the RMSE. Two additions: training grid points score like held-out ones (mean-state
collapse), and between 650 and 1200 m the samples beat every baseline under W1 in the band and top splits.
Figures: `<run>/eval/grid_w1.png` (+ `.csv`) and `<run>/eval/w1_profile.png` (+ `.csv`); code in `wmetrics.py`.

## Cost

Extraction 2.5 min on 8 CPU cores. Training 22 min on one A100. Generation of 80 hold-out and 400 grid
samples plus evaluation 3.5 min. Whole chain under one GPU hour on the dev QoS.

## Suggested next steps

- Strengthen the conditioning: classifier-free guidance (`cond_drop_prob` 0.1, guidance 2 to 3), longer
  training or a larger conditioning MLP, and the year as a third condition.
- Evaluate against snapshots as well as the time mean (nearest-snapshot RMSE, spread-skill), since the
  model is trained on snapshots that drift by up to 0.45 K at the bottom over the 20-year window.
- More samples per condition (32 instead of 8) for the ensemble mean.
- Extrapolation split: hold out the whole `c_k` = 0.8 row to test the regime corner.

## Figures

`full_3std/`: `grid_domain_mean.png` (+ `.csv`), `sections_holdout.png`, `rmse_profile.png`, `samples_final.png`,
`samples_levels.png` (true state and three random samples of T and S at three depths for one hold-out condition,
made with `plot_samples.py`). Raw metrics in `metrics.csv`, training curves in `train_log.csv`.
`full_3std/eval/`, `band3_3std/eval/`, `top_3std/eval/`: the three-split evaluations (32 samples per run and per grid point, RMSE and W1 figures); `full_3std/eval_n32/` is the earlier 32-sample rerun without W1.
`band3_3std/samples_levels.png` (condition ck0.2_eps2.222, middle held-out row) and `top_3std/samples_levels.png`
(condition ck0.8_eps0.5556, extrapolation): the same visual check for the two other splits.
`report/report.tex`, `report/report.pdf`: the short report on the scattered split (compile with `tectonic report.tex`).
