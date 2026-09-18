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
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
