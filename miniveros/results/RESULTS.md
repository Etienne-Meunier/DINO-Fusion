# Results: first end-to-end run (2026-09-18)

Conditional DDPM emulator of the Veros ACC temperature and salinity state as a function of the EKE-closure
coefficients `c_k` and `c_eps`. Code at commit `f912e66`, data = last 20 years of the 100 runs (241 snapshots
per run, 30-day output), 90 training runs, 10 interior hold-out runs chosen at random before any statistics.
Two identical trainings differing only in the normalisation: per-cell anomaly (`anomaly`) and DINO-Fusion's
per-level (`3-std`). 20,000 steps, batch 32, EMA, 1000 DDPM steps at sampling, 8 samples per hold-out condition.

## Hold-out metrics (mean over the 10 hold-out runs, water cells, against the true 20-year time-mean)

| method / metric                     |   full_anomaly |      full_3std |
|-------------------------------------|----------------|----------------|
| diffusion ensemble mean / RMSE (K)  |         0.0302 |         0.1537 |
| diffusion single sample / RMSE (K)  |         0.0716 |         0.1880 |
| neighbour average / RMSE (K)        |         0.0202 |         0.0202 |
| nearest training run / RMSE (K)     |         0.0370 |         0.0370 |
| training-set mean / RMSE (K)        |         0.1553 |         0.1553 |
| diffusion / domain-mean bias (K)    |        -0.0002 |        -0.0279 |
| diffusion / ensemble spread (K)     |         0.0485 |         0.1047 |
| truth / spread within window (K)    |         0.0311 |         0.0311 |
| diffusion / T-inversion fraction    |          8.4 % |          9.5 % |
| truth / T-inversion fraction        |          7.7 % |          7.7 % |
| diffusion / max salinity error (psu)|         0.0283 |         0.0098 |
| grid map, domain-mean T RMSE, hold-out (K) | 0.0121 |         0.0302 |
| grid map, domain-mean T RMSE, all runs (K) | 0.0448 |         0.0490 |

## What it shows

1. **Normalisation decides whether conditioning works.** With per-level normalisation the conditional model
   reproduces the training mean (RMSE 0.154 K versus 0.155 K for the mean state) and the generated parameter
   map is flat and noisy. With per-cell anomaly normalisation the run-to-run signal is order one and the model
   follows the parameters: RMSE 0.030 K, bias below a millikelvin, and the generated 10 x 10 map reproduces
   the true response including the warm high-`c_k` / low-`c_eps` corner.
2. **The diffusion emulator beats the nearest training run but not the neighbour average yet.** Averaging
   the four axis neighbours of a hold-out point on the log-parameter grid gives 0.020 K; the 8-sample
   ensemble mean gives 0.030 K. Single samples (0.072 K) are noisier than the true within-window
   variability (0.031 K): the generator is overdispersed by about 1.6x.
3. **Where the error is.** The RMSE profile puts the diffusion error above the baselines only below about
   1200 m, with a warm bias of 0.1 to 0.2 K in the deepest layers of several hold-out runs. This is also where
   about 2 percent of training values fall outside the clipping range at three standard deviations in anomaly
   mode, so clipping is the first suspect. The cold bias at the extreme corner run (`c_k` 0.8, `c_eps` 0.0875)
   is the regime change at the edge of the training set.
4. **Plumbing checks.** Salinity comes back within 0.03 psu of 35 without any constraint, land is exact,
   and the fraction of interfaces with temperature decreasing upward (8.4 %) is close to the truth's (7.7 %).

## Cost

Extraction 2.5 min on 8 CPU cores. Training 22 min per run on one A100. Generation of 80 hold-out and
400 grid samples plus evaluation 3.5 min per run. Whole chain under one GPU hour on the dev QoS.

## Suggested next steps

- Anomaly mode without clipping: raise the scale factor from 3 to 4 standard deviations, or disable
  `clip_sample`, to remove the deep bias.
- More samples per condition (32 instead of 8): the ensemble mean is what competes with the baselines.
- Extrapolation split: hold out the whole `c_k` = 0.8 row to test the regime corner.
- Classifier-free guidance (`cond_drop_prob` 0.1, guidance 2 to 3) to sharpen a small conditioning signal.
- Year as a third condition to absorb the residual drift inside the 20-year window.

## Figures

`full_anomaly/`: `grid_domain_mean.png`, `sections_holdout.png`, `rmse_profile.png`, `samples_final.png`.
`full_3std/`: same set. Raw metrics in `metrics.csv`, training curves in `train_log.csv`.
`full_anomaly/samples_levels.png`: true state and three random samples of T and S at three depths for one
hold-out condition (made with `plot_samples.py` from the hold-out samples and the raw runs).
`report/report.tex`, `report/report.pdf`: the short report (compile with `tectonic report.tex`).
