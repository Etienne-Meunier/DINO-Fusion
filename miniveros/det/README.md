# det: deterministic emulator of the last-year state

`net(c_k, c_eps) -> T`: the mean temperature of the last model year of a Veros ACC run, from the two EKE-closure
coefficients. Independent of the rest of the parent folder: no sampling, no GPU. The data are 100 states
(7.6 MB), and everything runs on a laptop CPU. Report: `report/report.pdf`.

## Data

- Target: y_r = mean of the last 12 monthly outputs (360 days) of run r; temperature on the 17,970 water cells
  (15 levels x 42 x 30, minus the 930 ridge cells). Salinity is 35 everywhere and is dropped.
- Why the last year: the runs still drift. Over the last 20 years, the cell RMS change from the first to the last
  year is 0.16 K, and the last-year mean differs from the 20-year mean by 0.08 K RMS.
- Input: z = (log c_k, log c_eps), standardised over the 10 x 10 design.

## Network

- y(z) = mu + sum_{k=1..K} a_k(z) v_k, where mu and v_k are the mean and the leading EOFs of the n training states,
  and K = min(20, n - 1).
- a(z) = MLP 2 -> 64 -> 64 -> K (scikit-learn `MLPRegressor`, L-BFGS, L2 weight alpha). The targets are
  a_k / std(a_1), so the data term is the physical MSE of the projected field up to a constant.
- Ensemble of 5 initialisations; the prediction is their mean. The evaluation is repeated with 4 such sets
  (`run.SEEDS`), and the results average them.
- Hyper-parameters: (activation, alpha) in {ReLU, tanh} x {1e-4, 1e-3, 1e-2, 0.1, 1}, chosen on the training runs
  of each split.
  - The search is 5-fold cross-validation repeated 3 times.
  - The one-standard-error rule then takes the most regularised configuration within one standard error of the
    best (`methods.select`).
  - The choice is ReLU with alpha 1e-2 everywhere except outer 75 (alpha 0.1) and four runs (alpha 1).
  - For leave-one-out the choice is made once, on all runs.
- Cost: a fit takes seconds on one core. The whole evaluation takes about 1 h on 8 cores.

## Baselines

None of them is tuned by cross-validation.
- Training mean.
- Nearest run in z.
- Thin-plate RBF on every cell (exact interpolation).
- GP per PCA coefficient on z (Matern 5/2, one length scale per input, white noise up to the full variance).
- **GP on s**: the same GP on s = w . z alone, where w comes from a linear fit of the first PCA coefficient on z.
  This is the one-dimensional collapse of the two parameters, roughly log c_k - 1.08 log c_eps.

## Results

There are eight hold-out splits of the 10 x 10 grid plus leave-one-out (LOO). The metric is the RMSE over the water
cells against the last-year mean, averaged over the held-out runs; for the network it is the mean over 4
initialisation sets, with their range in brackets. Full tables are in `results/summary.md`; the configuration
sensitivity is in `results/sensitivity.csv`.

| split | held out | network | GP on s | GP on z | RBF | nearest | mean state |
|---|---|---|---|---|---|---|---|
| leave-one-out | 100 | **0.024** (0.024-0.024) | 0.028 | 0.032 | 0.034 | 0.085 | 0.221 |
| scattered | 10 | **0.017** (0.017-0.017) | 0.019 | 0.022 | 0.021 | 0.038 | 0.161 |
| band of 3 rows | 30 | **0.038** (0.037-0.039) | 0.045 | 0.051 | 0.057 | 0.118 | 0.204 |
| centre 5 x 5 | 25 | **0.025** (0.024-0.025) | 0.029 | 0.035 | 0.037 | 0.080 | 0.162 |
| centre 7 x 7 | 49 | **0.029** (0.028-0.030) | 0.035 | 0.069 | 0.068 | 0.116 | 0.203 |
| top row | 10 | **0.049** (0.047-0.050) | 0.079 | 0.083 | 0.084 | 0.176 | 0.482 |
| outer 51 | 51 | **0.067** (0.066-0.067) | 0.074 | 0.081 | 0.093 | 0.134 | 0.265 |
| outer 75 | 75 | **0.114** (0.113-0.114) | 0.125 | 0.131 | 0.138 | 0.159 | 0.226 |
| four runs (failure case) | 96 | 0.147 (0.134-0.169) | **0.120** | 0.235 | 0.146 | 0.181 | 0.260 |

- **The network beats the GP on s in every interpolation evaluation.** Per-run paired Wilcoxon p ranges from 0.037
  to < 0.001, the margin is 0.003 to 0.007 K, and its errors are a third of the nearest run's.
- **In extrapolation it is better on outer 51 and outer 75** (p = 0.003 and 0.001). On the top row the gain is
  mostly the warm-corner run (0.16 vs 0.37 K); without it the network gets 0.037 vs 0.046 K, and the test is not
  significant (p = 0.19).
- **Regularisation matters, initialisation does not.**
  - Four initialisation sets change a split's mean by at most 0.002 K.
  - Among the configurations cross-validation cannot tell apart, the test error spans up to 0.035 K, and on
    centre 7 x 7, outer 51 and outer 75 only half of them beat the GP on s.
  - The rule's choice, the most regularised, is the best of that set on those three splits and on centre 5 x 5.
- **The error sits in the upper 500 m** (`profiles.png`). In interpolation it is 0.014 to 0.055 K there and 0.003
  to 0.016 K below 1000 m.
- **The warm corner (c_k 0.8, c_eps 0.0875) is the hardest state** (`examples.png`).
  - One row from the training set (top row): correlation 0.99, amplitude 0.99, 0.16 K.
  - Three cells away (outer 75): correlation 0.62, a third of the amplitude, 1.06 K. Every method is between 1.06
    and 1.45 K there.
- **Part of the extrapolation error is the basis.** Projecting the held-out truth on the training EOFs leaves 0.01
  to 0.02 K in interpolation and 0.03 to 0.08 K in extrapolation, mostly at the corner run.
- **The seed-ensemble spread is not an uncertainty.** It is 0.003 to 0.010 K while the error reaches 0.114 K.
- **Four training runs: a failure case.** Cross-validation cannot choose there, and the rule picks alpha = 1, which
  makes the network too flat (45 % of the anomaly at the centre run). Over the ten configurations the network
  ranges from 0.10 to 0.19 K, against 0.120 K for GP on s.

Figures in `results/`:
- `data.png`: the target over the grid and against s.
- `splits.png`: the table above.
- `profiles.png`: RMSE by depth for every split.
- `examples.png`: zonal-mean sections of the anomaly relative to the training mean (truth, network, error). The
  centre run of the 7 x 7 block is held out with the block and in the four-run split; the warm corner with the top
  row, with the outer 75 and in the four-run split.

## Usage (from the miniveros directory)

```bash
python -m det.lastyear --raw-dir ../full_state/mnk965ig   # data/veros_acc_lastyear.npz, seconds
python -m det.run                                         # results/metrics.csv, levels.csv, selected.csv, summary.md, examples.npz (~1 h)
python -m det.sensitivity                                 # results/sensitivity.csv (test RMSE of every configuration, ~4 min)
python -m det.figures                                     # results/*.png
```

Needs numpy, scipy, scikit-learn, joblib and matplotlib (all in the conda base environment); no torch.
