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

- y(z) = mu + sum_{k=1..20} a_k(z) v_k, where mu and v_k are the mean and the leading EOFs of the training states.
  Projecting the held-out truth on this basis leaves 0.01 to 0.02 K in interpolation and 0.03 to 0.08 K in
  extrapolation: a floor for this design.
- a(z) = MLP 2 -> 64 -> 64 -> 20 (scikit-learn `MLPRegressor`, L-BFGS, L2 weight alpha). The targets are
  a_k / std(a_1), so the loss is the physical MSE of the projected field up to a constant.
- Ensemble of 5 initialisations; the prediction is their mean.
- (activation, alpha) in {ReLU, tanh} x {1e-4, 1e-3, 1e-2, 0.1, 1} is chosen by 5-fold cross-validation on the
  training runs of each split. For leave-one-out it is chosen once, on all runs. The choice is ReLU with alpha
  1e-4 to 1e-2 everywhere except centre 7 x 7 (tanh, 1e-3) and four runs (alpha 1, from a meaningless 4-run CV).
  The CV scores are flat for alpha <= 1e-2 (`selected.csv`).
- The network is fitted on the 100 states, not on snapshots, because
  sum_t ||f(z_r) - x_{r,t}||^2 = n_t ||f(z_r) - mean_t x_{r,t}||^2 + const.
- Cost: a fit takes seconds on one core, and a prediction microseconds. The whole evaluation below takes about
  25 min on 8 cores, most of it in the hyper-parameter search.

## Baselines

- Training mean.
- Nearest run in z.
- Thin-plate RBF on every cell.
- GP per PCA coefficient on z (Matern 5/2, one length scale per input, white noise).
- **GP on s**: a GP on s = w . z alone, where w comes from a linear fit of the first PCA coefficient on z. This is
  the one-dimensional collapse of the two parameters, roughly log c_k - 1.05 log c_eps.

## Results

The splits are eight hold-out splits of the 10 x 10 grid, plus leave-one-out (LOO) over the 100 runs. The metric is the RMSE over the water cells against the last-year mean, averaged over the
held-out runs. Full tables are in `results/summary.md`.

| split | held out | network | GP on s | GP | RBF | nearest | mean state |
|---|---|---|---|---|---|---|---|
| leave-one-out | 100 | **0.024** | 0.028 | 0.032 | 0.034 | 0.085 | 0.221 |
| scattered | 10 | **0.016** | 0.019 | 0.022 | 0.021 | 0.038 | 0.161 |
| band of 3 rows | 30 | **0.037** | 0.046 | 0.051 | 0.057 | 0.118 | 0.204 |
| centre 5 x 5 | 25 | **0.026** | 0.029 | 0.037 | 0.037 | 0.080 | 0.162 |
| centre 7 x 7 | 49 | **0.035** | 0.036 | 0.070 | 0.068 | 0.116 | 0.203 |
| top row | 10 | **0.059** | 0.079 | 0.083 | 0.084 | 0.176 | 0.482 |
| outer 51 | 51 | **0.068** | 0.075 | 0.088 | 0.093 | 0.134 | 0.265 |
| outer 75 | 75 | **0.123** | 0.125 | 0.164 | 0.138 | 0.159 | 0.226 |
| four runs | 96 | 0.139 | **0.120** | 0.234 | 0.146 | 0.181 | 0.260 |

- **The network is best on 8 of the 9 evaluations.** In interpolation it is 0.016 to 0.037 K, half or less of the
  nearest run. Its biggest lead is on the top row (extrapolation): 0.059 vs 0.079 K for GP on s, and at the corner
  run 0.15 vs 0.37 K.
- **Its margin over GP on s is small inside the grid.** Centre 7 x 7: 0.035 vs 0.036 K, with 26 run wins of 49.
  LOO: 61 wins of 100. The parameters act almost only through s, and the network gains mostly at the grid edges.
- **The error sits in the upper 500 m.** In interpolation it is 0.02 to 0.05 K there and 0.01 to 0.02 K below
  1000 m (`profiles.png`). Within the grid, the largest LOO errors are at the two high-c_k corners
  (`loo_maps.png`).
- **Large gaps are where it loses.** On outer 75 (25 training runs), the network's median is 0.057 K vs 0.038 K for
  GP on s. On four runs, the network gets 0.139 K vs 0.120 K for GP on s, and the nearest run has the best median
  (0.064 K) in the flat regime.
- **The warm corner (c_k 0.8, c_eps 0.0875) is the hardest state** (`examples.png`).
  - One row from the training set (top row), the network reproduces its warming: correlation 0.99, amplitude 1.01,
    0.15 K.
  - Three cells away (outer 75), it keeps a weak version of the pattern (correlation 0.56) with 30 % of the
    amplitude: a cold bias of -0.71 K and 1.06 K RMSE.
  - Every method is above 1 K there (1.06 to 1.36 K). With all neighbours present (LOO), the error is 0.16 K.
- **The seed-ensemble spread is not an uncertainty.** It is 0.005 to 0.016 K in interpolation and 0.013 to 0.024 K
  in extrapolation, while the error grows from 0.016 to 0.139 K.

Figures in `results/`:
- `splits.png`: the table above.
- `loo_maps.png`: LOO error over the grid.
- `profiles.png`: RMSE by depth for every split.
- `examples.png`: zonal-mean sections of the anomaly (truth, network, error) for the centre run of the 7 x 7 block,
  and for the warm corner held out with the top row and with the outer 75.

## Usage (from the miniveros directory)

```bash
python -m det.lastyear --raw-dir ../full_state/mnk965ig   # data/veros_acc_lastyear.npz, seconds
python -m det.run                                         # results/metrics.csv, levels.csv, selected.csv, summary.md, examples.npz
python -m det.figures                                     # results/*.png
```

Needs numpy, scipy, scikit-learn, joblib and matplotlib (all in the conda base environment); no torch.
