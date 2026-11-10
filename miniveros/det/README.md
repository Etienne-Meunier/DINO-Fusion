# det: deterministic emulator of the last-year state

`net(c_k, c_eps) -> T`: the mean temperature of the last model year of a Veros ACC run, from the two constants of
the TKE vertical-mixing closure (`settings.c_k`, `settings.c_eps`; setup default 0.1, 0.7). Independent of the rest of the parent folder: no sampling, no GPU. The data are 100 states
(7.6 MB), and everything runs on a laptop CPU. Report: `report/report.pdf`.

## Setup

- Model: mini_veros, a JAX port of Veros by E. Meunier, ACC setup `acc/full` (the settings of Veros's
  `setups/acc/acc.py`): 2° cells, 30 x 42 x 15, about 42S to 42N, 2080 m, flat bottom. A 4°
  meridional wall runs from 20S to the northern boundary; south of 20S the domain is a re-entrant channel.
- Steady forcing: zonal wind over the channel (up to 0.1 N/m2) and north of 10N (up to 0.2 N/m2); surface
  temperature restored over 30 days. Initial T linear in depth from 15 C; salinity 35; nonlinear equation of state
  (type 3), density decreasing with T over the data range.
- Varied: the TKE constants, K_M = c_k l sqrt(e) and dissipation c_eps e^1.5 / l, with l = sqrt(2e)/N. Then K_M
  and the dissipation are unchanged under (c_k, c_eps, e) -> (lambda c_k, lambda c_eps, e/lambda), so the state
  depends mainly on c_k/c_eps. The EKE closure that sets K_gm is held at its setup values (0.4, 0.5).
- Check on the raw runs (last output, median over levels 3-11): along c_k/c_eps = 1/7, tke falls about 45-fold
  while c_k * tke stays at 1.7-1.9e-7 m2/s2, and eke stays at 7.5e-4 m2/s2. The sweep's own records (MiniVeros-Diffusion repository) also describe c_k and
  c_eps as the TKE closure's constants.

## Data

- Target: y_r = mean of the last 12 outputs of run r (instantaneous snapshots, one every 30 days); temperature on the 17,970 water cells
  (15 levels x 42 x 30, minus the 930 cells of the wall). Salinity is 35 everywhere and is dropped.
- Why the last year: the runs still drift. Over the last 20 years, the cell RMS change from the first to the last
  year is 0.16 K, and the last-year mean differs from the 20-year mean by 0.08 K RMS. The target is the year-100
  state of this protocol, not an equilibrium.
- Absolute temperatures are in degrees C; differences and errors in K.
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
  - The standard error is over runs, not folds: the std over the runs of the best configuration's per-run CV RMSE,
    divided by sqrt(n).
  - `det.run` also records the minimum-CV-RMSE choice (method `mlp_min`). It is ReLU with alpha 1e-3 or 1e-4,
    except outer 75 (1e-2) and four runs (1).
  - The rule is not blind to the test sets: it was adopted after an independent review had computed test errors
    of other configurations.
  - For leave-one-out the choice is made once, on all runs.
- Reproducibility: importing `det.methods` limits BLAS to one thread, in the main process and in the joblib
  workers.
  - A refit is then bit-identical; `det.sensitivity` checks this against `metrics.csv`.
  - With several threads, L-BFGS ends elsewhere. Against an earlier 8-thread run, split means moved by up to
    0.0003 K and single runs by up to 0.008 K.
  - Library versions used: scikit-learn 1.7.2, NumPy 2.2.5, OpenBLAS 0.3.30.
- Cost: a fit takes seconds on one core. On 8 cores the evaluation takes 46 min, and the sensitivity check 28 min.

## Baselines

None of them is tuned by cross-validation.
- Training mean.
- Nearest run in z. Interior runs often have 2 to 4 equidistant neighbours on the standardised grid; the first in
  storage order is taken, so this baseline depends on that order.
- Thin-plate RBF on every cell (exact interpolation).
- GP per PCA coefficient on z (Matern 5/2, one length scale per input, white noise up to the full variance).
- **GP on s**: the same GP on s = w . z alone, where w comes from a linear fit of the first PCA coefficient on z.
  This is the one-dimensional collapse of the two parameters, roughly log c_k - 1.08 log c_eps (empirical; the
  closure's invariance gives exponent 1).

## Results

There are eight hold-out splits of the 10 x 10 grid plus leave-one-out (LOO). The metric is the RMSE over the water
cells (unweighted) against the last-year mean, averaged over the held-out runs. For the network it is the mean over 4
initialisation sets, followed by the range of that mean across them.

Full tables are in `results/summary.md` (including both selection rules against the GP on s, with Holm-corrected
p over the eight main evaluations). The configuration sensitivity is in `results/sensitivity.csv`. The report's
tables are `results/table_*.tex`, written by `det.tables`.

| split | held out | network | range over initialisation sets | GP on s | GP on z | RBF | nearest | mean state |
|---|---|---|---|---|---|---|---|---|
| scattered | 10 | **0.017** | 0.0008 | 0.019 | 0.022 | 0.021 | 0.038 | 0.161 |
| band of 3 rows | 30 | **0.038** | 0.0010 | 0.045 | 0.051 | 0.057 | 0.118 | 0.204 |
| centre 5 x 5 | 25 | **0.025** | 0.0004 | 0.029 | 0.035 | 0.037 | 0.080 | 0.162 |
| centre 7 x 7 | 49 | **0.029** | 0.0006 | 0.035 | 0.069 | 0.068 | 0.116 | 0.203 |
| leave-one-out | 100 | **0.024** | 0.0004 | 0.028 | 0.032 | 0.034 | 0.085 | 0.221 |
| top row | 10 | **0.049** | 0.0028 | 0.079 | 0.083 | 0.084 | 0.176 | 0.482 |
| outer 51 | 51 | **0.067** | 0.0005 | 0.074 | 0.081 | 0.093 | 0.134 | 0.265 |
| outer 75 | 75 | **0.114** | 0.0009 | 0.125 | 0.131 | 0.138 | 0.159 | 0.226 |
| four runs (failure case) | 96 | 0.147 | 0.0345 | **0.120** | 0.235 | 0.146 | 0.181 | 0.260 |

Items marked † come from an independent review with scripts outside this repository and are not yet reproduced
here.

- **Interpolation: the network has the lowest mean error of the methods tested.**
  - Its margin over the GP on s is 0.0025 to 0.0075 K (13 to 17 %), on all five evaluations.
  - Per-run tests with a Holm correction give p <= 0.044 on band, both centre blocks and LOO, and p = 0.074 on
    scattered. They treat runs on one c_k/c_eps diagonal as independent, and those are near-replicates, so the
    p-values are optimistic. With diagonals as units no evaluation is significant (†).
  - Linear interpolation of the fields along s appears to match the network in interpolation and to beat the GP
    on s (†). Most of the interpolation accuracy is the collapse onto s.
  - Its mean error is 0.25 to 0.44 times the nearest run's, with the tie order above; larger with tied neighbours
    averaged (†).
- **Extrapolation: the network's mean error is lower on all three splits.**
  - Per run it is significant on outer 51 and outer 75 (Holm p = 0.013 and 0.004), not with diagonals as units
    (†). On outer 75 only with the one-SE choice (alpha 0.1).
  - On outer 75 nearly all of the margin comes from runs above the warm end of the training s-range, where the
    network continues the steep slope and the GP flattens (†). On outer 51 and the top row it comes mostly from
    the second direction s_perp (†).
  - On the top row the gain is mostly the warm-corner run (0.16 vs 0.37 K), and the test is not significant
    (p = 0.19). Without that run, the network gets 0.037 vs 0.046 K.
- **The margin over the GP on s depends on the selection rule.**
  - With the minimum CV RMSE, the network is still lower on 7 of 8 evaluations.
  - It ties on centre 7 x 7 (+0.0010 K, p = 0.75) and outer 75 (-0.0032 K, p = 0.59).
  - Per run, only LOO stays significant after the Holm correction.
  - Under both rules, it is below the GP on z, the RBF and the nearest run on the eight main evaluations.
  - With other CV partitions the one-SE choice is unchanged on all eight splits; the minimum changes on four.
- **Initialisation matters little, the configuration more.**
  - Four initialisation sets move a split's mean by 0.0004 to 0.0010 K, and by 0.0028 K on the top row.
  - Among the 5 to 6 configurations that cross-validation cannot tell apart, the test error spans up to 0.03 K
    (outer 75).
  - On centre 7 x 7 only 1 of 6 beats the GP on s.
  - The rule's choice is the best of that set on five of the seven splits (on centre 5 x 5 and outer 51 by less
    than the initialisation range). It ranks 3/5 on scattered and 4/6 on band.
- **In interpolation the error sits in the upper 500 m** (`profiles.png`). In interpolation it is 0.014 to 0.055 K there and 0.003
  to 0.016 K below 1000 m.
- **The warm edge is the hardest region, and the corner (c_k 0.8, c_eps 0.0875) the hardest run.** The values below are from `examples.png`, which
  shows the first initialisation set.
  - One row from the training set (top row): correlation 0.99, amplitude 0.99, 0.157 K.
  - Three cells away (outer 75): correlation 0.62, amplitude 0.33, 1.052 K (1.06 K over the four sets).
  - Every method is between 1.06 and 1.45 K there.
- **Part of the extrapolation error is the basis.** Projecting the held-out truth on the training EOFs leaves 0.010
  to 0.017 K in interpolation and 0.031 to 0.079 K in extrapolation. On outer 75 it is spread along the warm edge
  (†); the corner's own floor is 0.85 K.
- **Extrapolated fields can be statically unstable (†).** Toward the warm corner, predicted columns have
  temperature (and so density) inversions of up to several K, against at most 0.17 K in the truth, for every method
  and for the projection of the truth on the basis: the decoder is the cause. They are not usable model states
  without a convective adjustment.
- **The seed-ensemble spread is not an uncertainty.** It is 0.003 to 0.011 K while the error reaches 0.114 K.
- **Four training runs: cross-validation cannot choose.**
  - Cross-validation cannot choose there: its 5 folds are leave-one-out, so the repeats are identical.
  - Both rules pick alpha = 1, which makes the network too flat (45 % of the anomaly at the centre run).
  - Over the ten configurations the network ranges from 0.10 to 0.19 K, against 0.120 K for the GP on s.

Figures in `results/`:
- `data.png`: the domain-mean target (degrees C) over the grid, with the setup default, and against s.
- `splits.png`: the table above.
- `profiles.png`: RMSE by depth for every split.
- `examples.png`: zonal-mean sections of the anomaly relative to the training mean (truth, network, error). The
  centre run of the 7 x 7 block is held out with the block and in the four-run split; the warm corner with the top
  row, with the outer 75 and in the four-run split.

## Usage (from the miniveros directory)

```bash
python -m det.lastyear --raw-dir ../full_state/mnk965ig   # data/veros_acc_lastyear.npz, seconds
python -m det.run                                         # results/metrics.csv, levels.csv, selected.csv, summary.md, examples.npz (46 min)
python -m det.sensitivity                                 # results/sensitivity.csv (every configuration and seed set, other CV partitions; 28 min)
python -m det.tables                                      # results/table_*.tex for the report, and the table above
python -m det.figures                                     # results/*.png
python -m det.grid                                        # data/det_grid/<split>.npz: predictions at every grid point (30 s)
```

`det.grid` feeds the evaluation reports of the MiniVeros-Diffusion repository (branch `deterministic-evaluation`,
`deterministic_evaluation/`): the network and the GP on s fitted on each split, predicting all 100 runs.

Needs numpy, scipy, scikit-learn, joblib, threadpoolctl and matplotlib (all in the conda base environment); no
torch.
