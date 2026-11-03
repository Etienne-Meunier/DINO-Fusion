# Surface lightening of C7: experiments

Context: C7 (`negative_grad_density_huber`) lowers the upper-ocean density, almost entirely by warming, and the effect grows with eta (agentA/report.txt). Mechanism (agentB): the hinge only ever pushes level 0 lighter, the DPS backprop through the UNet keeps only the level-wide (DC) part of the push, and the constraint is never satisfied, so lbd grows and the push accumulates.

Code: `exp.py` (losses, copy of `sampling.sample` with several constraints, per-constraint eta / eps / lbd0 / mode / t_on), `run.py` (one run), `queue*.sh` (runs as launched), `calib.py` (data statistics, `calib.npz`), `evaluate.py` (table, `evaluate.txt` / `evaluate.csv`). Nothing outside this folder was modified. `run.py --variant c7` reproduces the repo C7 eta 100 run bit-exactly (`runs/check_c7_eta100`).

All runs: seed 0, batch 8, BorderZero, 1000 steps, paired member by member with `none_border0_b8`. C7 always at eta 100, eps 8.6e-8, lbd0 1 unless stated. Single seed, 8 members: the standard error of the paired level-0 shift is 0.03 to 0.10 kg/m3.

## Table

Bias = run minus data batch (sigma_0, volume-weighted horizontal mean, kg/m3). E = eval density error (% volume).

| run | bias level 0 | bias 61-299 m | T0 (C) | S0 | E | C7 loss / eps | final lbd |
|---|---|---|---|---|---|---|---|
| data | 0 | 0 | 20.0 | 35.3 | 0.36 | 1 | |
| none | +0.64 | +0.18 | 17.7 | 35.4 | 9.28 | 68800 | |
| C7 eta 30 | -0.19 | +0.07 | 20.8 | 35.3 | 1.74 | 5440 | 189 |
| C7 eta 100 | -1.11 | -0.51 | 24.3 | 35.4 | 0.75 | 2150 | 413 |
| profile hinge eta 30 | +0.26 | +0.30 | 19.0 | 35.3 | 4.14 | 16900 | 66 |
| profile hinge eta 100 | +0.17 | +0.26 | 19.4 | 35.3 | 3.93 | 14800 | 86 |
| C7 surface stop-grad eta 30 | +0.70 | +0.34 | 17.7 | 35.5 | 2.73 | 14100 | 246 |
| C7 surface stop-grad eta 100 | +0.44 | +0.29 | 18.9 | 35.5 | 2.09 | 6360 | 604 |
| profile (100) + C7 stop-grad (100) | +0.01 | +0.02 | 20.5 | 35.5 | 1.53 | 4380 | 51 / 508 |
| profile (100) + C7 (30) | -0.31 | 0.00 | 21.3 | 35.3 | 1.65 | 5180 | 59 / 173 |
| C7 + surface z, raw, eta 1e-5 | diverged | | 85.8 | 23.5 | 6.26 | | |
| C7 + surface z, raw, eta 1e-7 | -2.66 | -1.98 | 21.4 | 35.7 | 6.64 | 79200 | 5350 / 4.1 |
| C7 + surface z, raw, eta 1e-8 | +0.01 | +0.03 | 20.1 | 35.4 | 1.56 | 5570 | 621 / 0.016 |
| **C7 + surface z, sqrt, eta 1e-4** | **0.00** | +0.16 | 20.0 | 35.3 | **1.20** | 2710 | 565 / 0.41 |
| C7 + surface z, sqrt, eta 1e-5 | +0.06 | +0.13 | 20.0 | 35.4 | 1.33 | 2940 | 552 / 0.013 |
| C7 + surface TS, eta 0.1 | -0.43 | -0.22 | 21.9 | 35.4 | 1.10 | 3040 | 458 / 7.6 |
| C7 + surface TS, eta 0.1, t < 500 | -0.95 | -0.49 | 23.7 | 35.4 | 0.95 | 3170 | 416 / 6.3 |
| C7 + surface TS, eta 1 | -0.12 | -0.02 | 20.7 | 35.4 | 1.25 | 3050 | 505 / 18.6 |

Constraints:
- profile hinge: one-sided Huber on the sample's own volume-weighted mean density profile, eps 0 (never inverted on the 1800 training states).
- C7 surface stop-grad: rho[0] detached in the top pair.
- surface z: `SurfaceMean`, ((T0 - mT) / sT)^2 + ((S0 - mS) / sS)^2 with the volume-weighted level-0 mean T and S, mT, sT, mS, sS over the training states (19.99 +/- 0.125 C, 35.289 +/- 0.017), eps 9 (data max 8.3).
- surface TS: `SurfaceZeroMean`, the zero_mean (TS) loss restricted to the normalised level-0 T and S channels, eps 2.5e-4 (data max 2.4e-4), lbd0 1, as the paper TS run (eta 0.1).

## Reading

- C7 + a surface T/S constraint removes the bias. The sqrt mode of the surface constraint is the most robust: the push is bounded, the surface lbd settles by itself (0.4 at eta 1e-4, 0.013 at 1e-5, same result), and the surface constraint is satisfied (z^2 about 0.7 at the end). It reaches E 1.20, below C7 eta 30 (1.74), with T0 and S0 on the data.
- In raw mode the surface z loss is about 34000x the TS loss scale, so it needs eta about 1e-8; 1e-7 and above diverge.
- The surface TS loss at the paper TS settings (eta 0.1) is too weak against C7 (lbd ~ 450): the surface loss stays 40x above eps and only 60 % of the shift is removed. eta 1 removes 90 %. Starting the surface constraint at t < 500 is worse (-0.95): the level means are set early.
- Removing the bias costs density error: E goes from 0.75 (C7 eta 100) to 1.1-1.6. Part of C7's E came from over-stratifying the upper ocean, which hides small-scale inversions.
- The sqrt runs leave 61-299 m too dense (+0.13 to +0.16, close to none's +0.18): the level-0 constraint also holds back the lightening below, which was partly a correct correction of none.
- Reference-free alternative: profile hinge + C7 with the surface stop-grad gives the same bias (+0.01) and E (1.53) without any data statistics, but S0 is 0.2 too salty.

## Rounds 5-7: refinement of the two retained losses

Code: `queue5.sh`, `queue6.sh`, `queue7.sh`, full table in `logs/evaluate7.log`. New in `exp.py`: `Reference`, `SurfaceAnchor`, `SurfaceAnchorRho`, `ProfileAnchor`, `HuberSkipTop`, and `sample(reference=...)`, which samples the unconstrained trajectory of the same seed alongside (own generator, same noise; its final sample equals none_border0_b8 exactly) and exposes its x0 estimate to the anchor losses at every step. Seeds 1 and 2 are paired with their own unconstrained runs (`runs/none_s1`, `runs/none_s2`, or `ref.npy`).

Loss S, surface data target (C7 + `SurfaceMean`, sqrt). Retained: C7 eta 100, surface eta 1e-4, eps 9.

| setting | bias level 0 | bias 61-299 m | T0 | E |
|---|---|---|---|---|
| C7 eta 30 / 50 / 100 / 200 / 300 | ~0 | 0.20 / 0.18 / 0.16 / 0.11 / 0.07 | 20.0 | 1.65 / 1.43 / 1.20 / 1.31 / 1.46 |
| surface eps 1 / 4 / 9 / 16 | ~0 | 0.16 | 20.0 | 1.21 / 1.20 / 1.20 / 1.23 |
| surface eta 1e-3 / 3e-4 / 1e-5 / 1e-6 | diverged / -0.38 / +0.06 / -0.22 | | | 11 / 4.4 / 1.33 / 1.50 |
| seeds 1, 2 | ~0 | 0.22 / 0.10 (none 0.28 / 0.10) | 20.0 | 1.04 / 1.23 (C7 alone 0.63 / 0.62) |

Loss A, no surface data: the level-0 mean T and S are anchored to those of the unconstrained trajectory of the same seed, ((T0 - T0ref) / 0.1 C)^2 + ((S0 - S0ref) / 0.01 psu)^2, sqrt mode. Retained: C7 without the top pair (`HuberSkipTop`) at eta 100, anchor eta 1e-4, eps 1 (or larger eps as a trade-off).

| setting | bias level 0 | bias 61-299 m | dT0 vs none | S0 | E |
|---|---|---|---|---|---|
| C7 + anchor, eta 1e-4 eps 1 | +0.65 | +0.49 | +0.01 | 35.4 | 2.58 |
| same, eta 1e-5 / eps 0.25 / eps 4 | +0.55 to +0.67 | +0.47 to +0.49 | ~0 | 35.4 | 2.37-2.59 |
| same, eta 1e-3 | diverged | | | | 13 |
| C7 + one-sided density anchor | +0.64 | +0.40 | +0.3 to +0.4 | 35.5 | 2.4 |
| C7 + anchor to the final none sample | +0.34 | -0.13 | | 35.5 | 6.4 |
| C7 eta 30 / 200 / 300 + anchor | | | | | 2.6 / 3.3 / 3.4 |
| anchor on T only | +0.33 | +0.32 | -0.11 | 34.9 | 2.04 |
| anchor on the top 12 / all 36 level means | +0.56 / +0.21 | +0.24 / +0.26 | +0.25 / +1.55 | 35.4 / 35.3 | 3.6 / 3.1 |
| profile + C7 stop-grad + anchor | +0.64 | +0.57 | -0.02 | 35.4 | 2.47 |
| C7 without top pair, alone | -1.05 | -0.52 | +6.4 | 35.4 | 0.88 |
| **C7 without top pair + anchor, eps 1** | +0.65 | +0.23 | 0.00 | 35.4 | 2.24 |
| same, seeds 1 / 2 | +0.57 / +0.70 | +0.40 / +0.25 | +0.04 / -0.01 | 35.3 | 1.96 / 2.16 |
| same, anchor eta 1e-5 | +0.63 | +0.20 | +0.08 | 35.4 | 2.17 |
| same, eps 16 / 100 / 400 | +0.63 / +0.54 / +0.37 | +0.23 / +0.19 / +0.15 | +0.06 / +0.44 / +1.08 | 35.4 | 2.17 / 2.07 / 1.89 |

Reading:
- Loss S is settled and robust over 3 seeds: no surface bias, the subsurface keeps the bias of none, E 1.0-1.2 against 0.6-0.75 for C7 alone. eps is irrelevant between 1 and 16; eta 1e-4 is at the stability edge (3e-4 is unstable).
- The anchor does what it is built for: the surface stays at that of the unconstrained sample (dT0 ~0). But with the full C7 the constraint then densifies the levels below the surface (61-299 m +0.49 against +0.18 for none). Dropping the top pair of C7 removes most of it (+0.20 to +0.25, close to none on 2 of 3 seeds) and lowers E to 2.0-2.2.
- C7 without the top pair alone still warms the surface by 6.4 C: the warming is not caused by the top pair but by the level-wide (DC) response of the UNet to the push on the pairs below.
- The anchor floors at E about 2: it keeps the surface of none, which is too dense and inverted, and part of C7's surface change was a legitimate correction of it. Loosening the anchor trades E against surface drift (E 1.89 for a 1 C drift at eps 400) and it stays above loss S and above the reference-free profile + C7 stop-grad (E 1.53, bias +0.01, S0 0.2 too salty).

## Rounds 8-9: strength sweep and batch 16

Strength sweep of the surface data loss (`queue8.sh`, `fig_strength.py`, figures/fig_strength_surface.png): with the surface term the C7 eta moves E between 9.3 (eta 0), 2.5 (10), 1.65 (30) and a floor of 1.20 (100), rising again to 1.46 at 300, while the level-0 bias stays at 0 and the RMSE of the batch-mean sigma_0 to the data stays at 0.13 (C7 alone: 0.49 at eta 100, 1.15 at 300).

Batch 16, seed 0, next to the paper optimal runs (`queue9.sh`, `table_scores.py`, figures/table_scores.txt / .tex): C7 + surface T,S mean gives E 2.5 / 1.7 / 1.3 at C7 eta 10 / 30 / 100 with surface T 20.0 +/- 0.05 and bias 0.00 +/- 0.01, against C7 alone 2.8 / 1.7 (bias +0.18 / -0.22, T0 spread 1.2 C). The box metrics (BW, DW) are unchanged within the batch spread.

## Rounds 10-11: data-free alternatives to C7

Goal: a constraint that uses no information on the data profile and gives a smaller surface bias than C7 at the same density error E. Batch 8, seed 0 unless stated; the C7 reference at the same E is read on the C7 eta curve.

| run (profile eta / C7 stop-grad eta) | E | bias lev0 | bias 61-299 m | S0 | C7 alone at same E (eta, bias lev0) |
|---|---|---|---|---|---|
| profile + stop-grad 100 / 50 | 1.92 | +0.18 | +0.13 | 35.4 | ~25, -0.1 |
| profile + stop-grad 30 / 100 | 1.71 | +0.18 | +0.13 | 35.5 | 30, -0.19 |
| profile + stop-grad 100 / 100 | 1.53 | +0.01 | +0.02 | 35.5 | ~40, -0.33 |
| profile + stop-grad 300 / 100 | 1.38 | -0.18 | -0.11 | 35.5 | ~45, -0.42 |
| profile + stop-grad 100 / 200 | 1.03 | -0.57 | -0.42 | 35.7 | ~68, -0.72 |
| profile + stop-grad 100 / 300 | 0.80 | -1.05 | -0.81 | 36.0 | ~92, -1.01 |
| profile + stop-grad 300 / 300 | 0.76 | -1.22 | -0.93 | 35.9 | ~95, -1.05 |
| profile + C7 without top pair 100 / 100 | 0.87 | -1.11 | | 35.4 | same as C7 |
| T-only C7 50 | 1.74 | -0.70 | -0.13 | 35.4 | 30, -0.19 |
| T-only C7 100 | 0.90 | -1.42 | -0.63 | 35.5 | ~80, -0.87 |
| T-only profile + stop-grad 100 / 100 | 1.34 | -0.40 | -0.15 | 35.4 | ~48, -0.45 |
| T-only profile + stop-grad 300 / 100 | 1.22 | -0.63 | -0.29 | 35.4 | ~55, -0.55 |
| T-only profile + stop-grad 100 / 200 | 0.74 | -1.21 | -0.76 | 35.6 | ~97, -1.08 |
| T-only profile + stop-grad 300 / 300 | 0.51 | -1.82 | -1.18 | 35.7 | ~150, -1.5 |

- The profile hinge + C7 with surface stop-grad is the only data-free combination that beats C7: between E 1.4 and 1.6 the surface bias is 0.25 to 0.3 kg/m3 smaller. Beyond C7 eta 200 the gain is gone and the surface turns salty (S0 35.7-36.0 against 35.3 in the data).
- Removing the (0,1) pair from C7 changes nothing: the run follows the C7 curve, the surface still lightens through the coupling of the levels in the denoiser.
- Gradient through temperature only (salinity detached) is worse everywhere: the guidance can only warm the surface to make the column stable, so the warming (T0 22.8 at E 1.74 against 20.8 for C7) and the bias grow. The salinity drift goes away but the cure is worse than the drift.
- Batch 16, seed 0: profile 300 / C7 stop-grad 100 gives E 1.4, bias lev0 -0.25, T0 21.64 +/- 1.70, S0 35.55. C7 eta 50 gives E 1.3, bias -0.53; C7 eta 30 gives E 1.7, bias -0.22. The gain over C7 at equal E is about 0.2-0.3 kg/m3, with a saltier surface (+0.24) and the same T0 spread as C7. The surface T,S mean constraint stays far ahead (E 1.3, bias 0.00, T0 19.99 +/- 0.04) but uses the data climatology.

## Round 12: origin of the surface bias, centred sampler

Question: is the surface bias a season the model over-represents ? No.

- Training data (`seasonality.py`): 1800 states, period 36 states (1 year of 10-day outputs). The level-0 mean T moves only between 19.80 and 20.18 (19.99 +/- 0.12), S0 35.289 +/- 0.017 ; the `_summer` archive is 20.10 +/- 0.06.
- Generations against the 36-phase climatology (`season_match.py`, `figures/season_match.png`): RMSE of the level-0 T map to the nearest phase 2-6 C, against 0.35 for the data ; the nearest phase is always one of the two extremes (out of distribution, not a season). The zonal-mean offset is the same at every latitude (none_border0 -2 C, C7 eta 80 +3 to +4 C), while the seasonal envelope is < 0.5 C in the south and 1-2 C at 40-65N.
- One-step denoising of noised training states (`denoise_test.py`): surface-mean error within +/-0.1 C up to t = 900, +/-5 C at t = 990 only. No clipping (level-0 normalised T in [-0.62, 0.46]).
- Level-0 T of x0_hat along a generation (`traj_test.py`, b8 seed 0): the surface mean of each member is set between t = 999 and t = 950, then drifts slowly. Its spread comes from the spatial mean of the noise added at every step (centring only the initial noise changes nothing, centring the noise of every step takes the final spread from 1.11 to 0.15). Its offset comes from BorderZero: at high noise the land cells of x_t are noise in training, BorderZero puts them at 0 and the network then cools the surface by about 2.5 C, the same for every member.

| sampler (traj_test, b8) | final T0 |
|---|---|
| no projection | 19.45 +/- 3.93 |
| no projection, centred noise | 20.70 +/- 1.82 |
| BorderZero | 17.74 +/- 1.11 |
| BorderZero, centred noise | 17.54 +/- 0.15 |
| noisy land cells | 20.21 +/- 1.43 |
| noisy land cells, centred noise | 20.11 +/- 0.12 |

Centred sampler (`exp.ddpm_step`, `run.py --centred`): at every step the added noise has its per-channel spatial mean over the ocean removed, land and padding cells are refilled with noise of the next noise level, the guidance step is masked to the ocean and BorderZero is applied after the last step only. No data information. Full runs, b8 seed 0 (`queue12.sh`, `logs/evaluate12.log`):

| run | E | bias lev0 | bias 61-299 m | T0 | S0 | old sampler at same E (C7 eta, bias lev0) |
|---|---|---|---|---|---|---|
| centred, unconstrained | 2.42 | -0.01 | 0.00 | 20.11 +/- 0.13 | 35.293 | none_border0: E 9.3, +0.64 |
| centred, C7 10 | 1.50 | -0.35 | -0.08 | 21.28 +/- 0.14 | 35.266 | ~40, -0.33 |
| centred, C7 30 | 0.93 | -0.64 | -0.18 | 22.26 +/- 0.14 | 35.237 | ~80, -0.87 |
| centred, C7 50 | 0.61 | -0.85 | -0.28 | 22.9 | 35.2 | ~150, -1.6 |
| centred, C7 100 | 0.33 | -1.27 | -0.52 | 24.32 +/- 0.17 | 35.177 | ~300, -2.4 |

- Unconstrained, the centred sampler gives E 2.42 (none_border0 9.3, C7 eta 10 2.8) with no surface bias and the data T0 spread.
- C7 still warms the surface (+1.2 C at eta 10, +4.2 at eta 100), the member spread staying at the data level: the guidance moves the surface mean of all members together. At equal E the bias is about half the one of the old sampler from E 0.9 down.

## Round 13: zero_mean, surface mean and centred gradient on the centred sampler

Centred sampler, b8 seed 0 (`queue13.sh`, `logs/evaluate13.log`). zero_mean is the repo `ZeroMean` on the ocean cells (eps 0.01, raw, not volume-weighted), surface is `SurfaceMean` (level-0 T and S mean held at the data, sqrt, eta 1e-4, eps 9), centred gradient (`--centre_grad`) removes the per-level volume-weighted mean of the guidance step.

| run | E | bias lev0 | bias 5-49 m | T0 | S0 |
|---|---|---|---|---|---|
| centred, unconstrained | 2.42 | -0.01 | -0.27 | 20.11 | 35.29 |
| centred, zero_mean eta 0.03 / 0.1 / 0.3 / 1 / 3 | 1.27 / 1.18 / 1.11 / 1.06 / 1.03 | +0.02 | -0.01 to +0.02 | 19.8 | 35.2 |
| centred, zero_mean 0.1 + C7 10 | 0.86 | -0.05 | -0.03 | 19.99 | 35.2 |
| centred, zero_mean 1 + C7 10 | 0.89 | -0.02 | 0.00 | 19.88 | 35.2 |
| centred, C7 10 / 30 / 100 + surface | 1.73 / 1.25 / 0.83 | 0.00 | -0.13 / -0.03 / +0.03 | 20.0 | 35.3 |
| centred, C7 10 / 30 / 100, centred gradient | 2.37 / 2.30 / 2.16 | +0.00 to +0.06 | -0.26 to -0.21 | 19.8-20.1 | 35.3 |
| old sampler, zero_mean 0.1 (zm_b8) | 1.25 | +0.08 | +0.07 | 19.55 | 35.2 |

- zero_mean is data-free and the best E at no surface bias: E 1.0-1.2, saturating from eta 0.3, with T0 at 19.8 (old sampler 19.55). Adding C7 10 brings E to 0.86-0.89 with a bias of -0.02 to -0.05, against -0.35 for C7 10 alone at E 1.50.
- The surface constraint holds the bias at 0 at every C7 eta (E 0.83 at eta 100) but uses the data surface mean.
- The centred gradient removes the C7 effect almost entirely (E 2.16-2.37, unconstrained 2.42): the part of the C7 step that lowers E is its per-level mean, the same part that warms the surface. Dropped.

## Seasonal tests

Question: is the surface warm bias an over-representation of part of the year, and does a constraint flatten the seasons? `season_test.py` matches the level-0 T pattern (global mean removed) of each member to the 36-phase climatology of the 1800 training states; `band_test.py` matches, in the 35-60N band, the surface T and the stratification T(5 m) - T(87 m) separately to the climatology, a seasonal state getting the same quarter from both.

| run | pattern RMSE | phases (quarters) | band T0 | band T0 - T7 | same quarter |
|---|---|---|---|---|---|
| data (64) | 0.35 | 32 (14 17 14 19) | 14.90 +/- 0.52 | 0.84 +/- 0.41 | 0.52 |
| centred, unconstrained | 0.87 | 2 (0 0 8 0) | 15.94 +/- 0.17 | 0.68 +/- 0.20 | 0.00 |
| centred, C7 10 | 0.84 | 3 (0 0 8 0) | 17.12 +/- 0.19 | 1.59 +/- 0.12 | 1.00 |
| centred, C7 100 | 0.96 | 1 (0 0 8 0) | 20.43 +/- 0.18 | 3.19 +/- 0.11 | 1.00 |
| centred, C7 30 + surface | 0.79 | 7 (0 3 5 0) | 15.50 +/- 0.29 | 0.94 +/- 0.22 | 0.12 |
| centred, zero_mean 1 | 0.77 | 6 (0 4 4 0) | 15.11 +/- 0.32 | 0.84 +/- 0.23 | 0.50 |
| centred, zero_mean 1 + C7 10 | 0.76 | 6 (0 4 3 1) | 15.20 +/- 0.32 | 0.98 +/- 0.24 | 0.50 |
| paper zero_mean (b16) | 0.75 | 8 (0 7 9 0) | 14.84 +/- 0.39 | 0.62 +/- 0.26 | 0.62 |

- The seasonal amplitude of the level-0 pattern (at most 0.36 C RMSE to the annual mean) is below the pattern error of every generated state (0.75-0.96 C), so the nearest phase of a generated state is set by its error, not by a season: no run covers the year, all fall in the two middle quarters. The test cannot show a flattening of the seasons at this accuracy.
- The warm bias is not a season. Unconstrained, the band surface says summer and the stratification spring or autumn (same quarter 0.00): it is a regional northern warm bias. C7 makes both agree (1.00) only because it overshoots the stratification (1.59 at eta 10, 3.19 at eta 100, data summer maximum 1.42).
- zero_mean brings the band closest to the data (band T0 15.1-15.2, stratification 0.84-0.98) with the two features agreeing as often as in the data.

## Round 14: batch 32 of the centred sampler

b32 seed 0 (`queue14.sh`, `logs/evaluate14.log`), paired with `centred_none_b32`. The b8 results hold.

| run | E | bias lev0 | bias 5-49 m | T0 | band T0 | band T0 - T7 | band quarters from T0 | same quarter |
|---|---|---|---|---|---|---|---|---|
| data | 0.36 | 0 | 0 | 20.00 +/- 0.12 | 14.90 +/- 0.52 | 0.84 +/- 0.41 | 14 13 17 20 | 0.52 |
| centred, unconstrained | 2.47 | +0.03 | -0.24 | 19.94 +/- 0.23 | 15.66 +/- 0.34 | 0.47 +/- 0.28 | 0 1 26 5 | 0.00 |
| centred, C7 10 | 1.54 | -0.32 | -0.43 | 21.2 | 16.94 +/- 0.27 | 1.49 +/- 0.16 | 0 0 32 0 | 0.91 |
| centred, zero_mean 1 | 1.07 | +0.02 | +0.02 | 19.77 +/- 0.05 | 15.01 +/- 0.38 | 0.71 +/- 0.25 | 0 10 9 13 | 0.50 |
| centred, zero_mean 1 + C7 10 | 0.90 | -0.02 | 0.00 | 19.88 +/- 0.05 | 15.13 +/- 0.37 | 0.89 +/- 0.26 | 0 9 8 15 | 0.53 |
| centred, C7 30 + surface | 1.26 | 0.00 | -0.03 | 20.01 +/- 0.04 | 15.37 +/- 0.35 | 0.87 +/- 0.26 | 0 7 18 7 | 0.34 |

- zero_mean 1 + C7 10 on the centred sampler is the retained data-free setting: E 0.90 with no surface bias, the band surface T and stratification closest to the data and agreeing as often as in the data.
- C7 10 alone puts all 32 members in one quarter of the band, warm and over-stratified: it is C7, not the data-free mean constraint, that flattens the seasons. No run produces the winter quarter.
- zero_mean and the surface constraint reduce the member spread of the global T0 (0.04-0.05, data 0.12, unconstrained 0.23), as they hold the level means.

## Next

- C7 + surface z sqrt at 2 more seeds and batch 16, and the eta of C7 (30, 300) with the surface constraint on.
- Surface TS at eta 3 to 10 or in sqrt mode, to check that it matches the surface z sqrt result.
- A posteriori run of the retained setting.
- Other seeds of zero_mean 1 + C7 10 on the centred sampler, and its a posteriori run.
