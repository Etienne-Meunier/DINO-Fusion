# Preliminaries: eps, enforcement and relaxation of the constraints (exploration of 2026-10-14)

This note gathers the exploration carried out on the three preliminary questions of `../analytic_approach.md`. Unless stated otherwise, the numbers are obtained with 100-step DDPM sampling, a batch of size 8 and seed 0 (three seeds for the relaxation), with the model `tav0h83b`. All the runs use the `BorderZero` projection, and the Tweedie estimate is clipped to [-1, 1] as in `pipelines/sampling.py`. The 1000-step numbers come from the runs of the generation-pipeline session (`test_new_sampler_clip/`). Let us note that these are explorations on small batches, and not paper-grade statistics.

## Summary

The unconstrained generations lie 2 to 8 orders of magnitude above the data on every loss. Thus, we take the median of the training losses (q0.5, `outputs/eps_data_quantiles.csv`) as the target eps. Setting eps = 0 only makes sense for `negative_grad_density`, which is exactly zero on the training states.

The clip of the Tweedie estimate is required to enforce any constraint. Besides, the sqrt mode (g = sqrt(loss) - sqrt(eps)) outperforms the raw mode on every density loss. `zero_mean` is the easiest and most useful constraint, while `dist_isoton` works only in a narrow window of eta. A final LBFGS projection (`final_opt`) closes the remaining gap. The best pipeline we found is a guided `dist_isoton + zero_mean` with one dual variable per constraint, in sqrt mode, with eta = (1e-4, 1) at 100 steps or (5e-6, 0.1) at 1000 steps with the `zero_mean` eps relaxed by a factor 64 or more, followed by a joint post-hoc projection on `mean_density + 0.01 dist_isoton`. It reduces the local inversions from 25 % to 2 % (with almost no inversion above 0.01 kg/m3), removes the profile bias and yields a spread of the level means close to the data.

Guidance and post-hoc projection are equivalent on the constrained statistic itself. However, guidance also corrects the local structure that the constraint does not see, which results in fewer local inversions.

Regarding variability, the excess variance of the model mostly consists of random per-level biases (30x the data), plus a 2x excess of spatial anomalies. A constraint acts only on the variance of what it sees. Indeed, relaxing eps gives back that variance (eps ~ 64 x q0.5 reproduces the data for `zero_mean`) and nothing else. The excess of anomaly variance is horizontal small-scale noise (roughness 2.5x the data), which none of the constraints we tried is able to reduce.

## Files

| file | role |
|---|---|
| `eps_data.py` | constraint losses on the 1800 training states, quantiles in `outputs/eps_data_*` |
| `data_reference.py` | diversity and distance of random training batches, in `outputs/data_reference.json` |
| `pd_explore.py` | `sample_pd`, a copy of `sampling.sample` with variants (mode raw/rel/log/sqrt, t_start, lbd_max, final_opt, through_unet, stacked losses with one dual each), trace, and the metrics `diversity`, `distance`, `variance_split`, `inversion_fraction` |
| `sweep.py` | resumable grid runner, one folder per run in `$OCEANDATA/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/` (`generated.npy`, `lbd.npy`, `trace.npz`, `metrics.json`) |
| `analyze_sweep.py` | `outputs/sweep_table.csv`, `outputs/sweep_<loss>.png` (metrics vs. eta), `outputs/sweep_frontier.png` (loss vs. distance and diversity) |
| `local_metrics.py` | inversion fraction, roughness and variance split for data batches and every saved run, in `outputs/local_metrics.csv` |
| `plot_relax.py` | Q2 figure `outputs/relax_<loss>.png` |
| `compare_runs.py` | table of local metrics and losses for any generation folder (e.g., the 1000-step runs of `test_new_sampler_clip/`) |
| `joint_projection.py` | post-hoc projection of saved runs on mean_density + c dist_isoton, in `<run>_joint<c>/` |

The saved states are normalised, of shape (b, 37, 208, 64). The physical fields are obtained with `Diffusion_Model/post_processing.py --generated <path>/generated.npy`. The post-hoc projections of the unconstrained runs are stored in `prelim_sweep_clip/posthoc_{unc100,unc1000}_<loss>/`.

## Metrics

- Final losses: every constraint loss of `pipelines/losses.py` evaluated on the final states, median over the batch. The five losses are recorded for every run, so that the side effects of one constraint on the others are visible;
- Satisfaction: fraction of the batch with loss <= eps;
- Distance: RMS over the ocean cells of the difference with the unconstrained state of the same seed (normalised units). As a reference, two random training states are 0.044 apart;
- Diversity: mean over the ocean cells of the standard deviation over the batch. It is 0.022 for the data and 0.067 for the unconstrained generations;
- Variance split: the variance over the batch is split into the variance of the per-level spatial means (one number per channel and sample, i.e., what the horizontal-mean constraints act on) and the variance of the spatial anomalies. Data: 1.2e-4 + 8.9e-4. Unconstrained: 3.4e-3 to 4.1e-3 + 1.9e-3;
- Inversions: fraction of vertical pairs of ocean cells (strided levels) where the density decreases with depth. It is 0.6 % for the data and 25 % (100 steps) or 26 % (1000 steps) for the unconstrained generations. It is a local metric on which no constraint acts directly;
- Roughness: mean squared difference between neighbouring ocean cells (normalised). It is 1.2e-3 for the data and between 2.8e-3 and 3.1e-3 for every generation.

## 0 - eps from the data

The quantiles of the per-sample losses on the 1800 training states (`outputs/eps_data_quantiles.csv`) are reported below.

| loss | q0.5 | q0.9 | q1.0 | unconstrained generation (median) |
|---|---|---|---|---|
| mean_density | 6.3e-3 | 1.3e-2 | 2.7e-2 | 1.32 |
| mean_grad_density | 1.5e-3 | 1.9e-3 | 2.2e-3 | 0.92 |
| negative_grad_density | 0 | 0 | 0 | 0.40 |
| zero_mean | 2.6e-3 | 8.2e-3 | 1.1e-2 | 0.24 |
| dist_isoton | 3.5e-5 | 3.8e-5 | 4.3e-5 | 6.6e3 |

The unconstrained generations lie 2 to 8 orders of magnitude above the data on every loss. Consequently, eps = 0 is never the right target, except for `negative_grad_density`, which is exactly 0 on all the training states. The sweep therefore uses eps = q0.5 (floored at 1e-6 for `negative_grad_density`).

We found two bugs on the way. First, `IsotonicDensity` lost precision in float32 (densities of about 1028 accumulated in a cumsum) and gave a fake floor of 3.6e-2 on the data. We fixed it in `pipelines/losses.py` by working on the anomaly w.r.t. the surface value. Second, the losses use `nansum`, so that a NaN state reports a loss of 0. As a consequence, the constrained runs in `test_new_sampler/` (obtained before the clip) are all NaN and must be ignored.

## 1 - Can we enforce the constraints?

Without the clip of the Tweedie estimate, every constrained run diverges. Indeed, the loss of x0_hat is of the order of 1e6 to 1e8 at the first step, and the dual variable explodes. With the clip, we obtain the following results.

| loss | raw mode | sqrt mode (g = sqrt(loss) - sqrt(eps)) |
|---|---|---|
| mean_density | best eta 1e-2: 0.28 (dist 0.044), a larger eta fits the profile by T/S compensation (zero_mean 0.24 to 9.4) | eta 1: 1.25e-2, all samples in the data range, no T/S compensation (dist 0.062) |
| mean_grad_density | best eta 1e-2: 0.27 | eta 1: 1.0e-2 (dist 0.053) |
| negative_grad_density | best eta 0.1: 0.04 | eta 1: 4e-5, 2/8 satisfied (dist 0.040); eta 10: 7/8 but dist 0.225 |
| zero_mean | eta 10: 1.8e-3, 7/8 satisfied (dist 0.091) | eta 10: 8/8 (5e-6, overshoot), same states |
| dist_isoton | no usable eta, the loss only drops once the state is destroyed | narrow window at eta 1e-4: 5.2, inversions from 25 % to 5.4 % (dist 0.073); 1e-5 is too weak, 1e-3 blows up |

The sqrt mode is the better formulation for the density-profile losses. Indeed, its gradient is bounded near the constraint, so that the dual variable does not inflate on the huge early losses of x0_hat. At 1000 steps, eta transfers as eta_100 / 10: `mean_density` in sqrt mode with eta 0.1 gives 4/8 satisfied samples, 8/8 in the data range, and a profile_rmse of 0.020 (0.024 for the data).

At large eta, the per-sample dual runs away on part of the batch (lbd of about 40 at 100 steps and 185 at 1000 steps), and the state degrades (distance between 0.3 and 3, diversity up to 15x the data). In the case of `mean_density`, the profile can be matched by a T/S compensation, which `zero_mean` exposes. Besides, the last steps pull the loss back up. For `mean_density` in sqrt mode with eta 1, the loss of x0_hat reaches 5e-3 at t = 90, whereas the final state ends at 1.25e-2.

`zero_mean` is the constraint that works best. In addition, it drags all the other losses towards the data (mean_density 1.35e-2, negative_grad_density 6.6e-5, dist_isoton 56, diversity 0.028).

We also tested several variants of the sampler:

- `final_opt` (LBFGS on the final state until loss < eps, `OptimProjection`) closes the gap. For `mean_density` in raw mode with eta 1e-2, the loss goes from 0.28 to 3.1e-3 (8/8 satisfied) and the distance from 0.044 to 0.055. 90 % of the correction is a uniform shift per level;
- `t_start = 700` removes the overshoot of the raw mode at large eta, while `t_start = 400` is too late. At small eta, a late start only weakens the constraint;
- `lbd_max` does not help, since a large constant lbd from the start pushes the whole trajectory;
- `through_unet = False` (no Jacobian of the UNet) gives the same result and is 25 % faster;
- one dual per loss for `mean_density + zero_mean` (raw) is better than `mean_density` alone in raw mode, but worse than the sqrt mode.

`dist_isoton` is a local column constraint. It only works in sqrt mode, in a narrow window around eta = 1e-4 at 100 steps. The loss then goes from 6.6e3 to 5.2 and the local inversions from 25 % to 5.4 %, the best value over all single-constraint runs, at a distance of 0.073 and a diversity of 0.053. Without the UNet Jacobian (`through_unet = False`), we obtain the same result (5.3 %) 40 % faster. At 100 steps, it moves the mean profile only slightly (mean_density 0.95). However, at 1000 steps (eta 5e-6, i.e., a transfer of /20), it removes the inversions partly by drifting the T/S level means (profile_rmse 2x the unconstrained one, zero_mean 2x to 7x). Thus, it requires an anchor on the level means. The raw mode has no usable eta, and the sqrt mode with eta >= 1e-3 blows up unless the guidance starts late (`t_start = 300` with eta 1e-3 gives 8.4 % of inversions at distance 0.044).

### Combining a local and a mean constraint

We combine `dist_isoton + zero_mean` with one dual variable each (sqrt mode for both, `pd_explore.Stacked`), at 100 steps and with three seeds.

| config (eta iso, eta zm) | inversions | inversions > 0.01 kg/m3 | level-mean variance | anomaly variance | distance | diversity |
|---|---|---|---|---|---|---|
| data | 0.6 % | 0 | 1.15e-4 | 8.9e-4 | 0.044 (two states) | 0.022 |
| unconstrained | 25 % | 23 % | 3.4e-3 to 4.1e-3 | 1.9e-3 | 0 | 0.067 |
| zero_mean raw 10 | 10.2-11.4 % | 8 % | 2e-5 | 1.7e-3 | 0.091 | 0.028 |
| dist_isoton sqrt 1e-4 | 4.8-5.4 % | 2.5-3 % | 1.8e-3 to 2.2e-3 | 1.8e-3 | 0.073-0.084 | 0.053-0.056 |
| (1e-4, 1) | 2.4-2.6 % | 0.8 % | 7.4e-5 to 8.4e-5 | 1.7e-3 | 0.086-0.093 | 0.031 |
| (1e-4, 1) + final_opt (seed 0) | 0.39 % | 0 | 7.4e-5 | 1.7e-3 | 0.086 | 0.031 |
| (1e-4, 3) | 4.8-5.1 % | 1.5 % | 2e-7 to 4e-7 | 1.6e-3 | 0.093-0.100 | 0.028 |
| (3e-5, 1) | 5.9-6.1 % | 2.8 % | 8.5e-5 to 9e-5 | 1.7e-3 | 0.080-0.085 | 0.030 |
| dist_isoton + mean_density (1e-4, 1) | 9.9 % | 7.7 % | 1.3e-3 | 1.65e-3 | 0.105 | 0.046 |

The configuration (1e-4, 1) is the best one of the exploration. It yields ten times fewer inversions than the unconstrained model and a level-mean variance at the level of the data. With `final_opt`, the local inversions even go below the data value. Let us note that `zero_mean` is not satisfied at eps = q0.5 (7e-3), which is precisely what keeps its level-mean variance at the data level.

However, a stronger `zero_mean` dual (eta 3 or 10) over-shrinks the level means (variance between 1e-7 and 1e-6) and doubles the inversions, i.e., the two constraints compete. Besides, `mean_density` is a worse anchor than `zero_mean`. Indeed, it constrains only one T/S combination per level, and the compensating direction stays free.

The level-mean variance measures the spread across the batch, and not a bias shared by the whole batch. The configuration (1e-4, 1) still has a mean_density of 0.06 (q1.0 of the data is 2.7e-2), with all samples off the training profile in the same direction. Consequently, bias and spread have to be reported together (mean_density or profile_rmse next to the level-mean variance).

Adding `mean_density` as a third constraint (sqrt mode, one dual each) removes the bias, but at the cost of more inversions. The configuration (1e-4, 1, 0.3) gives a mean_density of 1.5e-3, a zero_mean of 3.7e-3, a level-mean variance of 4e-5 and 6 % of inversions, while (1e-4, 1, 1) gives 9 % of inversions. Again, the three constraints compete.

At 1000 steps (generation session, `test_new_sampler_clip/combo_*`, `compare_runs.py`), the zero_mean eta is bistable. On one hand, eta between 0.02 and 0.1 crushes the level means (variance from 4e-7 to 7e-7, inversions from 5 to 6 %). On the other hand, eta 0.01 lands at the data level (variance 8e-5, inversions 2.0 %, 0.6 % above 0.01 kg/m3), but with a shared profile bias (mean_density 0.215). Thus, we move the control to eps (zero_mean eps x16 and x64 at eta 0.1).

### Best pipeline so far: guided local and mean constraints, followed by a joint projection

Guiding a third constraint (`mean_density`) competes with `dist_isoton`: (1e-4, 1, 0.1) gives 4.6 % of inversions and (1e-4, 1, 0.3) gives 6 %, while raising the dist_isoton eta makes it worse (8 to 9 %). A post-hoc projection on `mean_density` alone also recreates inversions (from 2 % to 8 % at 1000 steps), since the uniform per-level T/S shift breaks the stability of the columns. In contrast, the joint post-hoc projection on `mean_density + 0.01 * dist_isoton` (eps 0, 10 LBFGS steps, `joint_projection.py`, proposed by the generation session) removes the bias without that cost.

| run (100 steps, two seeds) | level-mean variance | inversions | inversions > 0.01 | mean_density | distance |
|---|---|---|---|---|---|
| dist_isoton + zero_mean sqrt (1e-4, 1), eps q0.5 | 7.4e-5 to 8.3e-5 | 2.4-2.6 % | 0.8 % | 0.063 | 0.086-0.099 |
| + joint projection | 6.7e-5 to 7.7e-5 | 2.0-2.4 % | 0.002-0.003 % | 9e-4 | +0.001 |
| zero_mean eps x64 + joint | 3.0e-4 to 3.5e-4 | 2.9-3.2 % | 0.01 % | 3e-3 | |
| zero_mean eps x256 + joint | 8.2e-4 to 9.7e-4 | 4.0-4.5 % | 0.05 % | 7e-3 | |
| data | 1.15e-4 | 0.6 % | 0 | 6.3e-3 | 0.044 |

At 1000 steps (generation session, eta (5e-6, 0.1)), the zero_mean eps that yields the data level-mean variance is larger. Indeed, x64 still gives 2.9e-5 (x16 gives 7e-6), with 2.0 % of inversions and, after the joint projection, 1.55 % of inversions, 0.002 % above 0.01 kg/m3 and a mean_density of 5.6e-4. The guided triple at 1000 steps confirms the competition (mean_density eta 0.03 and 0.1 give 5.0 % and 7.1 % of inversions, with an anomaly variance up to 3e-3). We then relaxed the zero_mean eps at 1000 steps, with eta (5e-6, 0.1) and seed 0, each run being followed by the joint projection (c = 0.01).

| zero_mean eps | level-mean variance | inversions | inversions > 0.01 | mean_density before / after joint | profile_rmse |
|---|---|---|---|---|---|
| x64 | 2.6e-5 | 1.55 % | 0.002 % | 0.111 / 5.6e-4 | 0.007 |
| x256 | 5.0e-5 | 2.25 % | 0.006 % | 0.347 / 1.1e-3 | 0.009 |
| x512 | 7.8e-5 | 2.27 % | 0.0025 % | 0.671 / 2.2e-3 | 0.011 |
| x1024 | 3.0e-4 | 2.35 % | 0.03 % | 1.53 / 9.5e-3 | 0.022 |
| data | 1.15e-4 | 0.6 % | 0 | 6.3e-3 | 0.024 |

Up to x256, the early zero_mean dual sets the spread, which grows slowly. Between x512 and x1024, the dual is released and the level means become free. At x1024, the bias before projection (1.53) even exceeds the unconstrained one (1.33), and the projection can no longer bring mean_density below eps. The two relevant points of the trade-off are thus x64 + joint ("minimal inversions") and x512 + joint ("data-matched spread", with a level-mean variance of 0.68x the data, no inversion above 0.01 kg/m3, a mean_density below the data median and 2.3 % of inversions). The anomaly variance (1.9e-3), the roughness (3.2e-3) and the distance (0.105) are the same for all runs. The corresponding folders are `test_new_sampler_clip/combo_iso5e-6_zm0.1_eps{1.66e-1,6.66e-1,1.33,2.66}[_joint0.01]`. The next step is to run three seeds of both points at 1000 steps, which requires the multi-constraint `sampling.sample`.

Thus, eps acts as a calibration parameter of the spread in the constrained subspace. However, its value depends on the number of steps through the dual dynamics, so that it is not a property of the data alone. Besides, more spread costs a few inversions (from 2 % to 4 % between x1 and x256 at 100 steps).

### Guidance vs. post-hoc projection

The baseline is the unconstrained state followed by the same final projection (`OptimProjection`, 10 steps). On the constrained statistic itself, guidance and post-hoc projection are equivalent (`mean_density`: 8/8, distance between 0.048 and 0.055, diversity 0.055 in both cases). Indeed, these constraints amount to a few numbers per sample (one per level), which a near-uniform shift per level is able to fix.

However, they differ on what the constraint does not see. The table below gives the fraction of local density inversions at 1000 steps (0.6 % for the data, 26 % for the unconstrained model).

| constraint | post-hoc | guided | guided + final_opt |
|---|---|---|---|
| mean_density | 14.8 % | 12.7 % (sqrt eta 0.1) | 11.6 % |
| negative_grad_density | 19.7 % | 16.5 % (sqrt 0.1), 12.8 % (sqrt 0.3) | |
| zero_mean | 12.8 % | 10.3 % (raw 0.4), 9.8 % (raw 1) | |

The density_errors metric of the generation session gives the same ordering (negative_grad_density: 7.5 % post-hoc vs. 4.0 % guided). Guidance spreads the correction into the local structure of the columns through the prior of the model, whereas the post-hoc projection does not. The gap is smaller at 100 steps (zero_mean: 12.7 % vs. 10.2 %, roughly equal for mean_density). Let us note that every configuration remains 15 to 40 times above the data on this metric.

## 2 - Relaxing the constraint and the variability

The variability is better read through the variance split than through the total diversity. The unconstrained model has 5 to 6 times the variance of the data, and 64 to 68 % of it lies in the per-level spatial means (30x the data), i.e., random biases of the level means from one sample to the next. The anomaly variance is twice the data one.

We relaxed `zero_mean` (raw mode, eta 10, eps = eps_scale x q0.5, three seeds, `outputs/relax_zero_mean.png`).

| eps_scale | 0.25 | 1 | 4 | 16 | 64 | data | unconstrained |
|---|---|---|---|---|---|---|---|
| zero_mean (median) | 1.8e-3 | 1.85e-3 | 2.0e-3 | 2.7e-3 | 8.1e-3 | 2.6e-3 | 0.24 |
| satisfaction | 0 | 0.78 | 1 | 1 | 1 | | |
| level-mean variance | 2.2e-5 | 2.1e-5 | 2.4e-5 | 3.5e-5 | 1.15e-4 | 1.15e-4 | 3.4e-3 to 4.1e-3 |
| anomaly variance | 1.69e-3 | 1.69e-3 | 1.69e-3 | 1.69e-3 | 1.69e-3 | 8.9e-4 | 1.9e-3 |
| inversions | 10.7 % | 10.6 % | 10.8 % | 11.3 % | 13.3 % | 0.6 % | 25 % |

eps only acts on the variance of the subspace the constraint sees. At eps = q0.5, the constraint over-shrinks it (5x below the data). Indeed, the dual built in the first steps is not released once the constraint is satisfied, so that the samples end 2 to 4 times below eps. eps ~ 64 x q0.5 gives back exactly the data variance of the level means, at the cost of +2.5 points of inversions.

However, the anomaly variance (2x the data) does not move with eps, since no horizontal-mean constraint reaches it. `dist_isoton` does not reduce it either (1.7e-3 to 1.8e-3 in every configuration), so the excess of anomaly variance does not come from the vertical inversions. It is horizontal small-scale noise. Indeed, the horizontal roughness (mean squared difference between neighbouring ocean cells, `pd_explore.roughness`) is 1.2e-3 on the data and between 2.8e-3 and 3.1e-3 on every generation, at 100 and 1000 steps, constrained or not. None of the five losses sees it.

Consequently, the "variability per constraint" can be identified as the variance in the image of the constraint, and the trade-off of Q2 is between satisfaction and that variance. Relaxing the constraint does not give back diversity elsewhere.

## Open points

- `final_opt` uses `OptimProjection`, which optimises the whole batch until every sample is below eps, so that already satisfied samples keep moving. A per-sample mask would give a minimal projection;
- The relaxation is driven by the dual dynamics rather than by eps (early overshoot, slow release). A dual update that decays when g < 0, or a smaller eta with more steps, may make eps a more direct handle;
- Stacked constraints require a list of (loss, eta, eps) in `sampling.sample` (one dual per loss). `pd_explore.sample_pd` with `Stacked` is a minimal version, with one mode for the whole stack;
- The excess of anomaly variance (2x the data) is horizontal small-scale noise (roughness 2.5x the data), which no constraint reduces. Addressing it would require a smoothness or spectral constraint, or more sampling steps and a better model;
- Every number here is obtained on batches of size 8, mostly with one seed. Only the relaxation and the best configurations (`isoton_seeds`, `combo` grids) have three seeds.
