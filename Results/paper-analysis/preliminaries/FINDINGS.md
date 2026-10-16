# Preliminaries : eps, enforcement, relaxation (exploration of 2026-10-14)

Exploration notes for the three preliminary questions of `../analytic_approach.md`. Most numbers come from 100-step DDPM sampling, batch 8, seed 0 (3 seeds for the relaxation), model `tav0h83b`, always with the `BorderZero` projection, and the Tweedie estimate clipped to [-1, 1] as in `pipelines/sampling.py`. The 1000-step numbers come from the runs of the generation-pipeline session (`test_new_sampler_clip/`). These are explorations on small batches, not paper-grade statistics.

## Summary

- eps : the unconstrained generations are 2 to 8 orders of magnitude above the data on every loss, eps = q0.5 of the training losses is the target (`outputs/eps_data_quantiles.csv`), eps = 0 only makes sense for `negative_grad_density`.
- Enforcing : the clip of the Tweedie estimate is required. sqrt mode (g = sqrt(loss) - sqrt(eps)) beats the raw mode on every density loss. `zero_mean` is the easiest and most useful constraint, `dist_isoton` works in a narrow eta window, and a final LBFGS projection (`final_opt`) closes the remaining gap. Best pipeline : guided `dist_isoton + zero_mean`, one dual each, sqrt (1e-4, 1) at 100 steps / (5e-6, 0.1) with zero_mean eps x64 or more at 1000 steps, then a joint post-hoc projection on `mean_density + 0.01 dist_isoton` : local inversions 25 % -> 2 % (inversions > 0.01 kg/m3 ~0), no profile bias, level-mean spread close to the data.
- Guidance vs post-hoc projection : identical on the constrained statistic, but guidance also fixes the local structure the constraint does not see (fewer local inversions).
- Variability : the excess variance of the model is mostly random per-level biases (30x the data) plus a 2x excess of spatial anomalies. A constraint acts only on the variance of what it sees ; relaxing eps gives back that variance (eps ~ 64 x q0.5 reproduces the data for `zero_mean`) and nothing else. The anomaly excess is horizontal small-scale noise (roughness 2.5x the data), untouched by every constraint tried.

## Files

| file | role |
|---|---|
| `eps_data.py` | constraint losses on the 1800 training states, quantiles -> `outputs/eps_data_*` |
| `data_reference.py` | diversity / distance of random training batches -> `outputs/data_reference.json` |
| `pd_explore.py` | `sample_pd` : copy of `sampling.sample` with variants (mode raw/rel/log/sqrt, t_start, lbd_max, final_opt, through_unet, stacked losses with one dual each), trace, and the metrics `diversity`, `distance`, `variance_split`, `inversion_fraction` |
| `sweep.py` | resumable grid runner, one folder per run in `$OCEANDATA/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/` (`generated.npy`, `lbd.npy`, `trace.npz`, `metrics.json`) |
| `analyze_sweep.py` | `outputs/sweep_table.csv`, `outputs/sweep_<loss>.png` (metrics vs eta), `outputs/sweep_frontier.png` (loss vs distance / diversity) |
| `local_metrics.py` | inversion fraction, roughness and variance split for data batches and every saved run -> `outputs/local_metrics.csv` |
| `plot_relax.py` | Q2 figure `outputs/relax_<loss>.png` |
| `compare_runs.py` | local metrics + losses table for any generation folders (e.g. the 1000-step runs of `test_new_sampler_clip/`) |
| `joint_projection.py` | post-hoc projection on mean_density + c dist_isoton of saved runs -> `<run>_joint<c>/` |

The saved states are normalised (b, 37, 208, 64); `Diffusion_Model/post_processing.py --generated <path>/generated.npy` writes the physical fields. The post-hoc projections of the unconstrained runs are in `prelim_sweep_clip/posthoc_{unc100,unc1000}_<loss>/`.

## Metrics

- final losses : every constraint loss of `pipelines/losses.py` on the final states, median over the batch. All 5 are recorded for every run, so the side effects of one constraint on the others are visible.
- satisfaction : fraction of the batch with loss <= eps.
- distance : RMS over ocean cells of the difference with the unconstrained state of the same seed (normalised units). Reference : two random training states are 0.044 apart.
- diversity : mean over ocean cells of the standard deviation over the batch. Data 0.022, unconstrained 0.067.
- variance split : the variance over the batch is split into the variance of the per-level spatial means (one number per channel and sample, what the horizontal-mean constraints act on) and the variance of the spatial anomalies. Data : 1.2e-4 + 8.9e-4. Unconstrained : 3.4e-3 to 4.1e-3 + 1.9e-3.
- inversions : fraction of vertical pairs of ocean cells (strided levels) where the density decreases with depth. Data 0.6 %, unconstrained 25 % (100 steps) / 26 % (1000 steps). It is a local metric that no constraint acts on directly.
- roughness : mean squared difference between neighbouring ocean cells (normalised). Data 1.2e-3, every generation 2.8e-3 to 3.1e-3.

## 0 - eps from the data

Quantiles of the per-sample losses on the 1800 training states (`outputs/eps_data_quantiles.csv`) :

| loss | q0.5 | q0.9 | q1.0 | unconstrained generation (median) |
|---|---|---|---|---|
| mean_density | 6.3e-3 | 1.3e-2 | 2.7e-2 | 1.32 |
| mean_grad_density | 1.5e-3 | 1.9e-3 | 2.2e-3 | 0.92 |
| negative_grad_density | 0 | 0 | 0 | 0.40 |
| zero_mean | 2.6e-3 | 8.2e-3 | 1.1e-2 | 0.24 |
| dist_isoton | 3.5e-5 | 3.8e-5 | 4.3e-5 | 6.6e3 |

The unconstrained generations are 2 to 8 orders of magnitude above the data on every loss, so eps = 0 is never the right target except for `negative_grad_density` (exactly 0 on all training states). The sweep uses eps = q0.5 (floored at 1e-6 for `negative_grad_density`).

Two bugs found on the way. `IsotonicDensity` lost precision in float32 (densities ~1028 in a cumsum) and gave a fake floor of 3.6e-2 on the data : fixed in `pipelines/losses.py` by working on the anomaly to the surface value. The losses use `nansum`, so a NaN state reports a loss of 0 : the constrained runs in `test_new_sampler/` (before the clip) are all NaN and must be ignored.

## 1 - Can we enforce the constraints ?

Without the clip of the Tweedie estimate, every constrained run diverges (the loss of x0_hat is ~1e6-1e8 at the first step and the dual variable explodes). With the clip :

| loss | raw mode | sqrt mode (g = sqrt(loss) - sqrt(eps)) |
|---|---|---|
| mean_density | best eta 1e-2 : 0.28 (dist 0.044), larger eta fits the profile by T/S compensation (zero_mean 0.24 -> 9.4) | eta 1 : 1.25e-2, all samples in the data range, no T/S compensation (dist 0.062) |
| mean_grad_density | best eta 1e-2 : 0.27 | eta 1 : 1.0e-2 (dist 0.053) |
| negative_grad_density | best eta 0.1 : 0.04 | eta 1 : 4e-5, 2/8 satisfied (dist 0.040) ; eta 10 : 7/8 but dist 0.225 |
| zero_mean | eta 10 : 1.8e-3, 7/8 satisfied (dist 0.091) | eta 10 : 8/8 (5e-6, overshoot), same states |
| dist_isoton | no usable eta : the loss only drops once the state is destroyed | narrow window at eta 1e-4 : 5.2, inversions 25 % -> 5.4 % (dist 0.073) ; 1e-5 too weak, 1e-3 blows up |

- sqrt mode is the better formulation for the density-profile losses : its gradient is bounded near the constraint, so the dual does not inflate on the huge early losses of x0_hat. At 1000 steps eta transfers as eta_100 / 10 : `mean_density` sqrt eta 0.1 gives 4/8 satisfied, 8/8 in the data range, profile_rmse 0.020 (data 0.024).
- Failure mode at large eta : the per-sample dual runs away on part of the batch (lbd ~40 at 100 steps, ~185 at 1000), the state degrades (distance 0.3 to 3, diversity up to 15x the data). For `mean_density`, the profile can be matched by a T/S compensation that `zero_mean` exposes.
- The last steps pull the loss back up : on `mean_density` sqrt eta 1 the loss of x0_hat reaches 5e-3 at t = 90 and the final state ends at 1.25e-2.
- `zero_mean` is the constraint that works best and drags all the others towards the data (mean_density 1.35e-2, negative_grad_density 6.6e-5, dist_isoton 56, diversity 0.028).
- Variants : `final_opt` (LBFGS on the final state until loss < eps, `OptimProjection`) closes the gap (`mean_density` raw eta 1e-2 : 0.28 -> 3.1e-3, 8/8, distance 0.044 -> 0.055 ; 90 % of the correction is a uniform shift per level). `t_start = 700` removes the raw-mode overshoot at large eta, `t_start = 400` is too late, at small eta it only weakens the constraint. `lbd_max` does not help (a large constant lbd from the start pushes the whole trajectory). `through_unet = False` (no Jacobian of the UNet) gives the same result 25 % faster. One dual per loss for `mean_density + zero_mean` (raw) is better than `mean_density` alone in raw mode but worse than sqrt.
- `dist_isoton` (local column constraint) only works in sqrt mode in a narrow window around eta = 1e-4 (100 steps) : loss 6.6e3 -> 5.2, local inversions 25 % -> 5.4 %, the best of every run, at distance 0.073 and diversity 0.053. Without the UNet Jacobian (`through_unet = False`) it is the same (5.3 %) and 40 % faster. At 100 steps it moves the mean profile little (mean_density 0.95), but at 1000 steps (eta 5e-6, the /20 transfer) it removes the inversions partly by drifting the T/S level means (profile_rmse 2x unconstrained, zero_mean 2-7x) : it needs an anchor on the level means. Raw mode has no usable eta, and sqrt eta >= 1e-3 blows up unless the guidance starts late (`t_start = 300`, eta 1e-3 : inversions 8.4 %, distance 0.044).

### Combining a local and a mean constraint

`dist_isoton + zero_mean` with one dual each (sqrt mode for both, `pd_explore.Stacked`), 100 steps, 3 seeds :

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

- (1e-4, 1) is the best configuration of the exploration : 10x fewer inversions than unconstrained, the level-mean variance at the data level, and with `final_opt` the local inversions go below the data value. `zero_mean` is not satisfied at eps = q0.5 (7e-3), which is what keeps its level-mean variance at the data level.
- A stronger `zero_mean` dual (3 or 10) over-shrinks the level means (variance 1e-7 to 1e-6) and doubles the inversions : the two constraints compete.
- `mean_density` is a worse anchor than `zero_mean` (it constrains one T/S combination per level, the compensating direction stays free).
- The level-mean variance measures the spread across the batch, not a bias shared by the batch : (1e-4, 1) still has mean_density 0.06 (data q1.0 2.7e-2), all samples off the training profile in the same direction. Bias and spread have to be reported together (mean_density or profile_rmse next to the level-mean variance).
- Adding `mean_density` as a third constraint (sqrt, one dual each) removes the bias but costs inversions : (1e-4, 1, 0.3) gives mean_density 1.5e-3, zero_mean 3.7e-3, level-mean variance 4e-5, inversions 6 % ; (1e-4, 1, 1) gives 9 %. The three constraints compete.
- At 1000 steps (generation session, `test_new_sampler_clip/combo_*`, `compare_runs.py`) the zero_mean eta is bistable : eta 0.02-0.1 crushes the level means (variance 4e-7 to 7e-7, inversions 5-6 %), eta 0.01 lands at the data level (variance 8e-5, inversions 2.0 %, > 0.01 kg/m3 0.6 %) but with a shared profile bias (mean_density 0.215). The knob is moved to eps (zero_mean eps x16 and x64 at eta 0.1).

### Best pipeline so far : guided local + mean constraints, then a joint projection

Guiding a third constraint (`mean_density`) competes with `dist_isoton` : (1e-4, 1, 0.1) 4.6 % inversions, (1e-4, 1, 0.3) 6 %, raising the dist_isoton eta makes it worse (8-9 %). A post-hoc projection on `mean_density` alone also recreates inversions (2 % -> 8 % at 1000 steps, the uniform per-level T/S shift breaks the column stability). The joint post-hoc projection on `mean_density + 0.01 * dist_isoton` (eps 0, 10 LBFGS steps, `joint_projection.py`, proposed by the generation session) removes the bias without that cost :

| run (100 steps, 2 seeds) | level-mean variance | inversions | inversions > 0.01 | mean_density | distance |
|---|---|---|---|---|---|
| dist_isoton + zero_mean sqrt (1e-4, 1), eps q0.5 | 7.4e-5 to 8.3e-5 | 2.4-2.6 % | 0.8 % | 0.063 | 0.086-0.099 |
| + joint projection | 6.7e-5 to 7.7e-5 | 2.0-2.4 % | 0.002-0.003 % | 9e-4 | +0.001 |
| zero_mean eps x64 + joint | 3.0e-4 to 3.5e-4 | 2.9-3.2 % | 0.01 % | 3e-3 | |
| zero_mean eps x256 + joint | 8.2e-4 to 9.7e-4 | 4.0-4.5 % | 0.05 % | 7e-3 | |
| data | 1.15e-4 | 0.6 % | 0 | 6.3e-3 | 0.044 |

At 1000 steps (generation session, (5e-6, 0.1)) the zero_mean eps that gives the data level-mean variance is larger : x64 still gives 2.9e-5 (x16 7e-6), with inversions 2.0 % and, after the joint projection, 1.55 % with 0.002 % above 0.01 kg/m3 and mean_density 5.6e-4. The guided triple at 1000 steps confirms the competition (mean_density eta 0.03 / 0.1 : inversions 5.0 / 7.1 %, anomaly variance up to 3e-3). The generation session runs zero_mean eps x256 at 1000 steps (`test_new_sampler_clip/combo_iso5e-6_zm0.1_eps6.66e-1` and its `_joint0.01`) to get the data level-mean variance : zero_mean eps relaxation at 1000 steps, (5e-6, 0.1), seed 0, each followed by the joint projection (c = 0.01) :

| zero_mean eps | level-mean variance | inversions | inversions > 0.01 | mean_density before -> after joint | profile_rmse |
|---|---|---|---|---|---|
| x64 | 2.6e-5 | 1.55 % | 0.002 % | 0.111 -> 5.6e-4 | 0.007 |
| x256 | 5.0e-5 | 2.25 % | 0.006 % | 0.347 -> 1.1e-3 | 0.009 |
| x512 | 7.8e-5 | 2.27 % | 0.0025 % | 0.671 -> 2.2e-3 | 0.011 |
| x1024 | 3.0e-4 | 2.35 % | 0.03 % | 1.53 -> 9.5e-3 | 0.022 |
| data | 1.15e-4 | 0.6 % | 0 | 6.3e-3 | 0.024 |

Up to x256 the early zero_mean dual sets the spread (slow growth), between x512 and x1024 the dual is released and the level means go free : at x1024 the bias before projection (1.53) exceeds the unconstrained one (1.33) and the projection can no longer bring mean_density below eps. The two useful points of the trade-off are x64 + joint ("minimal inversions") and x512 + joint ("data-matched spread" : level-mean variance 0.68x the data, no inversion above 0.01 kg/m3, mean_density below the data median, 2.3 % inversions). Anomaly variance (1.9e-3), roughness (3.2e-3) and distance (0.105) are the same for all. Folders : `test_new_sampler_clip/combo_iso5e-6_zm0.1_eps{1.66e-1,6.66e-1,1.33,2.66}[_joint0.01]`. Next : 3 seeds of both points at 1000 steps (needs the multi-constraint `sampling.sample`).

So eps works as a calibration knob of the spread in the constrained subspace, but its value depends on the number of steps through the dual dynamics : it is not a property of the data alone. More spread costs a few inversions (2 % -> 4 % from x1 to x256 at 100 steps).

### Guidance vs post-hoc projection

The baseline is the unconstrained state followed by the same final projection (`OptimProjection`, 10 steps). On the constrained statistic itself guidance and post-hoc are equivalent (`mean_density` : 8/8, distance 0.048-0.055, diversity 0.055 in both cases). These constraints are a few numbers per sample (one per level), fixed by a near-uniform shift per level.

They differ on what the constraint does not see. Fraction of local density inversions (data 0.6 %, unconstrained 26 %), 1000 steps :

| constraint | post-hoc | guided | guided + final_opt |
|---|---|---|---|
| mean_density | 14.8 % | 12.7 % (sqrt eta 0.1) | 11.6 % |
| negative_grad_density | 19.7 % | 16.5 % (sqrt 0.1), 12.8 % (sqrt 0.3) | |
| zero_mean | 12.8 % | 10.3 % (raw 0.4), 9.8 % (raw 1) | |

The density_errors metric of the generation session gives the same ordering (negative_grad_density 7.5 % post-hoc vs 4.0 % guided). Guidance spreads the correction into the local column structure through the prior of the model, post-hoc does not. The gap is smaller at 100 steps (zero_mean 12.7 % vs 10.2 %, mean_density about equal). Every configuration stays 15-40x above the data on this metric.

## 2 - Relaxing the constraint and the variability

The variability is better read through the variance split than through the total diversity. The unconstrained model has 5-6x the variance of the data, and 64-68 % of it sits in the per-level spatial means (30x the data), i.e. random biases of the level means from one sample to the next. The anomaly variance is 2x the data.

Relaxation of `zero_mean` (raw, eta 10, eps = eps_scale x q0.5, 3 seeds, `outputs/relax_zero_mean.png`) :

| eps_scale | 0.25 | 1 | 4 | 16 | 64 | data | unconstrained |
|---|---|---|---|---|---|---|---|
| zero_mean (median) | 1.8e-3 | 1.85e-3 | 2.0e-3 | 2.7e-3 | 8.1e-3 | 2.6e-3 | 0.24 |
| satisfaction | 0 | 0.78 | 1 | 1 | 1 | | |
| level-mean variance | 2.2e-5 | 2.1e-5 | 2.4e-5 | 3.5e-5 | 1.15e-4 | 1.15e-4 | 3.4e-3 to 4.1e-3 |
| anomaly variance | 1.69e-3 | 1.69e-3 | 1.69e-3 | 1.69e-3 | 1.69e-3 | 8.9e-4 | 1.9e-3 |
| inversions | 10.7 % | 10.6 % | 10.8 % | 11.3 % | 13.3 % | 0.6 % | 25 % |

- eps only acts on the variance of the subspace the constraint sees. At eps = q0.5 the constraint over-shrinks it (5x below the data) : the dual built in the first steps is not released once the constraint is satisfied, so the samples end 2-4x below eps. eps ~ 64 x q0.5 gives back exactly the data variance of the level means, for +2.5 points of inversions.
- The anomaly variance (2x the data) does not move with eps : no horizontal-mean constraint reaches it. `dist_isoton` does not reduce it either (1.7e-3 to 1.8e-3 in every configuration), so the excess anomaly variance is not the vertical inversions. It is horizontal small-scale noise : the horizontal roughness (mean squared difference between neighbouring ocean cells, `pd_explore.roughness`) is 1.2e-3 on the data and 2.8e-3 to 3.1e-3 on every generation, at 100 and 1000 steps, constrained or not. None of the 5 losses sees it.
- So "variability per constraint" can be identified as the variance in the image of the constraint, and the trade-off of Q2 is between satisfaction and that variance. Relaxing the constraint does not give back diversity elsewhere.

## Open points

- `final_opt` uses `OptimProjection`, which optimises the whole batch until every sample is below eps : samples already satisfied keep moving. A per-sample mask would give a minimal projection.
- The relaxation is driven by the dual dynamics more than by eps (early overshoot, slow release). A dual update that decays when g < 0, or a smaller eta with more steps, may make eps a more direct handle.
- Stacked constraints need a list of (loss, eta, eps) in `sampling.sample` (one dual per loss) : `pd_explore.sample_pd` with `Stacked` is a minimal version (one mode for the whole stack).
- The excess anomaly variance (2x the data) is horizontal small-scale noise (roughness 2.5x the data) untouched by every constraint : a smoothness / spectral constraint, or more sampling steps / a better model, would be needed for that part.
- Every number here is batch 8, mostly one seed. The relaxation and the best configurations (`isoton_seeds`, `combo` grids) have 3 seeds.
