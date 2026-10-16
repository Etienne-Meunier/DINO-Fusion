# Density error: proposal

## Short answer

Yes, but the evidence points away from a better per-cell density loss. Most of the density error comes from the horizontal-mean (level-mean) profile, not from individual cells. A constraint on the profile, `zero_mean` combined with `dist_isoton`, already brings the eval density error below the data level. NGD fails for a structural reason. It removes inversions of the mean profile but leaves the profile close to neutral, and the generated horizontal anomalies, about 2x those of the data, then flip it locally. A loss "closer to the metric" therefore has to restore stratification, not only forbid g < 0. Our candidate for that (option 2) has not been tested yet.

A correction to earlier numbers: the "inversions" column of FINDINGS_short is not the eval metric. It is an unweighted fraction of vertical pairs on the 18 strided levels. The eval metric E (`compute_density_error`, % of ocean volume with rho[k+1] - rho[k] < 0, float64) gives much lower values: data 0.36 % (not 0.63 %), unconstrained 9.28 %, NGD sqrt 0.3 3.57 %, iso + zm x64 + joint 0.23 % (not 1.55 %). All numbers below use E, from `RESULTS.md` (8 samples per run, one seed, 1000 steps).

## Evidence

| run | E (%) | E with g <= -1e-3 (%) | dist to unconstrained | level-mean var | roughness |
|---|---|---|---|---|---|
| data | 0.36 | 0 | 0.044 (data-data) | 1.15e-4 | 1.2e-3 |
| unconstrained | 9.28 | 8.64 | 0 | 3.9e-3 | 3.1e-3 |
| NGD sqrt eta 0.3 | 3.57 | 3.21 | 0.081 | 2.8e-3 | 3.0e-3 |
| dist_isoton alone | 1.71 | 1.36 | 0.143 | 1.9e-3 | 2.9e-3 |
| zero_mean alone | 1.11 | 1.02 | 0.099 | 9.9e-6 | 3.4e-3 |
| iso + zm, zm eps 1.33 (no projection) | 0.23 +/- 0.03 | 0.16 | 0.106 | 9.4e-5 | 3.2e-3 |
| iso + zm x64 + joint 0.01 | 0.23 +/- 0.03 | 0.03 | 0.105 | 2.6e-5 | 3.3e-3 |

- Where the error lives. Unconstrained samples keep 7.56 % of E when only their level-mean profile is kept, and drop to 1.29 % when their anomalies are put on the data profile. So 86 % of E comes from the profile, and most of it sits in the deep levels (25-35).
- Why NGD falls short. 90 % of its E has |g| >= 1e-3, so the residual is real inversions, not ties or float32 artefacts (float32 vs float64 changes E by at most 12 %). Its own profile has only 0.81 % inversions, yet E is 3.57 %: the profile is close to neutral and the anomalies flip it.
- Anomalies are about 2x the data's in every guided run. On the data profile they give 1.2-1.3 % against 0.69 % for the data's own anomalies, which matches the 2.7x roughness. No density loss changes this.
- iso + zm eps 1.33 reaches E 0.23 % because its profile is somewhat more stratified than the data mean, which hides the excess anomaly inversions. The joint projection gives the same E but flattens the level means to 4x below the data variance, so it is not worth it.

## Option 1 (recommended): iso + zm guided, zm eps 1.33, no joint projection

No new code. Settings: guided sqrt mode, dist_isoton eta 5e-6, zero_mean eta 0.1 with eps 1.33 (about 8x the data median), as in run `test_new_sampler_clip/combo_iso5e-6_zm0.1_eps1.33`. It gives E 0.23 % (data 0.36 %) and level-mean variance 9.4e-5 (data 1.15e-4), with no post-hoc step.

Limits:
- It is not a density loss, and E sits below the data level for a partly artificial reason (the profile is over-stratified).
- It is one seed of 8 samples, so it is within noise of the neighbouring eps values.
- Roughness and anomaly variance stay 2-2.7x the data.

## Option 2 (to answer "a density loss closer to the metric"): volume-weighted NGD with margin

L = sum_cells w * tau * softplus((m - g) / tau) / sum_cells w, with g = rho[k+1] - rho[k] and w = the cell volume used by the metric. It differs from NGD in two ways:
- The volume weighting matches the metric. Thick deep cells, where most of E is, are currently under-weighted.
- The margin m > 0 pushes the profile away from neutral instead of stopping at g = 0. m is to be set from the data (low quantile of g on stable cells at levels >= 10, capped at 1e-3; not computed yet).

The point is to test whether a density-only loss can reach option 1's E at the same distance. NGD has never been pushed to distance about 0.10, so this comparison is still open.

## Not shown yet

- Seed robustness of option 1. The ranking between eps values and between option 1 and the joint projection is within noise.
- Whether a margin or volume weighting fixes NGD (option 2 has no evidence yet).
- Anything about the roughness gap, which drives the anomaly part of E and is outside the scope of density losses.
- DATA E (0.36 %) is measured after the stride and interpolation pipeline, so it is "data as the model represents it".

## Next runs (all need new diffusion sampling, so your approval)

1. Option 1 confirmation: `combo_iso5e-6_zm0.1_eps1.33`, 3 seeds x 8 samples, 1000 steps. Report E, E for g <= -1e-3, distance, level-mean variance, the profile/anomaly split and roughness, against the x64 + joint run.
2. Option 2: volume-weighted NGD with margin, sqrt mode, 2 eta values chosen to land at distance about 0.10, alone and with zero_mean (eps 1.33), 2 seeds x 8 samples, 1000 steps. The same metrics, compared with NGD sqrt 0.3 and option 1.

Run 1 is cheap and decides the recommendation. Run 2 is only needed if you want a density-only constraint.
