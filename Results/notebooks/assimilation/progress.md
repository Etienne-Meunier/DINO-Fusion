# Assimilation study: progress of phases 0 and 1 (handoff for phase 2)

Status as of 2026-10-29. The plan is in `plan.md`, next to this file. Nothing in `Results/notebooks/assimilation/` is committed yet (the folder is untracked).

## Question

The a posteriori NEMO integration wrecks generated states that are not stably stratified. If we generate states conditioned on an observation (here SST), does the assimilated signal survive the integration only when a stratification constraint is applied during sampling? Phase 0 picked the conditioning signal and the diagnostic. Phase 1 generated the conditioned states. Phase 2 integrates them with NEMO.

## Working conventions with the user

- Scripts are written by a coder subagent, then checked by a reviewer subagent. The main agent briefs, relays and does not code itself.
- On figure iterations, report the changes and the numbers. Do not interpret the plots unless asked.
- Math in plain ASCII in chat (the terminal doesn't render LaTeX).
- Code style: a module docstring with purpose, inputs and outputs; function docstrings near-systematic; inline comments only for facts not recoverable from the code plus one hop; never narration comments.
- Never add Claude attribution to commits. Don't commit unless asked.
- `Results/a_posteriori/dinostics` (submodule) shows as modified. That predates this work; don't touch it.
- BorderZero: the general project rule is "every sampling run uses BorderZero". This study is an explicit exception: `none` has no projection at all (see phase 1).

## Data facts

- Training tar: `TrainingConfig().data_file` = `$OCEANDATA/Dino-Fusion/dino_1_4_degree_coarse_240125.tar`. It holds `NNNNN.{toce,soce,ssh}.npy` for N = 0..1799, time-ordered, plus `infos/{mean,std,mask,min,max}.<field>.npy`. Locally `OCEANDATA=/Volumes/LoCe/oceandata/`.
- 36 states per year (10-day spacing, 360-day calendar), so 1800 states = 50 years. The model drifts over the run, so the analysis uses only the last 5 years (states 1620..1799). Phase = index mod 36.
- The calendar date of state 0 is unknown. Seasons were defined from the data instead: NH summer = phase 24 (max NH mean SST), NH winter = phase 7 (min).
- File grid (199, 62) with 36 levels. Columns 0 and -1 are periodic halo columns; drop them in statistics.
- The model sees T and S at levels `slice(0, -1, 2)`: 18 levels (0, 2, ..., 34) at depths 5, 26, 49, 74, 102, 136, 178, 231 m, ... A generated sample is a normalised tensor (b, 37, 208, 64): T channels 0:18, S 18:36, SSH 36, padded (rows 5/4, cols 1/1), land 0. Get physical T with `Observer('data', TrainingConfig(), device).unnormalized(x, result='strided')['toce']` -> (b, 18, 199, 62) degC with land NaN. `seasonal.generated_toce(x, observer)` wraps this.

## Phase 0 results (data, last 5 years)

- SSH has almost no seasonal cycle. The median annual amplitude is 0.6 cm, against a noise floor of 0.53 cm (the floor is the amplitude at non-annual periods of 45 and 30 states). The spatial std of the mean SSH is 61 cm and the non-seasonal residual is 3.6 cm. Only 50-70N is clear (about 1.5 cm). The domain-mean SSH is conserved. SSH was dropped as the conditioning signal.
- SST has a clean NH cycle north of 15N: phase coherence 0.98, amplitude about 1.1 degC at 50-70N, peak at phase 24. The SH and the tropics have almost no SST seasonality. SST is the conditioning signal.
- Depth reach of the seasonal T signal, NH band 15-70N:

| depth (m) | median amplitude (degC) | noise floor | band-mean anomaly, summer / winter |
|---|---|---|---|
| 5 | 0.425 | 0.054 | +0.55 / -0.54 |
| 49 | 0.420 | 0.056 | +0.50 / -0.50 |
| 74 | 0.319 | 0.072 | +0.28 / -0.33 |
| 102 | 0.125 | 0.088 | +0.05 / -0.09 |
| 136 | 0.089 | 0.092 | +0.02 / -0.03 |

  The signal reaches the floor at 136 m. The phase of the maximum lags with depth, from 24.3 at the surface to 29.8 at 136 m (about 55 days). So a seasonal response is only expected in the first 5 model T channels (5-102 m).
- The TS constraint (`zero_mean`) does not conflict with a seasonal SST. It pins the sum over all 37 channels of the squared unweighted level mean (normalised units), with `loss <= eps = 0.01`. The seasonal domain-mean T swing is +/-0.18 degC at 0-50 m, which adds about 3e-4 to the loss. Data from every season gives a loss of 6.4e-3 to 9.9e-3, so the budget is close to full on data, mostly because of the deep T/S level means.
- The diagnostic for generated states: `seasonal.band_profile(toce, w)`, the area-weighted NH-band (15-70N) mean T profile, halo excluded. Report it as an anomaly relative to the 5-year mean training band profile. "Split" = summer-conditioned minus winter-conditioned anomaly at the same depth: data split 0.99 degC at 49 m and 0.60 at 74 m.

## Phase 1 setup

- Source states (from the last year of the tar): summer 1788 (phase 24) and winter 1771 (phase 7). The whole batch of a run shares the same source SST.
- SST injection, "NL" (noisy replacement), the same idea as noisy land:
  - at each step, the ocean cells of channel 0 are overwritten with `scheduler.add_noise(sst0, z, t_next)`, the noise level of the new state;
  - the initial noise gets the same overwrite;
  - the clean SST is written after the last step;
  - z comes from a separate `torch.Generator` (seed + 2**32), so the trajectory noise equals that of a run without injection.

  The final SST equals the source to 1.4e-6 degC in every run.
- Two run sets exist:
  - `phase1_standard/`. This is the one to use: it is the setup the user asked for.
    - Sampler: the standard DDPM `sampling.sample`.
    - Configs:
      - `none` = no constraint, no projection;
      - `B0` = no constraint + per-step BorderZero;
      - `C7` = `negative_grad_density_huber` eta 30, eps 8.6e-8 + per-step BorderZero;
      - `TS` = `zero_mean` eta 0.1, eps 0.01 + per-step BorderZero.

      All use lbd0 1, mode raw, volume_weighted false, which are the paper `optimal` parameters.
    - Batch 16, seed 0, 1000 steps.
    - Members are exactly paired with the unconditioned paper runs `optimal/none`, `optimal/none_border0`, `optimal/negative_grad_density_huber_border0_30` and `optimal/zero_mean_border0` (plain sampling reproduces them with max abs diff 0).
  - `phase1/` (centred sampler from `Results/notebooks/machine/surface_bias/experiments/exp.py`: centred noise + noisy land + BorderZero only at the end; none / C7 eta 10 / TS eta 1). This was the wrong setup. It is kept for the record and not member-paired with its references.

## Phase 1 results (`phase1_standard`)

E = % of ocean volume where sigma_0 decreases with depth (`metrics.compute_density_error`). mInv = members whose horizontal-mean density profile has an inversion. The data last year gives E 0.354% and mInv 0/36. T columns are the NH-band T anomaly in degC, summer / winter.

| config | E % | mInv | T 49 m | split 49 m | T 74 m | split 74 m | fig1 section RMSE to source |
|---|---|---|---|---|---|---|---|
| data | 0.35 | 0/36 | +0.49 / -0.50 | 0.99 | +0.28 / -0.32 | 0.60 | |
| none | 23.6 / 23.6 | 16/16 | +4.90 / +4.41 | 0.49 | +7.72 / +7.41 | 0.31 | 1.10 / 1.12 |
| B0 | 7.76 / 7.83 | 16/16 | +0.74 / +0.12 | 0.62 | +1.75 / +1.33 | 0.42 | 0.255 / 0.256 |
| C7 | 1.89 / 1.95 | 8/16 | +1.17 / +0.46 | 0.71 | +0.54 / +0.05 | 0.49 | 0.19 / 0.19 |
| TS | 1.11 / 1.13 | 2/16, 4/16 | +0.14 / -0.53 | 0.67 | +0.11 / -0.33 | 0.44 | 0.085 / 0.079 |

Unconditioned references at 49 m: none +5.10, none_border0 -0.23, C7 +2.09, TS -0.25. At 102 m the conditioned runs show no seasonal split: values range from -0.90 to +1.02 against data +0.05 / -0.07.

## Code (`Results/notebooks/assimilation/`)

| file | role |
|---|---|
| `seasonal.py` | Shared helpers: tar reading, mask, harmonic fit, noise floor, `band_weights` / `band_profile` / `generated_toce`, provenance `save` |
| `0_seasonal_snapshots.py` | Summer/winter SSH and T maps for the last two years |
| `0_ssh_seasonality.py` | SSH seasonality (domain mean, Hovmoller, composite, harmonic) |
| `0_deep_t_seasonality.py` | T seasonality vs depth and the band-profile reference |
| `phase1.py` | Phase 1 helpers: sources, `normalised_state`, `ocean_mask`, `RUN_SETS` |
| `1_generate_sst.py` | SST-conditioned sampling, `--sampler {centred,standard}`, `--projection`, loss flags |
| `1_queue.sh`, `1_queue_standard.sh` | Run queues (skip finished runs) |
| `1_profiles.py` | Density/T profile figures, E table, band T profile, SST check; `--set {centred,standard}` |
| `1_density_sections.py` | Paper-fig1-style zonal sigma_0 sections; `--set` |

Python: `/Users/emeunier/miniforge3/envs/diffusion/bin/python`. Generation runs locally on MPS: about 4 min unconstrained, 10 min TS, 15 min C7, at batch 16.

Outputs are in `$OCEANDATA/models/dino-fusion/tav0h83b/inference/assimilation/`:
- `phase0/`: figures and caches (`toce_levels_1620_1799.npz`, `ssh_sst_1620_1799.npz`).
- `phase1_standard/<season>_<config>/`: `generated.npy`, `args.json`, plus `lbd.npy` and `trace.npy` for constrained runs.
- Figures in `phase1_standard/` and `phase1_standard/fig1/`:
  - `density_profiles(_top)`, `density_profiles_seasons(_nh)`, `temperature_profiles_seasons`;
  - `band_t_profile`, `sst_check`;
  - `fig1_phase1_{SH,NH,median}`.

## Phase 2: what is ready and what is open

Goal, from `plan.md`: start NEMO from the conditioned states at the right season, integrate about 2 weeks, and check whether the assimilated SST signal survives. The hypothesis is that it does with C7/TS and is destroyed for none.

Ready to reuse:
- The a posteriori pipeline in `Results/notebooks/a_posteriori/` (steps `0_data_batch` to `4_collect_run`, `settings.py` with the `EXPERIMENTS` and restart `PROTOCOLS`, Jean-Zay paths). The `minimal` protocol restarts from T, S and SSH with velocities at rest, which is itself a perturbation. The `data` run of that pipeline measures it, and is the right control. Existing experiments include `daily_2m_minimal` (2 months, daily 3D means), which would cover a 2-week horizon.
- Paired unconditioned generations for every config, so an unconditioned control integration can be compared member for member.

Open points to settle with the user:
- **Start date.** The calendar date of the source states in the NEMO forcing is unknown; only the phase is known. The NEMO restart needs a date consistent with the forcing, so the date of state 1788 / 1771 has to be found (e.g. from the original run's output timestamps).
- **Reference trajectory.** Integrate the true source state from the same start (same protocol), so each run is scored against the truth, not just against "looks stratified". The data states that follow the source in the tar (1789+, 1772+) are a cheap first check on the true evolution.
- **Members per config.** The existing a posteriori steps select a median member (`1_select_median.py`). Decide whether to integrate one member per config or several.
- **Metrics after 2 weeks.**
  - The NH-band T anomaly at 5-102 m and the split, kept vs lost.
  - E and the horizontal-mean density profile.
  - The distance to the reference trajectory.
  - Convective adjustment acts within hours to days, so 2 weeks should show it.
- **Configs to integrate.** Likely none, B0, C7, TS for both seasons, plus the data source states and the unconditioned paired members as controls.

Known caveats:
- The TS dual variable ends higher with the SST fixed than without (seen on the centred set: 2.9 vs 1.34), so the constraints work harder under conditioning.
- In TS the 49 m summer anomaly is +0.14 against data +0.49, a possible flattening by the level-mean constraint. This is a hypothesis, untested.
- The phase 1 standard code changes since the last review (B0, `--set`, the standard sampler) have not been through a reviewer yet.
