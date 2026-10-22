# Why are unconstrained states not centered on the data? Investigation summary

Night of 2026-10-19. A supervisor and two agents ran two rounds: lead-generation (scripts in `generation/`) and lead-theory (scripts in `theory/`). No existing result or repo file was modified.

## Short answer

There are two separate problems, and the paper currently conflates them.

1. The large ensemble-mean offsets of the current runs (e.g. -6.7 C at 230 m) come from a stale checkpoint. The local `tav0h83b/unet` holds the epoch ~3650 weights; the final epoch-4950 weights were never copied to this machine. The old centered 200-state run was generated elsewhere (most likely the cluster) with the final weights. Every local run since Jan 29 2025, including all of `optimal/*`, uses the epoch-3650 weights.
2. Independently of the checkpoint, the model does not control the per-channel spatial mean (the "DC" mode). Every generated state carries a constant offset over the whole channel with std ~0.3 in normalized units, i.e. 30 to 1000x the data spread of the level means. This holds in the old run too, which is centered only on average over 200 states. The mechanism is architectural plus the training objective: the final GroupNorm before the eps output erases the spatial mean, and iid noise gives that mode a weight of ~1/N in the eps loss. This is the known issue that "offset noise" addresses; it is not the Lin et al. terminal-SNR issue.

So, to the question "does the diffusion model fail to learn the mean?": yes, per state, in both checkpoints. The ensemble-mean bias, however, is the checkpoint. The zero-mean constraint works so well because it pins exactly the 36 directions the network does not control.

## Hypotheses, ranked

### H1. Stale checkpoint (epoch ~3650) explains current vs old run. Confirmed (~90%)

Evidence for:
- **Checkpoint file dates.** `unet/diffusion_pytorch_model.safetensors` (md5 bfb9f376..., also inside `tav0h83b.zip`) is timestamped 2025-01-29 18:05, the same minute as `epoch_3650.npy`. `epoch_3700` to `epoch_4950` were synced later without a new unet. `train.py` saves the model at each generation epoch, and `use_ema=False`.
- **Reproducing a training sample.** Using `test_step`'s exact settings (CPU generator seed 0, batch 8, 1000 steps), the local unet reproduces `epoch_3650.npy` best of all 100 epochs: corr 0.957, rms diff 0.132. Against `epoch_4950.npy` the corr is 0.870 and the rms diff 0.220. The match is not bit-exact, which is attributed to mps vs cluster GPU over 1000 steps.
- **Bias indicator.** The indicator is the mean off-diagonal correlation of (state - data climatology) across states. It is 0.30 to 0.37 for `epoch_3500-3750`, the local repro and every local run, but 0.025 for `epoch_4950` and 0.030 for old200. During training it decays steadily: 0.7 at epochs 500-2000, 0.3 at 3500, 0.10 at 4500, 0.027 at 4950.
- **Offset pattern.** The ensemble-mean offset field of `optimal/none` correlates 0.88 with the `epoch_3650` samples, 0.19 with `epoch_4950` and 0.38 with old200.
- **Provenance of the centered run.** The only centered unprojected runs (old200 `infesteps_1000/constraints_no_constraints/20250203-141645.npy` and its 8-state companion `20250203-140245`) have file times of Feb 3 2025 18:25, which do not match their names. This means they were copied in from elsewhere. Local runs have file times equal to their names.
- **Not a small-sample effect.** Over 5000 random 16-state subsets of old200, max|z| has median 2.45 and p99 5.15, and the RMS offset has median 1.0 C and max 2.8 C. The current run has max|z| = 11.1 and an RMS offset of 3.07 C, which no subset reaches.

Missing piece: the final weights are on the training cluster, presumably at `diffModel_experiences/tav0h83b/unet` (SLURM partition "hard" or Jean Zay). The decisive test is to fetch them, check the md5, and regenerate 16 unconstrained states with `sampling.py`. Expected result: indicator ~0.03 and |z| < ~2.5.

### H2. The model does not control the per-state spatial mean (DC mode). Established, present in both checkpoints

Evidence:
- **One constant offset per channel.** About 97.5 to 98% of the per-state level-mean error variance is a single constant offset over the whole channel, land included. The ocean level-mean offset tracks the land/padding mean with corr 0.99 and slope 1.05. The physical spread divided by 3*sigma_k is ~0.3 at every level, so the depth structure (large in the upper 500 m, small at depth) is just the normalization scale `x_n = (x - mu_k)/(3 sigma_k)`.
- **Same spread in old and current runs.** The per-state spread of the level means is ~160 to 260x the data std in every unprojected run, old200 included. For example at 26 m old200 has std 6.9 C against 0.2 C in the data.
- **Probe: the denoiser is blind to a uniform offset.** Adding a uniform offset delta to an in-distribution x_t, the eps prediction absorbs only 1.1 to 1.6% of it at every t from 10 to 990. An ideal denoiser would absorb ~100% for t > ~200. If the offset is on ocean only (land untouched, as with border0), it absorbs ~10 to 11%: the network sees the coastline step, not the mean.
- **Hooks localize the loss to the output GroupNorm.** Following a uniform input offset of 3.6e-2 through the network, the channel-mean change is 1.4e-2 after `conv_in`, 9.0e-3 before `conv_norm_out`, 8.9e-4 after `conv_norm_out` (a 10x drop), and 2.0e-3 in the eps prediction. Architecture: diffusers UNet2DModel, GroupNorm everywhere (32 groups, 2 to 4 channels per group), no global pooling, attention only at 26x8 behind a GroupNorm, output head `GroupNorm -> SiLU -> conv3x3`.
- **A 1-D model of the mean reproduces the spreads.** Feeding the measured absorption k into the DDPM recursion of the spatial mean:

  | absorption k | predicted spread of the mean | observed |
  |---|---|---|
  | 0.013 | 0.63 | 0.13 to 0.41 |
  | 0.11 (ocean-only / border0) | 0.13 | 0.044 to 0.094 |
  | Bayes-optimal | 0.003 | 0.003 (data) |

  The predicted 5x reduction from k = 0.013 to k = 0.11 matches the observed 4.5x RMS reduction of boundary projection (3.19 to 0.71 C). The spread comes from the noise injected at intermediate steps, not from x_T: its correlation with the spatial mean of x_T is -0.01.
- **Origin of the common shift within a checkpoint.** It is a small systematic bias of the eps spatial mean (~0.0035 RMS over channels on data, t >= 300). Because the mean is never pulled back, this bias integrates into a coherent shift. Fed through the 1-D model, it predicts the sign of the current offset in 29/37 channels (p ~ 4e-4). The bias differs between checkpoints, and there is no EMA to damp it, which links H2 to H1.

Literature: this is the known "offset noise" problem. Iid noise has a spatial-mean std of only 1/sqrt(N), so the eps loss gives this mode a weight of ~1/N (Guttenberg 2023, CrossLabs blog; Kutsuna 2024, arXiv:2412.03134). It connects to the spectral view in which the lowest frequencies are fixed at the highest noise levels (Dieleman 2024; Falck et al. 2025, arXiv:2505.11278). EMA strongly affects diffusion checkpoints (Karras et al., CVPR 2024, EDM2). Exposure bias (Ning et al., ICML 2023 and ICLR 2024) fits loosely: offsets accumulate during sampling and nothing corrects them. No paper was found that reports this per-sample DC error for diffusion-generated climate or ocean fields.

### Ruled out
- **Device (mps vs CPU).** UNet forward rel. diff ~1e-6, `scheduler.step` diff exactly 0, mps RNG is iid and Gaussian, and a different noise stream (CPU generator) gives the same bias.
- **Sampler code.** `sampling.py` reproduces `optimal/none` bit-exactly. The old `pipeline_tensor` loop is the same DDPM, 1000 steps, with no extra clamping. Local runs are bit-reproducible since Jan 29 2025.
- **Number of steps / leading spacing.** 1000 steps skip nothing, and 100-step runs start at t = 990 with alpha_bar 2e-4.
- **Non-zero terminal SNR (Lin et al., WACV 2024).** alpha_bar_T = 2.4e-9. A 0.3 offset leaks 1.5e-5 into x_T, against a noise-mean std of 8.7e-3.
- **clip_sample bias.** It shifts level means by < 0.004 C, and only 0.1% of the data lies outside ±1.
- **Land/padding contaminating level means.** They are masked out identically in all runs. They are, however, the best diagnostic of the DC offset.
- **The border projection difference between old and current runs.** The old run had no projection.

## Impact on the paper

- **All `optimal/*` runs, constrained ones included, use the epoch-3650 weights.** They should be regenerated with the final weights once those are retrieved, before Table 1 is final. A cheap first step is 16 unconstrained states with the final weights, to confirm H1.
- **Expected with final weights (old200 as proxy).** The unconstrained ensemble means move onto the data (DW T from 5.97 to ~5.5 C). The ± spreads and the density error stay large or grow (DW T ±1.0, density error 32 ± 12%). The 16-member RMS ensemble-mean offset drops from 3.07 to ~1.0 C, still ~10x the zero-mean constraint (0.11 C).
- **Narrative.** The story moves from "biased mean" to "a huge per-state spread of the spatial mean, which constraints fix". This arguably makes a stronger case for constraints. The ranking at the top (zero mean, then increasing density, then boundary projection, then none) should hold, since those constraints act on the spread, which does not depend on the checkpoint.
- **Mean density row.** This is the most likely to change rank. Its cold/fresh shift is density-compensated (26 m: dT = -4.6 C, dS = -1.17 g/kg, but d(rho) = 0.05 kg/m3), and its size depends on the checkpoint's eps-mean bias. The free compensated direction remains, so it is still worth a sentence as a limit of constraining density alone.
- **The A11 sentence about the unconstrained mean needs rewriting.** The per-state spatial mean is not learned (an architectural and loss-weighting issue). The ensemble mean is centered with the final weights, and the earlier offset came from an intermediate checkpoint.
- **Possible remedies to mention or test.** Offset-noise fine-tuning (re-run the probe in `theory/s11_dc_response.py`; k should rise toward 1), EMA weights, and replacing the output GroupNorm. The paper's mean-pinning constraints are already the sampling-time fix.
- **Housekeeping.**
  - `test_new_sampler/none` is bit-identical to the border-projected run, i.e. mislabelled.
  - `optimal/none` and `default_eps/none` are the same file.
  - `infesteps_1000/constraints_no_constraints/20250203-171119.npy` duplicates `20250203-141645.npy`.

## Key files
- `generation/run_stats.csv` and `run_stats.py`: per-run table of offsets, z and spread.
- `generation/bias_indicator.py`: indicator and repro of `epoch_3650`.
- `generation/unet_cpu_vs_mps.py`, `step_cpu_vs_mps.py`, `rng_test*.py`: device checks.
- `theory/s4_subsets.py`, `s5_epochs.py`, `s6_bias_pattern.py`: subset statistics and checkpoint evidence.
- `theory/s8_dc_mode.py`, `s11_dc_response.py`, `s13_dc_ocean_only.py`, `s16_*`: DC-mode diagnostics and hooks.
- `theory/s12_dc_walk.py`, `s14_eps_dc_bias.py`: 1-D model of the mean, and the sign of the shift.
- `theory/s17_*`: Table 1 recomputation with old200 as proxy.
