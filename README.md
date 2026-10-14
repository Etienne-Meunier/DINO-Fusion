# DINO Fusion

Code of *Physically Consistent Sampling For Ocean Model Initialization* (Tackling Climate Change with Machine Learning workshop, NeurIPS 2025).

A diffusion model is trained on post spin-up states (temperature, salinity, SSH) of the DINO configuration of NEMO, then sampled with or without physical constraints, and the generated states are evaluated before (a priori) and after (a posteriori) reintroduction in NEMO.

![](images/main.png)

## Layout

```
Diffusion_Model/          training and generation
  DataLoader.py, TransformFields.py   data loading and invertible transformation (stride, concat, normalise, mask, pad)
  train.py                train the diffusion model (config in configs/base_config.py)
  generate_constrained.py sample states with the constraints of pipelines/new_constraint.py
  pipelines/              State.py (observation operators : density, profiles), new_constraint.py (losses, projections),
                          constrained_sampling.py (sampling loops), constraints.py (constraints of the paper)
  generate_images.py      sample states with the constraints of the paper (pipelines/constraints.py)
  post_processing.py      un-normalise generated states -> <name>_clean/{toce,soce,ssh}.npy
Results/
  analysis_scripts/       shared code : EOS density and box metrics (metrics.py), mesh mask, mean density profile of the training set
  a_priori/               metrics on the generated states (paper Figures 2-3, Tables 1-2)
  a_posteriori/           generated state -> NEMO restart, NEMO run, density after integration (see its README)
scripts/xarray_numpy.py   build the training archive from DINO outputs
```

## Data and model

All paths are relative to the `OCEANDATA` environment variable, which must be set before running anything :

```bash
export OCEANDATA=/Volumes/LoCe/oceandata/
```

| What | Path in `$OCEANDATA` |
| --- | --- |
| Training archive (1800 states, T/S (36, 199, 62) and SSH, 1° DINO, with normalisation infos) | `Dino-Fusion/dino_1_4_degree_coarse_240125.tar` |
| Model of the paper (`3-std` normalisation) | `models/dino-fusion/tav0h83b/` |
| Generated states of the paper (C1, C2, TS) | `models/dino-fusion/Publication-Blandine/generated_npy_files/` |
| Generated states without constraint (8 and 200 states) | `models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_no_constraints/{20250131-110120,20250203-141645}_clean` |

Environment : `Diffusion_Model/diffusion-env.yml` (+ `accelerate`, `tensordict`, `einops`, `xarray`, `cmocean`, `cartopy`, `seaborn`, `pandas` for the analysis).

## 1. Train

```bash
cd Diffusion_Model
accelerate launch train.py          # or sbatch job_script_JZ.sh on Jean-Zay
```

Hyper-parameters are in `configs/base_config.py` (`TrainingConfig`). Every `generation_frequency` epochs the model is saved in `base_output_dir/<wandb id>/` with sample generations.

## 2. Generate

### Constraints framework (`generate_constrained.py`)

```bash
python Diffusion_Model/generate_constrained.py --model_path $OCEANDATA/models/dino-fusion/tav0h83b --batch 8 --method <method> [options]
```

| `--method` | Algorithm | Options |
| --- | --- | --- |
| `none` | DDPM sampling | |
| `gradient` | `x <- DDPM(x) - beta * grad C(x)`, beta updated with `--eta` (`--beta_type eq`); with `--tweedie` the loss is evaluated on the Tweedie estimate `x0_hat` returned by the scheduler instead of `x` | `--loss mean_density` (mean density profile), `mean_grad_density` (mean vertical density gradient), `negative_grad_density` (penalise inversions of the mean density profile), `zero_mean` (zero spatial mean of each normalised level), `dist_isoton` (squared distance of each density column to the isotonic cone), `--beta`, `--eta` |
| `projection` | augmented Lagrangian : after the DDPM step, `z = P(g(x) + lambda / (2 rho))` and `x <- x - tau * grad_x [rho E[(g(x) - z)^2] + lambda . E[g(x) - z]]`, dual ascent on lambda and rho | `--projection isotonic` (density increasing with depth in each column), `mean_profile` (mean density profile of the training set), `--tau`, `--eta`, `--eta_rho`, `--rho_min` |

`g` is the density on the strided levels (`Observer.density`). Both constrained methods apply `BorderZero` on the state after each step. The output folder (`--output`, default `<model_path>/inference/<method>_<option>_<time>/`) contains `generated.npy` (normalised), `metrics.csv` (dual variables along sampling), `args.json` and `clean/{toce,soce,ssh}.npy` (un-normalised, input of the a priori evaluation).

New losses / projections are added in `pipelines/new_constraint.py` : a `Loss` maps the normalised state to a scalar (gradient method), a `Projection` maps the observation `g(x)` to the closest point of the constraint set (projection method).

### Constraints of the paper (`generate_images.py`)

```bash
python Diffusion_Model/generate_images.py --model_path $OCEANDATA/models/dino-fusion/tav0h83b --batch 8 --beta 0 --constraints <constraints>
```

| Paper | `--constraints` |
| --- | --- |
| No constraint | *(none)* |
| Constraint TS | `border_zero gradient_zero_mean` |
| Constraint C1 (mean density profile) | `gradient_zero_mean_density` |
| Constraint C2 (mean density gradient profile) | `mean_gradient_density` |

`--beta` is the initial value of the dual variable (0 in the paper). The output (normalised, (b, 37, 208, 64)) is saved in `<model_path>/inference/infesteps_<n>/constraints_<...>/beta_<beta>_<time>.npy`. Then un-normalise it :

```bash
python Diffusion_Model/post_processing.py --generated <path>.npy
```

Notes :

- C1 and C2 build the density with a mask repeated over a batch of 8 : use `--batch 8`.
- The dual step size $\eta$ is set in the `Beta.update` calls of `pipelines/constraints.py` (0.001 for C1, 0.0001 for C2, 0.001 for TS).
- The target profile of C1 and C2 is `Results/analysis_scripts/mean_density_train.txt`, computed by `python Results/analysis_scripts/metrics.py`.

## 3. A priori evaluation

Run from `Results/a_priori/` with `--set paper` (generations of the paper, default) or `--set new` (generations of `generate_constrained.py` in `$OCEANDATA/models/dino-fusion/tav0h83b/inference/new_constraints/`, with the paper generations as reference). `--gens` selects the generations to plot. Sets are defined in `load_data.py`, figures are saved in `Results/a_priori/figures/<set>/`.

| Script | Content (paper) |
| --- | --- |
| `spatial_structure.py` | T and S at the surface and at 340 m, data vs generations (Figure 2 left) |
| `variability.py` | point by point variance of SST / SSS (Figure 2 right) and mean variance (Table 2) |
| `box_metrics.py` | Bottom-Water / Deep-Water T and S, density errors (Table 1) |
| `density.py` | zonally averaged density of the generated states |

## 4. A posteriori evaluation

See [Results/a_posteriori/README.md](Results/a_posteriori/README.md) : conversion of a generated state into a NEMO restart, launch of the run, and density figures (paper Figures 3 and 4).
