# miniveros

Conditional diffusion emulator of Veros ACC ocean states. Given the two coefficients of the
EKE eddy closure, `c_k` and `c_eps`, generate the temperature and salinity fields of the
post-spin-up state directly, skipping the spin-up integration.

This package is a fork of the `Diffusion_Model` package of DINO-Fusion (NEMO/DINO), adapted to:
a parameter ensemble instead of one trajectory, conditioning on scalar parameters, in-memory data
instead of webdataset tars, no vertical striding, and no `tensordict` dependency.

## Data

Raw input: one `ck<ck>_eps<eps>.npz` per run (100 runs on a 10 x 10 log-spaced grid), each with
`temp`, `salt`, `u`, `v`, `eke`, `tke` of shape `(t, x, y, z)` including two ghost cells per side,
plus `psi`, `time` (s), `zt` (m). 100 years, 30-day output.

`extract_data.py` reads only `temp` and `salt` (by seeking inside the zip), keeps the last 20 years
of every run on the interior grid, and writes one npz (the analogue of DINO's `xarray_numpy.py`):

| key | shape | meaning |
|---|---|---|
| `temp`, `salt` | `(N, 15, 42, 30)` float32 | `(sample, z, y, x)`; z index 0 is the bottom |
| `ck`, `eps`, `run_id`, `time_s`, `year` | `(N,)` | per-sample parameters and time |
| `run_names`, `run_ck`, `run_eps` | `(100,)` | per-run |
| `mask_land` | `(15, 42, 30)` bool | fixed ridge, 930 cells |
| `holdout_runs`, `train_runs` | | the split, chosen before statistics |
| `lvl_mean/std` `(2, 15)` | | per-level normalisation statistics, train split only |
| `cond_keys`, `cond_mean`, `cond_std` | | standardisation of `log ck`, `log eps` |

Salinity is exactly 35 in every water cell of every run. It is carried through the whole pipeline
so the code handles two active fields, but it contains no information in this dataset.

## Pipeline

```
fields {temp, salt} (15,42,30) --concat--> (30,42,30) --normalise--> --land to 0--> --pad--> (30,48,32)
```

* **Normalisation** (`norm_mode`, `"<k>-std"`): per vertical level, `(x - mean_z) / (k * std_z)`, as in
  DINO-Fusion, with statistics from the training runs only. The std is floored (`std_floor`) so the
  constant salinity maps to exactly 0.
* **Padding**: zeros, `(1, 1, 3, 3)` in `(x_left, x_right, y_low, y_high)`, giving 48 x 32 which halves
  four times. Padding equals the land value.
* **Model**: diffusers `UNet2DModel` (64, 64, 128, 128), plus an MLP that maps the standardised
  `(log ck, log eps)` into the time-embedding space (`class_embed_type="identity"`). Optional
  classifier-free guidance via `cond_drop_prob`.
* **Diffusion**: DDPM, 1000 steps, `squaredcos_cap_v2`, `clip_sample=True`, epsilon prediction, EMA.
* **Sampling**: DDPM loop with a constraints hook; `LandZero` re-imposes zeros on land and padding.

## Usage (from this directory)

```bash
python extract_data.py --raw-dir <raw> --out data/veros_acc_TS.npz          # once, CPU
python -m tests.test_roundtrip [data/veros_acc_TS.npz]                     # seconds, CPU
python train.py --preset dev  --set data_file=data/veros_acc_TS.npz run_dir=runs/dev
python train.py --preset full --set data_file=... run_dir=runs/full_3std
python generate.py --run-dir runs/full_3std --holdout --n-samples 8
python generate.py --run-dir runs/full_3std --grid --n-samples 4
python evaluate.py --data-file ... --samples runs/full_3std/samples/holdout_*.npz --grid-samples runs/full_3std/samples/grid_*.npz
```

Any `Config` field can be overridden with `--set key=value`. Re-running `train.py` with the same
`run_dir` resumes from `ckpt.pt`, so a run can be chained across short jobs.

## Cluster

Compute nodes have no `git` and no `module` function unless inherited from a login shell, so `submit.sh`
records the commit hash at submission time and the environment recipe sources the module init files itself.

`jobs/` holds SLURM templates with only generic resources in the `#SBATCH` headers. Account, QoS,
constraint, paths and the environment activation come from `jobs/jz_env.sh`, which is gitignored:
copy `jz_env.example.sh`, fill it in on the cluster, then

```bash
jobs/submit.sh extract
jobs/submit.sh train --preset full --set data_file=$MV_DATA run_dir=$MV_WORK/runs/full_3std
jobs/submit.sh generate_eval $MV_WORK/runs/full_3std 8
jobs/campaign.sh 3-std              # or the whole chain at once: extract -> train -> generate_eval
```

## Evaluation

On the hold-out runs, against the true time-mean state over the last 20 years, water cells only:
RMSE of the ensemble mean and of single samples, domain-mean bias, per-level RMSE profile,
ensemble spread against the true within-window spread, salinity error, and the fraction of interfaces with
temperature decreasing upward, compared with the same fraction in the truth (the true states do contain such
inversions in the cold southern region, so this is a distributional check, not an absolute stability test). Baselines: the mean
training state, the nearest training run in log-parameter space, and the average of the axis-neighbour
training runs on the grid. With `--grid-samples`, the 10 x 10 map of domain-mean temperature is compared
with the truth.
