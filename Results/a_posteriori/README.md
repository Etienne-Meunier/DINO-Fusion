# A posteriori evaluation : launch NEMO from a generated state

On Jean-Zay everything of the project lives under `/lustre/fswork/projects/rech/omr/romr004/etienne/` (`$OCEANDATA` mirror in `oceandata/`); the `/lustre/fsn1` paths of the old runs below are purged. The automated pipeline is [notebooks/a_posteriori](../notebooks/a_posteriori/README.md). `dinostics` is a git submodule (`git submodule update --init`), it provides `NemoExperiment` used to read the reference simulations and to regrid restarts.

## Step 1 : Generate the `.npy` files

Generate states and un-normalise them (see the main [README](../../README.md)) :

```bash
python Diffusion_Model/generate.py --model_path <model> --loss mean_grad_density --eta <eta>
```

This gives a folder `<model>/inference/<loss>_<time>/clean/` with `{toce,soce,ssh}.npy`. Copy it to `<base_path_generated>/<generation_name>/`.

## Step 2 : Convert npy to restart `.nc` file

Adapt the `__main__` block of [convert_state.py](convert_state.py) and run it. Its steps are importable functions (`convert` chains them), used by the automated pipeline of [notebooks/a_posteriori](../notebooks/a_posteriori/README.md) :

```python
BASE_REFERENCES = f'{os.environ["OCEANDATA"]}/Dino-Fusion-Data/Restart-format/'
REFERENCES = {'1deg' : '1deg_restart', '1_4deg' : 'restart25_arch'}

base_path_generated = "/lustre/fswork/projects/rech/omr/romr004/etienne/diffusion_states/"
generation_name = 'C2'
idx_batch = 4 # which generated state of the batch to use
```

The script :

1. writes the generated state in a 1° restart -> `generated_restart_<generation_name>_1deg.nc`
2. upsamples it to the 1/4° grid
3. writes the 1/4° restart -> `generated_restart_<generation_name>_1_4deg.nc` and prints its `time_counter`, the run starts at `nn_it000 = time_counter + 1`

**Important :** all `.npy` and `.nc` files are stored in `base_path_generated`. `BASE_REFERENCES` contains "real" simulations that we use only for the grid and the restart templates : `$OCEANDATA/Dino-Fusion-Data/Restart-format/{1deg_restart,restart25_arch}/`, each with its restart (`DINO_<nn_itend>_restart.nc`), `namelist_cfg`, `domain_cfg_out.nc` and `mesh_mask.nc` directly in the folder. Remove the macOS `._*` files there if the volume is not HFS/APFS : `NemoExperiment.open_restart` would pick `._DINO_*_restart.nc`.

The regridding needs xesmf with xgcm 0.8.1 / xnemogcm 0.4.2 (newer xgcm removed the `periodic` argument used by dinostics) : locally the `dinostics` conda env (`mamba create -n dinostics -c conda-forge python=3.12 xarray xesmf esmpy xgcm=0.8.1 xnemogcm=0.4.2 gcm_filters xrft f90nml netcdf4 cftime dask scipy matplotlib`).

## Step 3 : Launch DINO run

1. Copy the run structure :

```bash
#1. Download the script to copy run (already in /lustre/fswork/projects/rech/omr/romr004/etienne/nnemo.sh)
curl -O https://raw.githubusercontent.com/Etienne-Meunier/nnemo/main/nnemo.sh; chmod +x nnemo.sh

#2. run the script -> copy the configuration (deletes the destination first), the run directory is <destination>/restart0
./nnemo.sh /lustre/fswork/projects/rech/omr/romr004/etienne/runs/Constraint_test diffusion
```

2. Modify `restart.config` to include the path of the restart to use (`jobs/launch.sh` copies it into `cn_ocerst_in` of `namelist_cfg`). `nn_it000` in `namelist_cfg` must be the `time_counter` printed by `convert_state.py` (Step 2) + 1, and `nn_itend` = `nn_it000` + number of steps - 1 (43200 steps = 1 year at `rn_Dt = 720`)

```yaml
cn_ocerst_in=/lustre/fswork/projects/rech/omr/romr004/etienne/diffusion_states/generated_restart_C2_1_4deg.nc
```

`scripts/setup_links.sh` of the `diffusion` branch links `diffusion_states` and `emp_climatology_1_4deg.nc` to purged `/lustre/fsn1` copies : re-point `emp_climatology_1_4deg.nc` to `/lustre/fswork/projects/rech/omr/romr004/data/dino_files/emp_climatology_1_4deg.nc`.

3. Run the model

```bash
sbatch jobs/launch.sh
```

<u>Bonus</u> : if you want to start over the run you can clean up with `./scripts/clean_restart_folder.sh`

## Step 4 : Compute density / produce graphs

[density_nemo.py](density_nemo.py) plots the zonally averaged potential density $\sigma_2$ with the profile averaged between 65°S and 45°S :

- of the generated initial restarts (paper Figure 3) -> `figures/generated_states.png`
- after 10 years of integration, against the target 1/4° simulation (paper Figure 4) -> `figures/states_after_10y_integration.png`

Update the experiment paths at the top of the script for new runs.

## Runs of the paper

Historical : the `/lustre/fsn1` entries are purged.

| Name | Restart | Experiment |
| --- | --- | --- |
| No constraint | `/lustre/fsn1/projects/rech/omr/romr004/data/` (purged) + `cn_ocerst_in` of the run | `/lustre/fsn1/projects/rech/omr/romr004/Blandine/DINO/Diffusion_runs/Constraint_no_C` (purged) |
| Constraint TS | idem | `/lustre/fsn1/projects/rech/omr/romr004/Blandine/DINO/Diffusion_runs/Constraint_TS` (purged) |
| Constraint C1 | `.../data/diffusion_states/` (purged) | `/lustre/fsn1/projects/rech/omr/uym68qx/DINO/Blandine/restart25_C1` (purged) |
| Constraint C2 | `/lustre/fsn1/projects/rech/omr/romr004/data/diffusion_states/generated_restart_C2_1_4deg.nc` (purged) | `/lustre/fsn1/projects/rech/omr/uym68qx/DINO/Blandine/restart25_C2` (purged) |
| Target (1/4° spin-up) | `restart29/DINO_12960000_restart.nc` | `/lustre/fsstor/projects/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree` |
