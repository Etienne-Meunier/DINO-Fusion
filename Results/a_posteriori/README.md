# A posteriori evaluation : launch NEMO from a generated state

All paths below are on Jean-Zay. `dinostics` is a git submodule (`git submodule update --init`), it provides `NemoExperiment` used to read the reference simulations and to regrid restarts.

## Step 1 : Generate the `.npy` files

Generate states and un-normalise them (see the main [README](../../README.md)) :

```bash
python Diffusion_Model/generate.py --model_path <model> --loss mean_grad_density --eta <eta>
```

This gives a folder `<model>/inference/<loss>_<time>/clean/` with `{toce,soce,ssh}.npy`. Copy it to `<base_path_generated>/<generation_name>/`.

## Step 2 : Convert npy to restart `.nc` file

Adapt the header of [convert_state.py](convert_state.py) and run it (it can also be run cell by cell) :

```python
base_references = "/lustre/fsn1/projects/rech/omr/uym68qx/DINO/"
references = {'1deg' : 'Blandine/1deg_restart', '1_4deg' : 'Blandine/restart25_arch'}

base_path_generated = "/lustre/fsn1/projects/rech/omr/romr004/data/diffusion_states/"
generation_name = 'C2'
idx_batch = 4 # which generated state of the batch to use
```

The script :

1. writes the generated state in a 1° restart -> `generated_restart_<generation_name>_1deg.nc`
2. upsamples it to the 1/4° grid
3. writes the 1/4° restart -> `generated_restart_<generation_name>_1_4deg.nc` and prints the `time_counter` to use for `nn_it000`

**Important :** all `.npy` and `.nc` files are stored in `base_path_generated`. `base_references` contains "real" simulations that we use only for the grid.

## Step 3 : Launch DINO run

1. Copy the run structure :

```bash
#1. Download the script to copy run
curl -O https://github.com/Etienne-Meunier/nnemo/blob/main/nnemo.sh; chmod +x nnemo.sh

#2. run the script -> copy the configuration
./code/nnemo.sh DINO/Diffusion_runs/Constraint_test diffusion
```

2. Modify `restart.config` to include the path of the restart to use

```yaml
cn_ocerst_in=diffusion_states/generated_restart_C2_1_4_deg.nc
```

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

| Name | Restart | Experiment |
| --- | --- | --- |
| No constraint | `/lustre/fsn1/projects/rech/omr/romr004/data/` + `cn_ocerst_in` of the run | `/lustre/fsn1/projects/rech/omr/romr004/Blandine/DINO/Diffusion_runs/Constraint_no_C` |
| Constraint TS | idem | `/lustre/fsn1/projects/rech/omr/romr004/Blandine/DINO/Diffusion_runs/Constraint_TS` |
| Constraint C1 | `.../data/diffusion_states/` | `/lustre/fsn1/projects/rech/omr/uym68qx/DINO/Blandine/restart25_C1` |
| Constraint C2 | `/lustre/fsn1/projects/rech/omr/romr004/data/diffusion_states/generated_restart_C2_1_4deg.nc` | `/lustre/fsn1/projects/rech/omr/uym68qx/DINO/Blandine/restart25_C2` |
| Target (1/4° spin-up) | `restart29/DINO_12960000_restart.nc` | `/lustre/fsstor/projects/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree` |
