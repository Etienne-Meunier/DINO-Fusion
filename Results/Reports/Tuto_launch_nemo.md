# Tuto : Launch NEMO with custom initial state



### Step 1 : Generate `.npy` file 

`{toce,soce,ssh}.npy`



### Step 2 : Convert npy to restart .nc file 

https://github.com/Etienne-Meunier/DINO-Fusion/blob/dev-diffusion/Results/notebooks/convert_state.ipynb

Just follow the notebook adapting :

```python
base_references = "/lustre/fsn1/projects/rech/omr/uym68qx/DINO/"
references = {'1deg' : 'Blandine/1deg_restart', '1_4deg' : 'Blandine/restart25_arch'}

base_path_generated = "/lustre/fsn1/projects/rech/omr/romr004/data/diffusion_states/"
generation_name = 'C2'
```



**Important :** all `.npy` and `.nc` files are stored in the `base_path_generated` . `base_references` contains "real" simulation that we use only for the grid



### Step 3 : Launch DINO run

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

```
sbatch jobs/launch.sh 
```

<u>Bonus</u> : if you want to start over the run you can clean up with `./scripts/clean_restart_folder.sh`



### Step 4 : Compute density / produce graphs

https://github.com/Etienne-Meunier/DINO-Fusion/blob/dev-diffusion/Results/notebooks/ComputeDensity.ipynb



### Step 5 : Fill the report for reproducibility 

[Report](Runs.md)





