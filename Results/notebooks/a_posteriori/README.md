# A posteriori runs of the optimal sweep

One NEMO DINO 1/4° run per constraint of the optimal sweep (`generate/3_generate_optimal.py`), initialised from the median member of each generation batch. The pipeline reuses `paper/common.py` (generations, median criterion) and `Results/a_posteriori/convert_state.py` (npy -> 1° restart -> 1/4° restart), see [Results/a_posteriori/README.md](../../a_posteriori/README.md) for the manual version.

Everything is written next to the generation, in `<generation folder>/aposteriori/` with `<generation folder> = $OCEANDATA/models/dino-fusion/<RUN>/inference/<PREFIX>/<constraint>/<timestamp>/`. On Jean-Zay (host `jzjulie`) everything lives under `/lustre/fswork/projects/rech/omr/romr004/etienne/` and the same tree is mirrored under `JZ_OCEANDATA = .../etienne/oceandata/`, so the scripts run unchanged on both sides with `OCEANDATA` set accordingly (`export OCEANDATA=/lustre/fswork/projects/rech/omr/romr004/etienne/oceandata/` on JZ). `RUN` / `PREFIX` default to `tav0h83b` / `optimal` and follow the `PAPER_RUN` / `PAPER_PREFIX` environment variables, as the paper scripts.

| Step | Script | Where | Writes in `aposteriori/` |
| --- | --- | --- | --- |
| 1. Median member | `1_select_median.py` | local (or JZ) | `{toce,soce,ssh}.npy`, `selection.json` |
| sync | `sync_jz.py push` | local | - |
| 2. 1/4° restart | `2_convert_restart.py` | JZ | `restart_1deg.nc`, `restart_1_4deg.nc`, `conversion.json` |
| 3. NEMO run | `3_launch_nemo.py --submit` | JZ | `nemo_run/`, `launch.json` |
| 4. Collect | `4_collect_run.py --run` (twice) | JZ | `results/` (merged restart, outputs, logs, `collect.json`) |
| 5. Pull | `sync_jz.py pull --results` | local | json of steps 2-3, `results/` |

Cluster constants are in [settings.py](settings.py) (`JZ_HOST`, `JZ_ROOT`, `JZ_OCEANDATA`, `NNEMO = <JZ_ROOT>/nnemo.sh`, `NSTEPS`, `REQUIRED_LINKS`, ...). The reference runs used for the grids are `BASE_REFERENCES` / `REFERENCES` in `convert_state.py`, under `$OCEANDATA/Dino-Fusion-Data/Restart-format/` (locally and in the JZ mirror).

## Step 0 : a batch of training states (reference)

NEMO is restarted from T, S and SSH only, with the velocities at rest, which is itself a perturbation of the state. To measure it, the same pipeline runs on training states :

```bash
python 0_data_batch.py    # optimal/data/<timestamp>/states.npy (16 normalised states) and args.json
```

The 16 states follow, in the same seeded shuffle, the 200 states of the data batch of `paper/common.py`, so they are disjoint from the batch the median criterion compares to (the script checks that the shuffle reproduces that batch). They are saved as `states.npy`, not `generated.npy`, so the paper, graphs and viewer scripts do not load them as a generation. Step 1 adds them as the run `data`, steps 2 to 4 treat it as any constraint (`--constraints data`).

## Step 1 : select the median member

```bash
cd Results/notebooks/a_posteriori
python 1_select_median.py
```

For every constraint under `inference/<PREFIX>/` (latest timestamp if several), the median member is the upper median (`np.argsort(errors)[len(errors)//2]`) of the unweighted RMSE between the volume-weighted zonal $\sigma_0$ section of each member and the batch-mean section of a data batch, i.e. the member shown in `paper/fig1_density_profile.py` (`zonal_sections`, `rmse`, `median_member` of `paper/common.py`). The member is un-normalised with `save_clean` into `{toce,soce,ssh}.npy` of shapes `(1, 36, 199, 62)` / `(1, 1, 199, 62)`, the format of `convert_state.load_member` with `idx_batch=0`. `selection.json` records the member index, its RMSE and those of all members, the data batch size, the model and the source generation (relative to `$OCEANDATA`, with its `args.json`).

The data batch is a shuffled draw of 200 samples seeded with `DataLoader.DATA_SEED` (`paper/common.py`), the same batch as the paper figures, so re-running step 1 gives the same selection. Generations already converted or launched are skipped unless `--force`.

## Step 2 : convert to a 1/4° restart (Jean-Zay)

```bash
python sync_jz.py push --references --run   # local, sends aposteriori/{*.npy,selection.json} and the reference runs
# on JZ
cd <repo>/Results/notebooks/a_posteriori
python 2_convert_restart.py           # --constraints none zero_mean_border0 to restrict, --force to redo (refused once launched)
```

Needs the reference runs of `convert_state.BASE_REFERENCES` and an environment with xesmf, xgcm 0.8.1 and xnemogcm 0.4.2, no torch (locally the `dinostics` conda env, see [Results/a_posteriori/README.md](../../a_posteriori/README.md)). It also runs locally (the reference runs are in `$OCEANDATA`), then `python sync_jz.py push --all --references --run` sends the restarts, `conversion.json` and the reference runs to JZ (about 4.8 GB for the 5 constraints of the optimal sweep, mostly the 1/4° restarts). `conversion.json` holds the absolute restart paths (of the machine that converted, step 3 uses `aposteriori/restart_1_4deg.nc` itself) and the `time_counter` of the 1/4° restart (undecoded, checked equal to its `kt`), from which step 3 sets `nn_it000`.

## Step 3 : create and submit the NEMO runs (Jean-Zay)

`3_launch_nemo.py` and `settings.py` only use the python standard library (python >= 3.7, no torch / xarray), with `OCEANDATA` exported to the JZ mirror.

```bash
python 3_launch_nemo.py               # dry run, prints the actions
python 3_launch_nemo.py --submit      # --nsteps to change the length, 432000 steps = 10 years by default
```

For each converted generation :

1. `nnemo.sh <aposteriori>/nemo_run diffusion` : `rm -rf nemo_run`, shallow clone of the `diffusion` branch of Dinonline (from commit 944ff64) into it, `restart0/scripts/setup_links.sh`. The run directory is `nemo_run/restart0`. The branch itself sets the XIOS outputs (monthly 3D means `1m_3D`, monthly 2D means `1m`, 10-day snapshots `10d_inst` disabled), the `emp_climatology_1_4deg.nc` link to its `$WORK` copy and the slurm options (`--time=05:00:00`, no mails).
2. In `restart0` : `restart_1_4deg.nc` links to the restart of step 2 (read with `cn_ocerst_indir = '.'`), the dangling `diffusion_states` link of the branch is removed.
3. `restart.config` is written as `cn_ocerst_in=restart_1_4deg.nc` (copied into `namelist_cfg` by `jobs/launch.sh`), and `namelist_cfg` gets `nn_it000 = time_counter + 1`, `nn_itend = nn_it000 + NSTEPS - 1`, as `update_namelist.sh` of Dinonline main. `nn_stock` is left commented out, so the run writes a single restart at `nn_itend`.
4. The restart, the `REQUIRED_LINKS` targets (`emp_climatology_1_4deg.nc`) and the executables of `jobs/mpmd.conf` must exist, otherwise nothing is submitted. Then `sbatch --job-name=aposteriori_<constraint> --output=aposteriori_<constraint>.out jobs/launch.sh` from `restart0` (past runs of this layout took about 170 min for 432000 steps, so about 3 h per run, within the 5 h limit of `launch.sh`). The job id, the namelist values and the XIOS output groups read from `file_def_nemo-oce.xml` go to `launch.json`.

A generation with a `launch.json`, or with a `nemo_run/` but no `launch.json` (interrupted launch), is skipped : `nnemo.sh` deletes its destination, remove `nemo_run/` by hand to start over.

### Experiments

`--experiment` (steps 3, 4 and `sync_jz.py pull --results`) picks an entry of `EXPERIMENTS` in `settings.py`. Each experiment starts from the same `restart_1_4deg.nc` and has its own run directory, launch record and results folder in `aposteriori/` :

| experiment | length | XIOS outputs | slurm | run directory, record, results |
|---|---|---|---|---|
| `10y` (default) | 432000 steps, 10 years | `file_def_nemo-oce.xml` of the branch, monthly means | `launch.sh` (`qos_cpu-t3`, 5 h) | `nemo_run/`, `launch.json`, `results/` |
| `daily_2m` | 7200 steps, 2 months | `file_def_nemo-oce_daily.xml` of the branch copied onto `file_def_nemo-oce.xml` : daily means of T, S, e3t, U, V and of the 2D fields, ~12 GB per run | `--qos=qos_cpu-dev --time=01:00:00` | `nemo_run_daily_2m/`, `launch_daily_2m.json`, `results_daily_2m/` |

```bash
python3 3_launch_nemo.py --experiment daily_2m --submit
python3 4_collect_run.py --experiment daily_2m --run
```

### GitHub mirror on Jean-Zay

GitHub is not always reachable from the JZ login nodes, and `nnemo.sh` clones Dinonline from it. When the bare clone `NNEMO_MIRROR` (`<JZ_ROOT>/code/Dinonline.git`) exists, step 3 runs `nnemo.sh` with git's `url.<mirror>.insteadOf` set (environment variables only, nothing written in the git config), so the branch is cloned from the mirror. Refresh the mirror after each push to the branch, from a machine that reaches GitHub :

```bash
git clone --bare --single-branch --branch diffusion https://github.com/Etienne-Meunier/Dinonline.git /tmp/Dinonline.git
rsync -a --delete /tmp/Dinonline.git/ jzjulie:/lustre/fswork/projects/rech/omr/romr004/etienne/code/Dinonline.git/
```

## Step 4 : collect a finished run (Jean-Zay)

```bash
export OCEANDATA=/lustre/fswork/projects/rech/omr/romr004/etienne/oceandata/
python3 4_collect_run.py --constraints none_border0          # dry run, prints the status and the actions
python3 4_collect_run.py --constraints none_border0 --run    # submits the rebuild job (qos_cpu-dev, 30 min)
python3 4_collect_run.py --constraints none_border0 --run    # again once the rebuild job is done : checks, moves to results/
```

For each launched run (`launch.json`), in `nemo_run/restart0` :

1. The run is finished if its job (`job_id` of `launch.json`) is no longer in `squeue`, `time.step` reached `nn_itend`, `ocean.output` has NEMO's normal end marker `AAAAAAAA` and no `E R R O R`, and the slurm log `aposteriori_<constraint>.out` has the `real` line of `time srun` (elapsed time). Otherwise, or if `squeue` is unavailable, nothing is touched.
2. The final restart, split in one `DINO_<nn_itend>_restart_XXXX.nc` per process, is merged into `DINO_<nn_itend>_restart.nc` by `restart0/rebuild_nemo.exe` (shipped by the `diffusion` branch, it reads `filebase` / `ndomain` from the namelist `nam_rebuild` that the script writes), with the modules of `jobs/launch.sh` and `OMP_NUM_THREADS=4`. The number of pieces must equal `DOMAIN_number_total` of piece `0000`. By default the merge is a `qos_cpu-dev` job, whose id is written in `rebuild_<constraint>.json` (no second submission while it is in the queue); `--login` runs it on the login node instead. Its output goes to `rebuild_<constraint>.out` and must contain `NEMO rebuild completed successfully` : this proof is required before anything is moved or cleaned, a merged file without it is never used. A failed merge (partial file removed on the login node) is reported until `rebuild_<constraint>.{json,out}` are removed.
3. `ncdump -h` checks the merged restart dimensions (`y=797, x=242, nav_lev=36`).
4. The merged restart and the XIOS outputs listed in `launch.json` (`DINO_1m_grid_{T,U,V}_3D.nc`, `DINO_1m_grid_T_2D.nc`) are moved to `aposteriori/results/`, the slurm and rebuild logs and `ocean.output` are copied there, with `collect.json` (run status, elapsed time, rebuild proof, number of pieces, file sizes). A run with `results/collect.json` is skipped. The restart pieces stay in `nemo_run/restart0` unless `--clean-pieces`, which removes them only after the proven merge and the dimension check.

Like step 3, the script needs only the python standard library.

## Step 5 : pull the results

```bash
python sync_jz.py pull --results --constraints none_border0         # prints the rsync command
python sync_jz.py pull --results --constraints none_border0 --run   # ~25 GB per run
```

`--results` adds `aposteriori/results/{*.nc,*.json,*.out,ocean.output}` to the pull; `nemo_run/` (restart pieces included) stays on JZ.

## Fully on Jean-Zay

Steps 1 to 3 can all run on JZ if the generations, the training data (`TrainingConfig.data_file`) and a torch environment are there : set `OCEANDATA` to the JZ mirror and skip `sync_jz.py`.

## Results

```
<generation folder>/aposteriori/
    toce.npy soce.npy ssh.npy   median member (step 1)
    selection.json              (step 1)
    restart_1deg.nc             (step 2, JZ)
    restart_1_4deg.nc           initial state of the run (step 2, JZ)
    conversion.json             (step 2)
    nemo_run/restart0/          NEMO run directory (step 3, JZ), split restart pieces DINO_<nn_itend>_restart_XXXX.nc
    launch.json                 (step 3)
    results/                    (step 4, JZ, pulled by step 5), about 25 GB per run :
        DINO_1m_grid_{T,U,V}_3D.nc    monthly means of the 3D fields over the 10 years
        DINO_1m_grid_T_2D.nc          monthly means of the 2D fields
        DINO_<nn_itend>_restart.nc    merged final restart
        aposteriori_<constraint>.out, ocean.output, collect.json
```

`python sync_jz.py pull --run` brings `conversion.json` and `launch.json` back locally (`--all` also the `.nc` restarts, `--results` the `results/` of step 4). `push` sends `*.npy` and `selection.json` (with `--all` also the `.nc` restarts and `conversion.json`, with `--references` also `Dino-Fusion-Data/Restart-format/`). `nemo_run/` and macOS `._*` files are never transferred, and `--update` never overwrites a newer file on the receiving side. `--constraints none_border0 ...` restricts the `aposteriori/` transfers to these constraints (as for steps 2 and 3). Without `--run` the script only prints the rsync commands, `--dry-run` runs them with `rsync --dry-run` (which connects to JZ).
