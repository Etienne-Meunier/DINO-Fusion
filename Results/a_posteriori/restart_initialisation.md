# Initial NEMO state from a diffusion generation

How [notebooks/a_posteriori](../notebooks/a_posteriori/README.md) and [convert_state.py](convert_state.py) turn a generated state into the initial restart of a NEMO DINO 1/4° run. Checked against the code, the reference runs in `$OCEANDATA/Dino-Fusion-Data/Restart-format/` and the `ocean.output` of the `data` run (`tav0h83b/inference/optimal/data/201026-183649/aposteriori/results_daily_2m/`). The NEMO source is not available locally: *unverified* = depends on NEMO internals, *numerical* = from a numerical check of the files.

`<method>` is the constraint (`none`, `zero_mean_border0`, ..., `data`). On disk the files are `<INFERENCE>/<PREFIX>/<method>/<timestamp>/aposteriori/restart_{1deg,1_4deg}.nc`; the method is in the folder, not in the name.

The model only produces T, S, SSH on the 1° grid. Everything else comes from a reference run, is zero, or is derived by NEMO.

## Pipeline

```python
# A. 1_select_median.py : median member of the 16 generated states
x = load('generated.npy')                             # (16, 37, 208, 64), normalised
data = batch(200, seed=DATA_SEED)                     # reference training batch
err = rmse(zonal_sigma0(x), mean(zonal_sigma0(data))) # TEOS-10 sigma_0, volume-weighted zonal mean
m = argsort(err)[len(err) // 2]                       # upper median

# B. TransformationPipeline.__uncall__ + post_processing.unnormalise
f = x[m][:, 5:-4, 1:-1]                               # unpad -> (37, 199, 62)
f[mask_strided] = nan                                 # training mask, 37 channels
f = f * 3 * std + mean                                # one mean/std per channel, broadcast over (y, x)
T, S, ssh = split(f, [18, 18, 1])
T, S = unstride(T), unstride(S)                       # levels 0,2,...,34 -> 36: odd k = mean(k-1, k+1), 35 = 34
T[mask_full] = S[mask_full] = nan
T[:, 1] = T[:, 2]; S[:, 1] = S[:, 2]; ssh[1] = ssh[2] # row y=1 <- y=2 (assumes y=0 is the land row)

# C. write_restart_1deg -> restart_1deg_<method>.nc
r = open('1deg_restart/DINO_00000002_restart.nc')     # found from nn_itend=2, contains kt=21312000
r.tn = r.tb = T; r.sn = r.sb = S; r.sshn = r.sshb = ssh
r.rhop = seos(T, S, z=gdept_0_1deg)                   # bug: NEMO's rhop uses z=0, see table

# D. dinostics GridManipulation.regrid_restart
r[~tmask_1deg] = nan
r = fill_nearest(r, 'x'); r = fill_nearest(r, 'y')    # extrapolate onto land
h = xesmf.Regridder(r, template_1_4deg, 'bilinear', extrap_method='nearest_s2d')(r)  # target grid = template of E
h[~tmask_1_4deg] = 0 # Land values to 0 
h.un = h.ub = h.vn = h.vb = 0 # Zero velocity

# E. write_restart_1_4deg -> restart_1_4deg_<method>.nc
out = template_1_4deg = open('restart25_arch/DINO_10800000_restart.nc') # spun-up 1/4° run, 250 years 
out[tn tb sn sb un ub vn vb rhop sshn sshb] = h[...]  # everything else, time metadata included, from the template
out = out.fillna(0)

# F. 3_launch_nemo.py
nnemo.sh nemo_run diffusion                           # clone Dinonline branch diffusion, run dir nemo_run/restart0
restart.config: cn_ocerst_in = restart_1_4deg_<method>.nc  # written into namelist_cfg by jobs/launch.sh
nn_it000 = time_counter + 1 = 10800001
nn_itend = nn_it000 + NSTEPS - 1                      # 432000 (10y) or 7200 (daily_2m)
```

Details:

- **A.** The ranking uses `observer.unnormalized`, i.e. B without the row copy. Volumes come from `Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc`.
- **Step 0 (`data` run).** 16 training states (200 to 215 of the same shuffle, disjoint from the reference batch), saved as `data/<timestamp>/states.npy` and then run through A to F. Measures the effect of the restart procedure itself (velocities at rest).
- **B.** Training data `dino_1_4_degree_coarse_240125.tar` is presumably 1/4° coarse-grained to 1°; only the name and the 199 x 62 shape support this.
- **D.** The 1° time values (`kt`, `ndastp`, ...) are carried through the regridding but discarded at E.
- **E.** The template variables `x(y, x)`, `y(y, x)` are dropped by `open_restart` and replaced by `lon`, `lat`. The kept variables are identical to the template (*numerical*).
- **F.** The run uses `ln_rstart = T`, `nn_rstctl = 2` (calendar read in the restart) and `ln_1st_euler = F` (`namelist_ref` default). So the first step should be a leapfrog step with before = now (*unverified*).

## Restart variables

`restart_1_4deg_<method>.nc`: `x = 242`, `y = 797`, `nav_lev = 36`, `time_counter = 1`. "Template" = `restart25_arch/DINO_10800000_restart.nc`. "Read" is taken from `ocean.output`: `read ... ok` lines, or `info` for the "Info read in restart" block.

| Variable | Meaning | Read | Value at start |
| --- | --- | --- | --- |
| `lon`, `lat` | T-point coordinates | no | 1/4° `glamt`, `gphit` (replace template `x`, `y`) |
| `nav_lev` | level coordinate | no | template |
| `time_counter` | time step | no | template, 10800000; sets `nn_it000` |
| `kt` | time step | info | template, 10800000 |
| `ndastp` | date | info | template, 2491130 (0249/11/30) |
| `adatrj` | elapsed days | info | template, 90000 |
| `ntime` | time of day | info | template, 0 |
| `rdt` | time step (s) | no | template, 720 |
| `tn`, `tb` | temperature | yes | generated, `tb = tn` |
| `sn`, `sb` | salinity | yes | generated, `sb = sn` |
| `sshn`, `sshb` | SSH | yes | generated, `sshb = sshn` |
| `un`, `ub`, `vn`, `vb` | velocities | yes | 0 |
| `rhop` | potential density | yes | S-EOS of generated T, S **at** `z = gdept_0`; see below |
| `utau_b`, `vtau_b`, `qns_b`, `emp_b`, `sfx_b` | surface fluxes of the previous step | yes | template |
| `sbc_hc_b`, `sbc_sc_b`, `qsr_hc_b`, `fraqsr_1lev` | surface heat/salt content trends, solar penetration | yes | template |
| `en`, `avt_k`, `avm_k`, `dissl` | TKE state, vertical diffusivity/viscosity | yes | template |

Not in the restart:

| Variable | Value at start |
| --- | --- |
| `e3t`, `e3u`, `e3v` | Derived from SSH: `e3t = e3t_0 * (1 + ssh / ht_0)`, with `ht_0 = sum_k e3t_0 * tmask` (z-star, QCO, `ln_linssh = F`: confirmed in the log). The formula is *unverified*, but for `e3t` it matches the run output to 3e-5 m (*numerical*). Before = now since `sshb = sshn` |

**`rhop` error.** NEMO's `rhop` is the S-EOS at `z = 0` (*numerical*: NEMO-written restarts match `z = 0`; source *unverified*). The pipeline keeps the term `mu1 * z`, so the stored value is off by `a0 * mu1 * z * (10 - T)`: too heavy where T < 10°C, up to 0.78 kg/m3 at depth (*numerical*). NEMO reads it (log line 1499), but it is probably recomputed before any use (*unverified*). Fix: `z = 0` in `write_restart_1deg`.

## Notes

- Effective initial state: generated T, S, SSH upsampled from 1°, ocean at rest, plus the forcing memory, mixing state and calendar of the spun-up 1/4° run.
- SSH with zero velocity is unbalanced, so the run starts with a geostrophic adjustment. The `data` run measures this.
- The odd levels and level 35 are interpolated, not generated.
- Masks used: training (B), 1° `mesh_mask` (D), 1/4° `mesh_mask` (D), and the selection mask file (A). Whether the 1° masks are identical was not checked; any mismatch is filled by D's nearest extrapolation.
- The figures and losses use TEOS-10 sigma_0, while the model runs S-EOS. TEOS-10 is 0.76 to 0.86 kg/m3 heavier, and the vertical contrast is nearly equal (2.62 against 2.59) (*numerical*).

## Open questions

| Question | How to settle it |
| --- | --- |
| `rhop` at `z = 0` in NEMO? | `eos_insitu_pot` in `eosbn2.F90` |
| `rhop` recomputed before use? | `stp` / `istate`, or `rhop` of `output.init_0000.nc` vs restart |
| First step leapfrog with before = now? | `rst_read` / `stp` |
| `e3t/u/v` formula at start? | `domqco.F90`, or `vovvle3t` of `output.init_0000.nc` |
| `y = 0` is the land row? | training mask at `y = 0` |
| Training mask = 1° `mesh_mask.tmask`? | compare them |
| Training data = coarse-grained 1/4°? | script that built the tar |
