# Initial NEMO state from a diffusion generation

How [notebooks/a_posteriori](../notebooks/a_posteriori/README.md) and [convert_state.py](convert_state.py) turn a generated state into the initial restart of a NEMO DINO 1/4° run. Checked against the code, the reference runs in `$OCEANDATA/Dino-Fusion-Data/Restart-format/`, the `ocean.output` of the `data` run (`tav0h83b/inference/optimal/data/201026-183649/aposteriori/results_daily_2m/`) and the NEMO 4.2 source (`src/OCE` of 4.2.2 from forge.nemo-ocean.eu; the blocks cited below are identical in 4.2.0 and 4.2.2, and the log messages of the run match this source). *numerical* = from a numerical check of the files.

**Status.** This document describes the **minimal restart** protocol, implemented (step 2 writes it, experiments `10y_minimal`, `daily_2m_minimal`, `weekly_2y_minimal` and `daily_2w_minimal` of `settings.py` run it); the `data` run of the optimal sweep has run `daily_2m_minimal` and `weekly_2y_minimal`. The runs done so far (experiments `10y` and `daily_2m`) used the **template restart** protocol: every variable not generated came from the spun-up 1/4° run, and `rhop` was computed at `z = gdept_0`. See [Previous protocol](#previous-protocol-template-restart).

`<method>` is the constraint (`none`, `zero_mean_border0`, ..., `data`). On disk the files are `<INFERENCE>/<PREFIX>/<method>/<timestamp>/aposteriori/restart_{1deg,1_4deg}_minimal.nc` (`restart_{1deg,1_4deg}.nc` for the template restart); the method is in the folder, not in the name.

The model only produces T, S, SSH on the 1° grid. The restart holds only these fields, the zero velocities and the calendar. NEMO initialises everything else itself, as for a start from rest.

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

# Grids (lon, lat = glamt, gphit of domain_cfg_out.nc; tmask of mesh_mask.nc) come from the reference runs
# 1deg_restart (1°, 199 x 62) and restart25_arch (1/4°, 797 x 242) in Restart-format/, attached by open_restart

# C. write_restart_1deg -> restart_1deg_minimal.nc
r = open('1deg_restart/DINO_00000002_restart.nc')     # 1° grid container (found from nn_itend=2, contains kt=21312000)
r.tn = T; r.sn = S; r.sshn = ssh                      # other variables keep the 1° reference values, never written at E

# D. dinostics GridManipulation.regrid_restart
r[~tmask_1deg] = nan
r = fill_nearest(r, 'x'); r = fill_nearest(r, 'y')    # extrapolate onto land
h = xesmf.Regridder(r, template_1_4deg, 'bilinear', extrap_method='nearest_s2d')(r)  # target grid = template of E
h[~tmask_1_4deg] = 0                                  # land values to 0
h.un = h.vn = 0                                       # zero velocity

# E. write_restart_1_4deg -> restart_1_4deg_minimal.nc   (minimal restart)
template_1_4deg = open('restart25_arch/DINO_10800000_restart.nc')  # used for the grid and the calendar only
out = template_1_4deg[['kt', 'ndastp', 'adatrj', 'ntime']]          # calendar 0249/12/01, forcing day 0 (kt = 250 years)
out[kt time_counter adatrj ndastp] = start_calendar(kt, start_day)  # if selection.json has a start_day (see Start date)
out[tn sn sshn un vn] = h[...]                                      # nothing else is written
out = out.fillna(0)

# F. 3_launch_nemo.py
nnemo.sh nemo_run_minimal diffusion                   # clone Dinonline branch diffusion, run dir nemo_run_minimal/restart0 (per experiment)
restart.config: cn_ocerst_in = restart_1_4deg.nc     # link to restart_1_4deg_minimal.nc, written into namelist_cfg by jobs/launch.sh
nn_it000 = time_counter + 1 = 10800001                # 10800001 + 120 * start_day with a start_day
nn_itend = nn_it000 + NSTEPS - 1                      # 432000 (10y), 7200 (daily_2m) or 1680 (daily_2w_minimal)
ln_1st_euler = .true.                                 # Euler first step, before fields = now fields
```

Details:

- **A.** The ranking uses `observer.unnormalized`, i.e. B without the row copy. Volumes come from `Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc`.
- **Step 0 (`data` run).** 16 training states (200 to 215 of the same shuffle, disjoint from the reference batch), saved as `data/<timestamp>/states.npy` and then run through A to F. Measures the effect of the restart procedure itself, so it must use the same protocol as the generations.
- **B.** Training data `dino_1_4_degree_coarse_240125.tar` is 1/4° data coarse-grained to the 1° grid (199 x 62).
- **D.** The 1° time values (`kt`, `ndastp`, ...) are carried through the regridding but not written at E.
- **F.** `ln_rstart = T`, `nn_rstctl = 2`: the calendar is read in the restart and `nn_it000 - kt` must be 1 (`daymod.F90:334`). Without `sshb` in the restart, NEMO would force `l_1st_euler = .true.` anyway, with a warning (`domain.F90:427-433`); setting it in the namelist makes it explicit.

## Start date

Two clocks are read from the restart, and they disagree in the template:

- **Calendar.** With `nn_rstctl = 2` NEMO takes its date from the restart: `ndastp` is the date of the restart, as YYYYMMDD with a 3-digit year, in the 360-day calendar of 30-day months, and the first time step is on the next day. The template gives `ndastp = 2491130`, so the calendar of every run so far starts on 0249/12/01. The `ocean.output` of the `data` run (`results_daily_2m_minimal/`) shows it: `nrstdt = 2 : calendar parameters read in restart`, `date ndastp : 2491130`, then `New day, DATE Y/M/D = 0249/12/01 nday_year = 331` at time step 10800001.
- **Forcing phase.** The DINO seasonal forcing (`usrdef_sbc.F90`, `compute_day_of_year`) takes its phase from the time step only, `(kt / 120) mod 360` days (`rn_Dt = 720 s`, 120 steps per day); `ndastp` enters only through the year, and whole years cancel. The template `kt = 10800000` is exactly 250 years of 360 days, so the forcing of every run so far starts on forcing day 0, i.e. 1 January: the earlier "1 January" was right about the forcing, not about the calendar. The `data` run confirms it: its `qsr` matches forcing day 0 at `kt = 10800001`, while the calendar reads 0249/12/01.

The forcing has an annual cycle, so a state of a known season must start at the matching forcing day. `convert_state.convert` and `write_restart_1_4deg` take an optional `start_day` (forcing day of year, 1 to 359) and then replace the calendar of the template by `convert_state.start_calendar`: `kt = time_counter = 10800000 + 120 * start_day` (start of the template's forcing year plus `start_day` days), `adatrj = kt / 120` days, `ndastp` the date of day `start_day` in year 250, all with the dtypes of the template, `ntime` unchanged. Step 3 then sets `nn_it000 = kt + 1`, `nn_itend = nn_it000 + nsteps - 1`, and the final restart of step 4 is `DINO_<nn_itend>_restart.nc` from `launch*.json`. Without `start_day` the calendar is that of the template, as before. Step 2 (`2_convert_restart.py`) passes the `start_day` of `selection.json` when it has one, and records `start_day`, `kt`, `time_counter`, `adatrj` and `ndastp` of the written restart in `conversion_minimal.json`.

The assimilation study (`notebooks/assimilation/2_select.py`) uses it: phase `p` of the training data (state index mod 36) is mapped to forcing day `10 p`, so summer (phase 24) has `start_day = 240`, `kt = 10828800`, `adatrj = 90240`, `ndastp = 2500830` (first day 0250/09/01) and winter (phase 7) `start_day = 70`, `kt = 10808400`, `adatrj = 90070`, `ndastp = 2500310` (first day 0250/03/11). The mapping is approximate: the calendar date of the training states is unknown. Its `data` configuration is the source state in the model representation (18 T/S levels, odd levels interpolated, row `y = 1` copied from `y = 2`), as the `data` run of the pipeline.

## Restart variables

What NEMO 4.2 does with each variable at start, with `ln_rstart = T`, `ln_1st_euler = T`, `nn_rstctl = 2`. A variable read without existence check stops the run if it is missing (`iom_varid` defaults to `ldstop = .TRUE.`, `iom.F90:904`).

**Written**:

| Variable | Meaning | NEMO code | Value at start |
| --- | --- | --- | --- |
| `tn`, `sn` | temperature, salinity | required, `restart.F90:299-300` | generated |
| `sshn` | SSH | required, `restart.F90:381` | generated |
| `un`, `vn` | velocities | required, `restart.F90:297-298` | 0 |
| `kt` | time step | optional, `daymod.F90:319`; if present, the 3 below are required | template, 10800000 (forcing day 0); `10800000 + 120 * start_day` with a `start_day` |
| `ndastp`, `adatrj`, `ntime` | date, elapsed days, time of day | `daymod.F90:340-343` | template: 0249/11/30, 90000 days, 0; the calendar starts on 0249/12/01 (360-day calendar), the forcing on forcing day 0 (set by `kt`). With a `start_day`, `kt`, `time_counter`, `adatrj` and `ndastp` follow `start_calendar` (see [Start date](#start-date)), `ntime` stays 0 |
| `nav_lev`, `lon`, `lat`, `time_counter` | coordinates | not read | metadata; `time_counter` sets `nn_it000` in the pipeline |

**Not written**, initialised by NEMO:

| Variable | NEMO code | Value at start |
| --- | --- | --- |
| `tb`, `sb`, `ub`, `vb` | `restart.F90:302-306` | = now fields |
| `sshb` | `restart.F90:383-386` | = `sshn` |
| `rhop` | `restart.F90:318-330` | computed by `eos` from `tn`, `sn`: S-EOS at `z = 0` (`eosbn2.F90:455-466`) |
| `utau_b`, `vtau_b`, `qns_b`, `emp_b`, `sfx_b` | `sbcmod.F90:540-559` | = fluxes of the first step, computed from the generated state |
| `sbc_hc_b`, `sbc_sc_b` | `trasbc.F90:113-127` | 0, Euler forward weight (`zfact = 1`) |
| `qsr_hc_b` | `traqsr.F90:139-150` | 0, Euler forward weight (`z1_2 = 1`) |
| `fraqsr_1lev` | `traqsr.F90:444-447` | 1, recomputed each step; only feeds `frq_m` (SAS / coupled), no effect here |
| `en`, `avt_k`, `avm_k`, `dissl` | `zdftke.F90:866-882` (all four or none) | `en = rn_emin = 1e-6` m2/s2, `dissl = 1e-12`, `avt_k = rn_avt0 = 1.2e-5`, `avm_k = rn_avm0 = 1.2e-4` m2/s (uniform: `nn_avb = nn_havtb = 0`; set in `zdfphy.F90:172-174`) |
| `rdt` | `domain.F90:416` | not checked |
| `ub2_b`, `un_bf`, ... (time splitting) | `dynspg_ts.F90:937-964` | 0; already the case in the previous runs (`ln_bt_fw = F`, not in the template) |

**Not in the restart**, derived by NEMO (z-star with QCO, `ln_linssh = F`):

| Variable | Value at start |
| --- | --- |
| `e3t`, `e3u`, `e3v` | `e3t = e3t_0 * (1 + r3t * tmask)` with `r3t = ssh / ht_0`, `e3u = e3u_0 * (1 + r3u * umask)` with `r3u` the area-weighted mean of `ssh` at the U point divided by `hu_0` (same for V) (`domzgr_substitute.h90:14-16`, `domqco.F90:159-172`). Before = now since `sshb = sshn`. The saved `e3t` of the `data` run matches to 3e-5 m (*numerical*) |

`rhop` is overwritten every step by `eos` before the pressure gradient (`stpmlf.F90:234`). Between the read and this call, it is only used by diagnostics or by options that are off in DINO (`tramle`: `ln_mle = F`). The `z = gdept_0` error of the previous protocol therefore had no effect on the dynamics.

## Previous protocol (template restart)

Runs done so far:

- **C, D.** `tb = tn`, `sb = sn`, `sshb = sshn`, `ub = vb = 0` and `rhop = seos(T, S, z = gdept_0)` were also written.
- **E.** The template was copied and only `tn tb sn sb un ub vn vb rhop sshn sshb` were replaced. Everything else came from the spun-up run: surface flux memory (`utau_b`, ..., `qsr_hc_b`, `fraqsr_1lev`), TKE state (`en`, `avt_k`, `avm_k`, `dissl`), `rdt` and the calendar. The template variables `x(y, x)`, `y(y, x)` were dropped by `open_restart` and replaced by `lon`, `lat`.
- **F.** `ln_1st_euler = F`: leapfrog first step with before = now, all the `_b` fields read from the restart (log lines 2052, 2169, 2175).
- **`rhop`.** Stored with the term `mu1 * z`, off by `a0 * mu1 * z * (10 - T)` from NEMO's definition: too heavy where T < 10°C, up to 0.78 kg/m3 at depth (*numerical*). Read by NEMO (log line 1499) but without effect on the dynamics (see above).

The flux memory had an effect for one step. The TKE state had an effect for hours to days: mixing matched the spun-up stratification, not the generated one.

## Notes

- Effective initial state: generated T, S, SSH upsampled from 1°, ocean at rest, minimum TKE and background mixing, calendar 0249/12/01 and forcing day 0 (forcing day `start_day` with a `start_day`).
- SSH with zero velocity is unbalanced, so the run starts with a geostrophic adjustment. The `data` run measures this.
- The odd levels and level 35 are interpolated, not generated.
- Masks used: training (B), 1° `mesh_mask` (D), 1/4° `mesh_mask` (D), and the selection mask file (A). Whether the 1° masks are identical was not checked; any mismatch is filled by D's nearest extrapolation.
- The figures and losses use TEOS-10 sigma_0, while the model runs S-EOS. TEOS-10 is 0.76 to 0.86 kg/m3 heavier, and the vertical contrast is nearly equal (2.62 against 2.59) (*numerical*).

## Open questions

| Question | How to settle it |
| --- | --- |
| Is the NEMO binary of the `diffusion` branch built from unmodified 4.2 sources? | `ocean.output` of the first minimal-restart run: `Kbb u, v and T-S fields set to Kmm values`, `nit000-1 surface forcing fields set to nit000`, `previous run without TKE scheme, set en to background values` |
| `y = 0` is the land row? | training mask at `y = 0` |
| Training mask = 1° `mesh_mask.tmask`? | compare them |