# Rest start from a diffusion generation (proposal)

**Not implemented.** Alternative to [restart_initialisation.md](restart_initialisation.md). The generated T, S are used like a climatology in a classical spin-up: NEMO starts from rest (`ln_rstart = F`) and reads only T and S. Nothing comes from the spun-up 1/4° run `DINO_10800000_restart.nc`, so the run tests the generated state alone.

Sources: namelists of the Dinonline branch `diffusion` (`restart0/namelist_cfg`, `restart0/namelist_ref`, `restart0/jobs/launch.sh`). The NEMO source is not available locally: *unverified* = depends on NEMO internals, to be confirmed in the `ocean.output` of a first test run.

## What changes

| | Current (restart) | Rest start |
| --- | --- | --- |
| T, S | generated | generated |
| SSH | generated | 0 (generated SSH lost) |
| U, V | 0 | 0 |
| TKE, `avt`, `avm` | spun-up template | minimum TKE, background mixing |
| surface flux memory (`*_b`) | spun-up template | none, first step Euler forward |
| calendar | template: date 0249/12/01, forcing day 0 (1 January, set by `kt`) | `nn_date0`, 1 January (`010101` by default) |
| `rhop`, `e3t` | `rhop` written (wrong `z`), `e3t` from SSH | computed by NEMO, `e3t = e3t_0` since SSH = 0 |
| first time step | 10800001 | 1 |

The forcing has an annual cycle (`ln_ann_cyc = T` in `namusr_def`), so the start matters. The DINO forcing takes its phase from the time step only, `(kt / 120) mod 360` days (`usrdef_sbc.F90`, `compute_day_of_year`), not from the date. The template restart has `kt = 10800000` = 250 years of 360 days, so its runs start on forcing day 0 (1 January) although their calendar date is 0249/12/01 (`ndastp = 0249/11/30`, `adatrj = 90000` days; see [restart_initialisation.md](restart_initialisation.md#start-date)). A rest start with `nn_it000 = 1` also starts on forcing day 0 (1 January) whatever `nn_date0`.

## Pipeline

```python
# A, B, D : unchanged (median member, un-normalisation, regridding to 1/4°), see restart_initialisation.md
T, S, ssh = unnormalise(median_member(generated))      # 1°, (36, 199, 62)
T4, S4 = regrid_1deg_to_1_4deg(T, S)                   # same fill_nearest + xesmf bilinear, 0 on 1/4° land
# ssh is dropped

# C'. T/S initial file -> ts_1_4deg_<method>.nc (replaces the restart files)
ds = Dataset(dims=('time_counter', 'z', 'y', 'x'))     # 1 record, 1/4° model grid (797 x 242), no weights
ds.votemper = T4[None]                                 # names of sn_tem / sn_sal in namtsd
ds.vosaline = S4[None]
# *unverified*: with ln_sco, dta_tsd is believed to interpolate vertically from the 1D data levels gdept_1d
# to the 3D model levels gdept_0. T4, S4 are on the model levels, so they may first need to be interpolated
# onto gdept_1d (or the NEMO interpolation be checked to be harmless).

# F'. namelist_cfg
namrun:  ln_rstart = .false.
         nn_it000 = 1;  nn_itend = NSTEPS               # 432000 = 10 years
         nn_date0 = 010101                              # date only; the forcing starts on day 0 (1 January) from nn_it000 = 1, as the current runs
namtsd:  ln_tsd_init = .true.
         sn_tem = 'ts_1_4deg_<method>.nc', ..., 'votemper', ...
         sn_sal = 'ts_1_4deg_<method>.nc', ..., 'vosaline', ...
# nn_initcase = 4 (analytic usr_def_istate) is believed to be ignored when ln_tsd_init = T (*unverified*)
# jobs/launch.sh exits if restart.config has no cn_ocerst_in line: keep a dummy one (unused with ln_rstart = F)
```

## Initial state seen by NEMO

What NEMO is believed to do at a rest start (*unverified*, standard `istate_init`, `zdf_tke_init`, `sbc_init` behaviour; the `ocean.output` of the first run shows it).

| Variable | Value at start |
| --- | --- |
| `tn`, `tb`, `sn`, `sb` | generated T, S read from `ts_1_4deg_<method>.nc` (`dta_tsd`), before = now |
| `sshn`, `sshb` | 0 |
| `un`, `ub`, `vn`, `vb` | 0 |
| `e3t`, `e3u`, `e3v` | `e3*_0`, since SSH = 0 |
| `rhop` | computed by `eos` from T, S |
| `en` | `rn_emin = 1e-6` m2/s2 (`namelist_ref`, `namzdf_tke`) |
| `avt`, `avm` | background `rn_avt0 = 1.2e-5`, `rn_avm0 = 1.2e-4` m2/s (`namzdf`) |
| `dissl` | derived from the minimum TKE |
| surface flux memory (`utau_b`, `qns_b`, `emp_b`, ...) | none: the first step is Euler forward, the fluxes are computed at step 1 |
| `kt`, date | `nn_it000 = 1`, `nn_date0 = 010101`, `nn_time0 = 0` |

`nn_istate = 1` (already in the branch namelist) writes `output.init`, which shows this state directly.

## Trade-offs

- **Fairer test:** nothing but T, S comes from a spun-up run. TKE, mixing and the forcing memory spin up from the generated state.
- **SSH lost:** the generated SSH is not used. It starts at 0 and adjusts within days (barotropic adjustment), which is also what a classical spin-up from a climatology does.
- **Harsher start:** minimum TKE plus zero velocity gives a stronger initial shock than the current protocol. The `data` run must use the same protocol, otherwise generated and data runs are not comparable.
- **Vertical interpolation risk:** `dta_tsd` with s-coordinates may reinterpolate T, S (see C'). Check this before trusting the result.
- **Cost:** one new writer (`ts_1_4deg_<method>.nc` instead of `restart_1_4deg_<method>.nc`) and a different namelist edit in `3_launch_nemo.py`. Steps A, B, D are reused.

## To check on a first test run

| Question | Where to look |
| --- | --- |
| Is `ln_tsd_init` read and is the analytic `usr_def_istate` skipped? | `ocean.output` (`istate_init`, `dta_tsd`) |
| Does `dta_tsd` reinterpolate vertically with `ln_sco`? | `dtatsd.F90`, or `output.init` T vs `ts_1_4deg_<method>.nc` |
| TKE and mixing at start | `ocean.output` (`zdf_tke_init`), `output.init` |
| First step Euler forward | `ocean.output` |
| Calendar and annual cycle phase | `ocean.output` (date at `nit000`) |
