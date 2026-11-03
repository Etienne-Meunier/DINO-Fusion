#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 of the assimilation study (see plan.md): analysis of the 2-week NEMO DINO 1/4 deg integrations started from
the initial states of 2_select.py (one per season and config, a posteriori pipeline with --prefix assimilation/phase2
--experiment daily_2w_minimal). Every run is compared with the `data` run of its season, started from the source state.

    python 2_integration.py           # once the results are pulled (a posteriori step 5)
    python 2_integration.py --test    # code check: the 2-month daily data run of the optimal sweep stands for every run

Day 0 is the initial state, read from the 1/4 deg restart of the run (tn level 0 as SST, sshn); day k >= 1 is the
daily mean of day k, drawn at k - 0.5. All statistics are area-weighted (e1t * e2t of the 1/4 deg mesh) over the ocean
(tmask), without the periodic halo columns x = 0 and x = -1. The NH band is seasonal.NH_BAND (15-70N, gphit).

Products, per season:
    a. RMSE of SST and of SSH to the data run, days 0 to NDAYS, and of SSH with the area-weighted domain mean of the
       difference removed (SSH')
    b. maps of SST and SSH at day NDAYS for each config, and of their difference to the data run
    c. NH-band area-weighted T profile anomaly to the data run at days DAYS (0, 1, NDAYS); and the summer - winter
       split of the band T over time at the 1/4 deg levels nearest the first N_CHANNELS model T channels (5-102 m)
    d. printed tables: per season and config, SST, SSH and SSH' RMSE to data at days 1 and NDAYS and the band T minus
       the data run at the levels nearest TABLE_DEPTHS (49 and 74 m) at days DAYS; per config, the summer - winter
       band T split at the same levels and days
    e. maps at days SST_DAYS (0 to 4), two rows per config: SST, and D the relative change of the sigma_0 profile of
       each water column from day k - 1 to day k,
           D(k) = sum_z |sigma_0(k-1, z) - sigma_0(k, z)| e3t_0(z) / sum_z |sigma_0(k-1, z)| e3t_0(z)
       over the ocean levels (tmask) of the column, blank at day 0. One colour range per season and field, the
       1st-99th percentile over the ocean of all its panels (positive values of D only), log scale for D if the range
       spans more than D_LOG_SPAN; the same with T at model channel T_CHANNEL (26 m, the 1/4 deg level of the channel
       map below) in place of SST, its range over the ocean of that level
    f. SST drift of each run from its own initial state, RMSE of SST(k) - SST(0) (area-weighted as in a.), days 0 to
       NDAYS at integer days; printed at DRIFT_DAYS
    g. SST drift of f. at day E_DRIFT_DAY against the density error E of the initial state of the run (% of the ocean
       volume where sigma_0 decreases with depth, metrics.compute_density_error on the 1 deg toce, soce of the
       integrated member, in float64 with the 1 deg MASK_FILE as in 1_profiles.py); and the same with the RMSE of the
       full 3D T between day E_DRIFT_DAY and day 0, volume-weighted (e1t * e2t * e3t_0) over all ocean cells (tmask)
    h. area-weighted horizontal-mean sigma_0 and T profiles over the ocean cells (tmask per level) of the box BOX_LAT
       (gphit) x centre +/- BOX_HALF_WIDTH (glamt), the centre the midpoint of the glamt range of the surface ocean
       (halo excluded), at days BOX_DAYS; printed at the levels nearest BOX_TABLE_DEPTHS at days 0 and NDAYS
    i. zonal-mean T sections (latitude the zonal mean of gphit per row, halo excluded) averaged over the ocean cells
       (tmask) weighted by e1t * e3t_0, depth stretched as d**DEPTH_POWER (paper fig7): day 0, then T(k) - T(0) for
       k = 1..NDAYS with one symmetric colour range per season, the 99th percentile of |T(k) - T(0)| over all
       difference panels; and the difference to the data run of the season on the same day, T(k) - T_data(k) for
       k = 0..NDAYS, without the data column, with and without none, each with its own colour range

sigma_0 = potential density referenced at the surface - 1000 (kg m-3), from T and S by get_density_at_surface of
`analysis_scripts/metrics.py` (the TEOS-10 polynomial EOS used across the project, as in 1_profiles.py; the project
does not use gsw), computed in float64; the 1000 offset is removed so that the denominator of D is not dominated by it.
It is a diagnostic applied to the model T and S, not the model's own density: these runs use NEMO's simplified EOS
(ln_seos = .true.). Day 0 from the restart tn, sn; day k from toce, soce of the 3D daily means.

Inputs:
    <INFERENCE>/<prefix>/<season>_<config>/<timestamp>/aposteriori/   (settings.aposteriori_dirs, latest timestamp)
        <results>/DINO_1d_grid_T_2D.nc   daily sst, ssh (time_counter, y 797, x 242)
        <results>/DINO_1d_grid_T_3D.nc   daily toce, soce (time_counter, deptht 36, y, x)
        <restart_1_4deg>                 initial state of the run (restart of the experiment's protocol): tn, sn, sshn
        toce.npy, soce.npy               the integrated member on the 1 deg grid (save_clean of 2_select.py, the source
                                         state for data)
    MESH                                 1/4 deg mesh mask (e1t, e2t, e3t_0, tmask, gphit, glamt, gdept_1d)
    MASK_FILE                            1 deg mesh mask (seasonal.MASK_FILE), for E
    with --test: the run `data` of the optimal sweep, experiment daily_2m, for every season and config
Outputs in OUT2 (OUT2/test with --test), PDFs with seasonal.save provenance:
    rmse_timeseries.pdf       a., rows the seasons, columns SST / SSH / SSH'
    maps_day<NDAYS>_<season>.pdf   b., rows SST, SST - data, SSH, SSH - data, columns the configs
    band_t_anomaly.pdf        c., rows the seasons, columns the DAYS, top PROFILE_DEPTH m
    band_t_split.pdf          c., one panel per level
    sst_days_<season>.pdf     e., rows <config> SST and <config> D for each config, columns the SST_DAYS
    t26m_days_<season>.pdf    e., the same with rows <config> T 26 m (T at model channel T_CHANNEL)
    sst_drift.pdf             f., columns the seasons, one line per config
    e_vs_sst_drift.pdf        g., one point per run (colour the config, marker the season); left all configs, right
                              without none
    e_vs_t_drift.pdf          g., the same with the full 3D T RMSE in place of the SST RMSE
    zonal_t_<season>.pdf      i., columns the configs, rows day 0 then the differences of days 1 to NDAYS
    zonal_t_vs_data_<season>.pdf          i., columns none, B0, C7, TS, rows day 0 then T - T data at days 0 to NDAYS
    zonal_t_vs_data_nonone_<season>.pdf   i., the same without none
    box_density_profiles.pdf  h., rows the seasons over the top BOX_DEPTH m then the seasons over the top BOX_ZOOM m,
                              columns the configs, one line per day of BOX_DAYS (day 0 thick red), the data run day 0
                              dashed black
    box_temperature_profiles.pdf   h., the same for T
Printed: the tables of d., the colour ranges of e., the median of D over the ocean per season, config and day, and
the SST drift of f. at DRIFT_DAYS, E and the drift of g. per run, the box bounds and the box sigma_0 and T tables of h.,
the colour ranges of i.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/'); sys.path.append(f'{PRP}/Results/analysis_scripts/')

import argparse
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from metrics import compute_density_error, get_density_at_surface
from settings import INFERENCE, aposteriori_dirs, experiment_paths
from seasonal import LEVELS, MASK_FILE, NH_BAND, OUT, load_mask, map_panel, save

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

PREFIX2 = 'assimilation/phase2'
EXPERIMENT = 'daily_2w_minimal'
OUT2 = OUT.replace('phase0', 'phase2')
MESH = f'{os.environ["OCEANDATA"]}/Dino-Fusion-Data/Restart-format/restart25_arch/mesh_mask.nc'
TEST = dict(prefix='optimal', run='data', experiment='daily_2m')
SEASONS = ['summer', 'winter']
CONFIGS = ['data', 'none', 'B0', 'C7', 'TS']
COLORS = {'data': 'k', 'none': 'tab:red', 'B0': 'tab:orange', 'C7': 'tab:blue', 'TS': 'tab:green'}
NDAYS = 14
DAYS = (0, 1, NDAYS)
SST_DAYS = (0, 1, 2, 3, 4)
D_LOG_SPAN = 10
N_CHANNELS = 5
T_CHANNEL = 1
DRIFT_DAYS = (1, 4, 7, NDAYS)
E_DRIFT_DAY = 3
MARKERS = {'summer': 'o', 'winter': 's'}
TABLE_DEPTHS = (49, 74)
PROFILE_DEPTH = 300
BOX_LAT = (35, 45)
BOX_HALF_WIDTH = 5
BOX_DAYS = (0, 1, 2, 3, 7, NDAYS)
BOX_DEPTH = 1000
BOX_ZOOM = 400
BOX_ZOOM_MARGIN = 0.05
BOX_TABLE_DEPTHS = (50, 100, 200, 500)
DEPTH_POWER = 0.625
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000]
FILE_2D, FILE_3D = 'DINO_1d_grid_T_2D.nc', 'DINO_1d_grid_T_3D.nc'


def load_mesh():
    """1/4 deg mesh as numpy: tmask (z, y, x) bool and ocean = tmask[0], both False on the halo columns x = 0 and -1,
    area e1t * e2t and e1t, lon glamt, lat gphit (y, x), e3t e3t_0 (z, y, x), depth gdept_1d (z)."""
    m = xr.open_dataset(MESH).isel(time_counter=0)
    tmask = m.tmask.values.astype(bool)
    tmask[..., 0] = tmask[..., -1] = False
    return {'tmask': tmask, 'ocean': tmask[0], 'area': (m.e1t * m.e2t).values, 'lon': m.glamt.values,
            'lat': m.gphit.values, 'e3t': m.e3t_0.values, 'depth': m.gdept_1d.values, 'e1t': m.e1t.values}


def run_folders(test):
    """{(season, config): aposteriori/ folder} of the runs, and the experiment whose results they hold. With `test`,
    the folder of TEST['run'] of the TEST sweep for every key. Exits if a run is missing."""
    if test:
        experiment = TEST['experiment']
        found = aposteriori_dirs(INFERENCE, TEST['prefix'], experiment_paths(experiment)['results'], [TEST['run']])
        folders = {(s, c): found.get(TEST['run']) for s in SEASONS for c in CONFIGS}
    else:
        experiment = EXPERIMENT
        found = aposteriori_dirs(INFERENCE, PREFIX2, experiment_paths(experiment)['results'])
        folders = {(s, c): found.get(f'{s}_{c}') for s in SEASONS for c in CONFIGS}
    missing = [f'{s}_{c}' for (s, c), f in folders.items() if f is None]
    if missing:
        sys.exit(f'No {experiment} results for {", ".join(missing)}')
    return folders, experiment


def band_weights_1_4(mesh, band=NH_BAND):
    """Area weights (z, y, x) of the ocean cells of every level with band[0] <= gphit <= band[1], zero elsewhere
    (halo columns included, through mesh['tmask'])."""
    in_band = (mesh['lat'] >= band[0]) & (mesh['lat'] <= band[1])
    return mesh['area'] * in_band * mesh['tmask']


def box_weights(mesh):
    """Area weights (z, y, x) of the ocean cells of every level (mesh['tmask']) with BOX_LAT[0] <= gphit <= BOX_LAT[1]
    and |glamt - centre| <= BOX_HALF_WIDTH, centre the midpoint of the glamt range of mesh['ocean']; and the
    longitude bounds (lo, hi)."""
    lon = mesh['lon'][mesh['ocean']]
    centre = (lon.min() + lon.max()) / 2
    bounds = (centre - BOX_HALF_WIDTH, centre + BOX_HALF_WIDTH)
    in_box = ((mesh['lat'] >= BOX_LAT[0]) & (mesh['lat'] <= BOX_LAT[1]) & (mesh['lon'] >= bounds[0])
              & (mesh['lon'] <= bounds[1]))
    return mesh['area'] * in_box * mesh['tmask'], bounds


def band_profile(toce, w):
    """Weighted horizontal mean (..., z) of `toce` (..., z, y, x) with weights `w` (z, y, x), land values ignored, NaN
    on the levels without weight."""
    total = w.sum(axis=(-2, -1))
    return np.einsum('...kyx,kyx->...k', np.where(w > 0, toce, 0), w) / np.where(total > 0, total, np.nan)


def sigma0(toce, soce, tmask):
    """sigma_0 (z, y, x) float64 in kg m-3 of `toce`, `soce` (z, y, x): get_density_at_surface - 1000, NaN outside
    `tmask`."""
    rho = get_density_at_surface(toce.astype(np.float64), soce.astype(np.float64), tmask.astype(np.float64))
    return np.where(tmask, rho - 1000, np.nan)


def profile_change(before, after, mesh):
    """D (y, x), relative change of the sigma_0 profile of each column from `before` to `after` (z, y, x):
    sum_z |before - after| e3t / sum_z |before| e3t over the ocean levels of mesh['tmask'], NaN on columns without
    ocean."""
    w = np.where(mesh['tmask'], mesh['e3t'], 0)
    num = np.nansum(np.abs(before - after) * w, axis=0)
    den = np.nansum(np.abs(before) * w, axis=0)
    return np.where(mesh['ocean'] & (den > 0), num / np.where(den > 0, den, 1), np.nan)


def zonal_mean(field, mesh):
    """Zonal mean (z, y) of `field` (z, y, x) over the ocean cells of mesh['tmask'], weighted by e1t * e3t; NaN where
    a row of a level has no ocean."""
    w = np.where(mesh['tmask'], mesh['e1t'] * mesh['e3t'], 0)
    total = w.sum(-1)
    return (np.where(w > 0, field, 0) * w).sum(-1) / np.where(total > 0, total, np.nan)


def stretch(d):
    """Stretched depth axis coordinate d**DEPTH_POWER of depths `d` (m), as paper fig7."""
    return np.asarray(d, dtype=float) ** DEPTH_POWER


def load_run(folder, experiment, w_band, mesh, t_level, w_box):
    """Fields of the run of `folder` (aposteriori/) for `experiment`: sst, ssh (NDAYS + 1, y, x), day 0 the initial
    restart, then the first NDAYS daily means; band (NDAYS + 1, z) the NH-band T profile (band_profile with
    `w_band`); depth (z) the T levels of the 3D output; d (max(SST_DAYS) + 1, y, x) the profile_change of sigma_0
    from day k - 1 to day k, NaN at day 0; t_level (max(SST_DAYS) + 1, y, x) T at the 1/4 deg level `t_level`, day 0
    from the restart, then the daily means; box (len(BOX_DAYS), z) the sigma_0 profile (band_profile with `w_box`) at
    the BOX_DAYS; box_t (len(BOX_DAYS), z) the same for T; t_drift (NDAYS + 1) the RMSE of T to day 0 over all
    ocean cells (mesh['tmask']), weighted by the cell volume e3t * area, 0 at day 0; zonal (NDAYS + 1, z, y) the zonal
    mean of T (zonal_mean)."""
    paths = experiment_paths(experiment)
    results = f'{folder}/{paths["results"]}'
    with xr.open_dataset(f'{folder}/{paths["restart_1_4deg"]}', decode_times=False) as r:
        tn = r.tn.isel(time_counter=0).values
        sn = r.sn.isel(time_counter=0).values
        sst0, ssh0 = tn[0], r.sshn.isel(time_counter=0).values
    with xr.open_dataset(f'{results}/{FILE_2D}', decode_times=False) as d2:
        if d2.sizes['time_counter'] < NDAYS:
            raise ValueError(f'{results}/{FILE_2D} : {d2.sizes["time_counter"]} days < {NDAYS}')
        days = d2.isel(time_counter=slice(0, NDAYS))
        sst, ssh = days.sst.values, days.ssh.values
    with xr.open_dataset(f'{results}/{FILE_3D}', decode_times=False) as d3:
        band = [band_profile(tn, w_band)]
        box_t = {0: band_profile(tn, w_box)}
        volume = np.where(mesh['tmask'], mesh['e3t'] * mesh['area'], 0)
        t_drift = [0.]
        zonal = [zonal_mean(tn, mesh)]
        for t in range(NDAYS):
            toce = d3.toce.isel(time_counter=t).values
            band.append(band_profile(toce, w_band))
            t_drift.append(np.sqrt((np.where(volume > 0, toce - tn, 0) ** 2 * volume).sum() / volume.sum()))
            zonal.append(zonal_mean(toce, mesh))
            if t + 1 in BOX_DAYS:
                box_t[t + 1] = band_profile(toce, w_box)
        depth = d3.deptht.values
        d = [np.full(tn.shape[1:], np.nan)]
        previous = sigma0(tn, sn, mesh['tmask'])
        box = {0: band_profile(previous, w_box)}
        for t in range(max(SST_DAYS)):
            day = d3.isel(time_counter=t)
            current = sigma0(day.toce.values, day.soce.values, mesh['tmask'])
            d.append(profile_change(previous, current, mesh))
            box[t + 1] = band_profile(current, w_box)
            previous = current
        for k in set(BOX_DAYS) - set(box):
            day = d3.isel(time_counter=k - 1)
            box[k] = band_profile(sigma0(day.toce.values, day.soce.values, mesh['tmask']), w_box)
        t_days = d3.toce.isel(time_counter=slice(0, max(SST_DAYS)), deptht=t_level).values
    return {'sst': np.concatenate([sst0[None], sst]), 'ssh': np.concatenate([ssh0[None], ssh]),
            'band': np.stack(band), 'depth': depth, 'd': np.stack(d),
            't_level': np.concatenate([tn[t_level][None], t_days]), 'box': np.stack([box[k] for k in BOX_DAYS]),
            'box_t': np.stack([box_t[k] for k in BOX_DAYS]), 't_drift': np.array(t_drift),
            'zonal': np.stack(zonal)}


def initial_density_error(folder, mask):
    """E (%) of the 1 deg toce.npy, soce.npy (1, 36, 199, 62) of `folder` (aposteriori/), by compute_density_error
    with `mask` (MASK_FILE opened as in 1_profiles.py), fields in float64."""
    fields = {k: np.load(f'{folder}/{k}.npy').astype(np.float64) for k in ('toce', 'soce')}
    return float(compute_density_error(fields, mask)[0])


def area_rmse(a, b, w, demean=False):
    """Area-weighted RMSE (t) of `a` - `b` (t, y, x) over the cells with `w` (y, x) > 0; with `demean`, of the
    difference minus its area-weighted domain mean at each time."""
    diff = np.where(w > 0, a - b, 0)
    if demean:
        diff = np.where(w > 0, diff - (np.einsum('tyx,yx->t', diff, w) / w.sum())[:, None, None], 0)
    return np.sqrt(np.einsum('tyx,yx->t', diff ** 2, w) / w.sum())


def day_axis():
    """Plot abscissa of the NDAYS + 1 records: 0 for the initial state, k - 0.5 for the daily mean of day k."""
    return np.concatenate([[0.], np.arange(1, NDAYS + 1) - 0.5])


def plot_rmse(rmses, path, sources):
    """a.: rows the seasons, columns SST / SSH / SSH with the domain-mean difference removed, one line per config
    except data."""
    fields = [('sst', 'SST', 'degC'), ('ssh', 'SSH', 'm'), ('ssh_demeaned', 'SSH - domain mean', 'm')]
    fig, axs = plt.subplots(len(SEASONS), len(fields), figsize=(9.6, 2.2 * len(SEASONS)), sharex=True, squeeze=False)
    for i, s in enumerate(SEASONS):
        for j, (field, name, unit) in enumerate(fields):
            ax = axs[i, j]
            for c in CONFIGS[1:]:
                ax.plot(day_axis(), rmses[s, c][field], marker='.', color=COLORS[c], label=c)
            ax.set_title(f'{s.capitalize()}, {name} RMSE to data', loc='left')
            ax.set_ylabel(f'RMSE ({unit})')
            ax.grid(True, linestyle='--', alpha=0.5)
    for ax in axs[-1]:
        ax.set_xlabel('day (0 = initial state, daily means at k - 0.5)')
    axs[0, 0].legend()
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


def plot_maps(runs, season, mesh, path, sources):
    """b.: rows SST, SST - data, SSH, SSH - data at day NDAYS, columns the configs; shared colour range per row."""
    ref = runs[season, 'data']
    rows = []
    for field, unit in [('sst', 'degC'), ('ssh', 'm')]:
        values = [runs[season, c][field][NDAYS] for c in CONFIGS]
        diffs = [v - ref[field][NDAYS] for v in values]
        ocean = mesh['ocean']
        lo, hi = np.nanpercentile(np.concatenate([v[ocean] for v in values]), [1, 99])
        dmax = np.nanpercentile(np.abs(np.concatenate([d[ocean] for d in diffs[1:]])), 99) or 1e-12
        rows.append((values, f'{field.upper()} ({unit})', 'RdYlBu_r', dict(vmin=lo, vmax=hi)))
        rows.append((diffs, f'{field.upper()} - data ({unit})', 'RdBu_r', dict(vmax=dmax, symmetric=True)))
    fig, axs = plt.subplots(len(rows), len(CONFIGS), figsize=(2.1 * len(CONFIGS), 3.6 * len(rows)), sharex=True,
                            sharey=True, squeeze=False)
    for i, (fields, label, cmap, scale) in enumerate(rows):
        for j, c in enumerate(CONFIGS):
            map_panel(axs[i, j], fields[j], mesh, cmap, label, **scale)
            axs[i, j].set_title(f'{season} {c}, day {NDAYS}', loc='left')
        axs[i, 0].set_ylabel('Latitude (deg)')
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight', dpi=200)
    plt.close(fig)


def colour_range(values, positive=False):
    """1st and 99th percentiles of the finite `values` (1D), of their positive part only with `positive`, and the
    colourbar extension that flags the values beyond them."""
    values = values[np.isfinite(values) & ((values > 0) if positive else True)]
    lo, hi = np.percentile(values, [1, 99])
    extend = {(False, False): 'neither', (True, False): 'min', (False, True): 'max', (True, True): 'both'}[
        (values.min() < lo, values.max() > hi)]
    return lo, hi, extend


def plot_days(runs, season, mesh, field, name, level, path, sources):
    """e.: maps of `season`, columns the SST_DAYS (record index = day, see load_run), two rows per config: the load_run
    `field` (y, x) in degC, labelled `name`, and D (load_run 'd', blank axis at day 0). `field` is drawn over the
    ocean of the 1/4 deg level `level` (mesh['tmask'][level]), D over the surface ocean. One colour range and
    colourbar per field over all its panels (colour_range over its ocean, positive D only), log scale for D if
    hi / lo > D_LOG_SPAN. Returns {'t': (lo, hi), 'd': (lo, hi, log)}."""
    ocean = mesh['ocean']
    t_mesh = dict(mesh, ocean=mesh['tmask'][level])
    t = {c: [runs[season, c][field][day] for day in SST_DAYS] for c in CONFIGS}
    d = {c: [runs[season, c]['d'][day] for day in SST_DAYS] for c in CONFIGS}
    t_lo, t_hi, t_extend = colour_range(np.concatenate([f[t_mesh['ocean']] for c in CONFIGS for f in t[c]]))
    d_lo, d_hi, d_extend = colour_range(np.concatenate([f[ocean] for c in CONFIGS for f in d[c][1:]]), positive=True)
    log = d_hi / d_lo > D_LOG_SPAN
    rows = [(c, row) for c in CONFIGS for row in (name, 'D')]
    fig, axs = plt.subplots(len(rows), len(SST_DAYS), figsize=(0.95 * len(SST_DAYS) + 1.1, 2.15 * len(rows)),
                            sharex=True, sharey=True, squeeze=False)
    for i, (c, row) in enumerate(rows):
        for j, day in enumerate(SST_DAYS):
            ax = axs[i, j]
            if row == name:
                pm_t = map_panel(ax, t[c][j], t_mesh, 'RdYlBu_r', '', vmin=t_lo, vmax=t_hi, colorbar=False)
            elif day == 0:
                ax.set_aspect('equal')
                ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                pm_d = map_panel(ax, d[c][j], mesh, 'viridis', '', vmin=d_lo, vmax=d_hi, colorbar=False, log=log)
            ax.set_xlabel('Longitude (deg)' if i == len(rows) - 1 and j == len(SST_DAYS) // 2 else '')
            if i == 0:
                ax.set_title(f'day {day}')
        axs[i, 0].set_ylabel(f'{c} {row}' + ('\nLatitude (deg)' if row == name else ''))
    fig.suptitle(f'{season.capitalize()}', x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 0.86, 0.99))
    fig.colorbar(pm_t, cax=fig.add_axes((0.88, 0.55, 0.015, 0.35)), label=f'{name} (degC)', extend=t_extend)
    fig.colorbar(pm_d, cax=fig.add_axes((0.88, 0.1, 0.015, 0.35)), label='relative sigma_0 profile change',
                 extend=d_extend)
    save(fig, path, sources=sources, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return {'t': (t_lo, t_hi), 'd': (d_lo, d_hi, log)}


def plot_sst_drift(drift, path, sources):
    """f.: columns the seasons, `drift[season, config]` (NDAYS + 1) against the day (integer, 0 the initial state, k
    the daily mean of day k), one line per config, shared y axis."""
    fig, axs = plt.subplots(1, len(SEASONS), figsize=(6.4, 2.6), sharey=True, squeeze=False)
    for ax, s in zip(axs[0], SEASONS):
        for c in CONFIGS:
            ax.plot(np.arange(NDAYS + 1), drift[s, c], marker='.', color=COLORS[c], label=c)
        ax.set_title(f'{s.capitalize()}, SST RMSE to day 0', loc='left')
        ax.set_xlabel('day')
        ax.set_xticks(np.arange(0, NDAYS + 1, 2))
        ax.grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].set_ylabel('RMSE (degC)')
    axs[0, 0].legend()
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


def plot_e_vs_drift(e, drift, ylabel, path, sources):
    """g.: one point per run (season, config), x `e[run]` (%), y `drift[run]` labelled `ylabel`; colour COLORS of the
    config, marker MARKERS of the season. Left all CONFIGS, right without none, each panel autoscaled to its points."""
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), squeeze=False)
    for ax, configs in zip(axs[0], [CONFIGS, [c for c in CONFIGS if c != 'none']]):
        for s in SEASONS:
            for c in configs:
                ax.scatter(e[s, c], drift[s, c], color=COLORS[c], marker=MARKERS[s])
        ax.set_title('all configs' if 'none' in configs else 'without none', loc='left')
        ax.set_xlabel('E of the initial state (% of ocean volume)')
        ax.grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].set_ylabel(ylabel)
    handles = [Line2D([], [], color=COLORS[c], marker='o', linestyle='', label=c) for c in CONFIGS]
    handles += [Line2D([], [], color='gray', marker=MARKERS[s], linestyle='', label=s) for s in SEASONS]
    axs[0, 0].legend(handles=handles)
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


def plot_box_profiles(runs, field, label, depth, path, sources):
    """h.: columns the configs, rows the seasons over the top BOX_DEPTH m, then the seasons over the top BOX_ZOOM m with
    x limits the range of the plotted values above BOX_ZOOM m in the row, widened by BOX_ZOOM_MARGIN of it on each
    side. Each panel: the box profile load_run `field` ('box' or 'box_t') at each of BOX_DAYS, day 0 thick red and the
    other days on viridis, and the day 0 profile of the data run of the season dashed black; x labelled `label`,
    shared per row."""
    rows = [(s, BOX_DEPTH) for s in SEASONS] + [(s, BOX_ZOOM) for s in SEASONS]
    fig, axs = plt.subplots(len(rows), len(CONFIGS), figsize=(2.0 * len(CONFIGS), 3.0 * len(rows)), sharey='row',
                            squeeze=False)
    colours = ['tab:red', *plt.cm.viridis(np.linspace(0, 0.9, len(BOX_DAYS) - 1))]
    for i, (s, bottom) in enumerate(rows):
        top = depth <= bottom
        reference = runs[s, 'data'][field][0][top]
        for j, c in enumerate(CONFIGS):
            ax = axs[i, j]
            if j:
                ax.sharex(axs[i, 0])
            for profile, day, colour in zip(runs[s, c][field], BOX_DAYS, colours):
                ax.plot(profile[top], depth[top], color=colour, linewidth=2.2 if day == 0 else 1, label=f'day {day}')
            ax.plot(reference, depth[top], 'k--', linewidth=1, label='data day 0')
            ax.set_title(f'{s.capitalize()}, 0-{bottom} m, {c}', loc='left')
            ax.set_xlabel(label)
            ax.grid(True, linestyle='--', alpha=0.5)
        axs[i, 0].set_ylabel('depth (m)')
        axs[i, 0].set_ylim(bottom, 0)
        if bottom == BOX_ZOOM:
            values = np.concatenate([reference, *(runs[s, c][field][:, top].ravel() for c in CONFIGS)])
            lo, hi = np.nanmin(values), np.nanmax(values)
            axs[i, 0].set_xlim(lo - BOX_ZOOM_MARGIN * (hi - lo), hi + BOX_ZOOM_MARGIN * (hi - lo))
    axs[0, -1].legend()
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


def plot_zonal_t(runs, season, mesh, depth, path, sources, configs=CONFIGS, vs_data=False):
    """i.: columns `configs`, against the latitude (zonal mean of gphit over the interior columns) and the stretched
    depth (full depth). Top row the zonal-mean T (load_run 'zonal') of day 0 on one sequential colour range
    (colour_range), then one row per day of differences on one symmetric RdBu_r range, the 99th percentile of their
    absolute value over the panels shown: T(k) - T(0) of the same run for k = 1..NDAYS, or with `vs_data`
    T(k) - T_data(k), the data run of `season`, for k = 0..NDAYS. Returns the difference limit (degC)."""
    lat = mesh['lat'][:, 1:-1].mean(1)
    y = stretch(depth)
    zonal = {c: runs[season, c]['zonal'] for c in configs}
    if vs_data:
        days = np.arange(NDAYS + 1)
        diffs = {c: zonal[c] - runs[season, 'data']['zonal'] for c in configs}
        diff_label = 'T - T data (degC)'
    else:
        days = np.arange(1, NDAYS + 1)
        diffs = {c: zonal[c][1:] - zonal[c][:1] for c in configs}
        diff_label = 'T(k) - T(0) (degC)'
    t_lo, t_hi, t_extend = colour_range(np.concatenate([zonal[c][0].ravel() for c in configs]))
    all_diffs = np.abs(np.concatenate([diffs[c].ravel() for c in configs]))
    dmax = np.percentile(all_diffs[np.isfinite(all_diffs)], 99)
    d_extend = 'both' if np.nanmax(all_diffs) > dmax else 'neither'
    n_rows = len(days) + 1
    fig, axs = plt.subplots(n_rows, len(configs), figsize=(1.9 * len(configs) + 1.0, 1.25 * n_rows), sharex=True,
                            sharey=True, squeeze=False)
    for j, c in enumerate(configs):
        pm_t = axs[0, j].pcolormesh(lat, y, zonal[c][0], cmap='viridis', vmin=t_lo, vmax=t_hi, shading='nearest',
                                    rasterized=True)
        axs[0, j].set_title(c)
        for i, diff in enumerate(diffs[c], start=1):
            pm_d = axs[i, j].pcolormesh(lat, y, diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax, shading='nearest',
                                        rasterized=True)
        axs[-1, j].set_xlabel('Latitude (deg)')
    axs[0, 0].set_ylabel('day 0 T\ndepth (m)')
    for i, day in enumerate(days, start=1):
        axs[i, 0].set_ylabel(f'day {day}\ndepth (m)')
    axs[0, 0].set_ylim(stretch(np.nanmax(depth)), 0)
    axs[0, 0].set_yticks(stretch(DEPTH_TICKS))
    axs[0, 0].set_yticklabels([str(d) for d in DEPTH_TICKS])
    fig.suptitle(f'{season.capitalize()}, zonal-mean T' + (', difference to the data run' if vs_data else ''),
                 x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 0.88, 0.99))
    top = axs[0, -1].get_position()
    fig.colorbar(pm_t, cax=fig.add_axes((0.9, top.y0, 0.012, top.height)), label='T day 0 (degC)', extend=t_extend)
    fig.colorbar(pm_d, cax=fig.add_axes((0.9, 0.2, 0.012, 0.5)), label=diff_label, extend=d_extend)
    save(fig, path, sources=sources, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return dmax


def plot_band_anomaly(runs, depth, path, sources):
    """c.: rows the seasons, columns the DAYS, NH-band T anomaly to the data run vs depth, one line per config, top
    PROFILE_DEPTH m."""
    fig, axs = plt.subplots(len(SEASONS), len(DAYS), figsize=(9.6, 2.6 * len(SEASONS)), sharey=True, squeeze=False)
    top = depth <= PROFILE_DEPTH
    for i, s in enumerate(SEASONS):
        for j, day in enumerate(DAYS):
            ax = axs[i, j]
            for c in CONFIGS:
                anomaly = runs[s, c]['band'][day] - runs[s, 'data']['band'][day]
                ax.plot(anomaly[top], depth[top], marker='.', color=COLORS[c], label=c)
            ax.axvline(0, color='gray', linewidth=0.8)
            ax.set_title(f'{s.capitalize()}, day {day}', loc='left')
            ax.set_xlabel('NH-band T - data run (degC)')
            ax.grid(True, linestyle='--', alpha=0.5)
        axs[i, 0].set_ylabel('depth (m)')
    axs[0, 0].set_ylim(PROFILE_DEPTH, 0)
    axs[0, 0].legend()
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


def plot_split(runs, depth, levels, path, sources):
    """c.: one panel per level of `levels`, summer - winter NH-band T over the days, one line per config."""
    fig, axs = plt.subplots(1, len(levels), figsize=(2.0 * len(levels), 2.4), sharex=True, squeeze=False)
    for ax, k in zip(axs[0], levels):
        for c in CONFIGS:
            split = runs['summer', c]['band'][:, k] - runs['winter', c]['band'][:, k]
            ax.plot(day_axis(), split, marker='.', color=COLORS[c], label=c)
        ax.set_title(f'{depth[k]:.0f} m', loc='left')
        ax.set_xlabel('day')
        ax.grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].set_ylabel('summer - winter NH-band T (degC)')
    axs[0, 0].legend()
    fig.tight_layout()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of the phase 2 NEMO integrations')
    parser.add_argument('--test', action='store_true', help=f'use the {TEST["run"]} run of the {TEST["prefix"]} sweep '
                        f'({TEST["experiment"]}) for every season and config, figures in {OUT2}/test')
    args = parser.parse_args()

    out = f'{OUT2}/test' if args.test else OUT2
    folders, experiment = run_folders(args.test)
    mesh = load_mesh()
    w_band = band_weights_1_4(mesh)
    w_box, box_lon = box_weights(mesh)
    w_surface = mesh['area'] * mesh['ocean']
    channel_depths = load_mask(interior=False)['depth'][LEVELS[:N_CHANNELS]]
    channels = np.abs(mesh['depth'][:, None] - channel_depths[None]).argmin(0)

    loaded = {}
    runs = {}
    for key, folder in folders.items():
        if folder not in loaded:
            loaded[folder] = load_run(folder, experiment, w_band, mesh, channels[T_CHANNEL], w_box)
        runs[key] = loaded[folder]
    depth = next(iter(loaded.values()))['depth']
    table_levels = np.abs(depth[:, None] - np.array(TABLE_DEPTHS)[None]).argmin(0)
    rmses = {(s, c): {f: area_rmse(runs[s, c][f], runs[s, 'data'][f], w_surface) for f in ['sst', 'ssh']}
             for s, c in runs}
    for s, c in runs:
        rmses[s, c]['ssh_demeaned'] = area_rmse(runs[s, c]['ssh'], runs[s, 'data']['ssh'], w_surface, demean=True)
    drift = {(s, c): area_rmse(runs[s, c]['sst'], runs[s, c]['sst'][:1], w_surface) for s, c in runs}
    results = experiment_paths(experiment)['results']
    sources = [MESH, *(f'{f}/{results}' for f in dict.fromkeys(folders.values()))]
    mask_1deg = xr.open_dataset(MASK_FILE).sel(time_counter=0).rename({'nav_lev': 'depth', 'y': 'nav_lat',
                                                                       'x': 'nav_lon'})
    e_initial = {key: initial_density_error(folder, mask_1deg) for key, folder in folders.items()}
    drift_day = {key: d[E_DRIFT_DAY] for key, d in drift.items()}
    t_drift_day = {key: run['t_drift'][E_DRIFT_DAY] for key, run in runs.items()}

    plot_rmse(rmses, f'{out}/rmse_timeseries.pdf', sources)
    for s in SEASONS:
        plot_maps(runs, s, mesh, f'{out}/maps_day{NDAYS}_{s}.pdf', sources)
    plot_band_anomaly(runs, depth, f'{out}/band_t_anomaly.pdf', sources)
    plot_split(runs, depth, channels, f'{out}/band_t_split.pdf', sources)
    t_name = f'T {depth[channels[T_CHANNEL]]:.0f} m'
    limits = {(s, 'sst'): plot_days(runs, s, mesh, 'sst', 'SST', 0, f'{out}/sst_days_{s}.pdf', sources)
              for s in SEASONS}
    limits.update({(s, 't26m'): plot_days(runs, s, mesh, 't_level', t_name, channels[T_CHANNEL],
                                          f'{out}/t26m_days_{s}.pdf', sources) for s in SEASONS})
    plot_sst_drift(drift, f'{out}/sst_drift.pdf', sources)
    plot_box_profiles(runs, 'box', 'sigma_0 (kg m-3)', depth, f'{out}/box_density_profiles.pdf', sources)
    plot_box_profiles(runs, 'box_t', 'T (degC)', depth, f'{out}/box_temperature_profiles.pdf', sources)
    e_sources = sources + [f'{f}/{k}.npy' for f in dict.fromkeys(folders.values()) for k in ('toce', 'soce')]
    plot_e_vs_drift(e_initial, drift_day, f'SST RMSE day {E_DRIFT_DAY} to day 0 (degC)',
                    f'{out}/e_vs_sst_drift.pdf', e_sources)
    plot_e_vs_drift(e_initial, t_drift_day, f'3D T RMSE day {E_DRIFT_DAY} to day 0 (degC)',
                    f'{out}/e_vs_t_drift.pdf', e_sources)
    zonal_limits = {f'zonal_t_{s}': plot_zonal_t(runs, s, mesh, depth, f'{out}/zonal_t_{s}.pdf', sources)
                    for s in SEASONS}
    for s in SEASONS:
        for name, configs in [('', CONFIGS[1:]), ('_nonone', [c for c in CONFIGS[1:] if c != 'none'])]:
            zonal_limits[f'zonal_t_vs_data{name}_{s}'] = plot_zonal_t(
                runs, s, mesh, depth, f'{out}/zonal_t_vs_data{name}_{s}.pdf', sources, configs=configs, vs_data=True)

    print(f'Model channels {channel_depths.round(0)} m -> 1/4 deg levels {channels} at {depth[channels].round(0)} m')
    rmse_cols = [(f, d) for f in ['sst', 'ssh', 'ssh_demeaned'] for d in (1, NDAYS)]
    names = {'sst': 'SST', 'ssh': 'SSH', 'ssh_demeaned': "SSH'"}
    head = [f'{names[f]} d{d}' for f, d in rmse_cols]
    head += [f'T{depth[k]:.0f}-dat d{d}' for k in table_levels for d in DAYS]
    print("RMSE to the data run (SSH' with the domain-mean difference removed), NH-band T minus the data run (degC)")
    print(f'{"run":14s} ' + ' '.join(f'{h:>12s}' for h in head))
    for s, c in runs:
        anomaly = runs[s, c]['band'] - runs[s, 'data']['band']
        cells = [rmses[s, c][f][d] for f, d in rmse_cols] + [anomaly[d, k] for k in table_levels for d in DAYS]
        print(f'{s + "_" + c:14s} ' + ' '.join(f'{v:+12.4f}' for v in cells))
    print('Summer - winter NH-band T (degC)')
    print(f'{"config":14s} ' + ' '.join(f'{f"split{depth[k]:.0f} d{d}":>12s}' for k in table_levels for d in DAYS))
    for c in CONFIGS:
        split = runs['summer', c]['band'] - runs['winter', c]['band']
        print(f'{c:14s} ' + ' '.join(f'{split[d, k]:+12.4f}' for k in table_levels for d in DAYS))
    for (s, prefix), (t, d) in ((k, (v['t'], v['d'])) for k, v in limits.items()):
        print(f'{prefix}_days_{s}.pdf: T {t[0]:.2f} to {t[1]:.2f} degC, D {d[0]:.3e} to {d[1]:.3e} '
              f'({"log" if d[2] else "linear"})')
    print('Median of D over the ocean')
    print(f'{"run":14s} ' + ' '.join(f'{f"day {d}":>10s}' for d in SST_DAYS[1:]))
    for s, c in runs:
        print(f'{s + "_" + c:14s} ' + ' '.join(f'{np.nanmedian(runs[s, c]["d"][d][mesh["ocean"]]):10.3e}'
                                               for d in SST_DAYS[1:]))
    print('SST RMSE to day 0 (degC)')
    print(f'{"run":14s} ' + ' '.join(f'{f"day {d}":>8s}' for d in DRIFT_DAYS))
    for s, c in runs:
        print(f'{s + "_" + c:14s} ' + ' '.join(f'{drift[s, c][d]:8.4f}' for d in DRIFT_DAYS))
    print(f'E of the initial state (%), SST and 3D T RMSE day {E_DRIFT_DAY} to day 0 (degC)')
    for s, c in runs:
        print(f'{s + "_" + c:14s} {e_initial[s, c]:8.3f} {drift_day[s, c]:8.4f} {t_drift_day[s, c]:8.4f}')
    box_levels = np.abs(depth[:, None] - np.array(BOX_TABLE_DEPTHS)[None]).argmin(0)
    box_columns = [(k, i) for k in box_levels for i in (0, BOX_DAYS.index(NDAYS))]
    for field, name in [('box', 'sigma_0 (kg m-3)'), ('box_t', 'T (degC)')]:
        print(f'Box {name}, lat {BOX_LAT[0]}-{BOX_LAT[1]}N, lon {box_lon[0]:.3f} to {box_lon[1]:.3f}, '
              f'{int((w_box[0] > 0).sum())} surface cells')
        print(f'{"run":14s} ' + ' '.join(f'{f"{depth[k]:.0f}m d{BOX_DAYS[i]}":>10s}' for k, i in box_columns))
        for s, c in runs:
            print(f'{s + "_" + c:14s} ' + ' '.join(f'{runs[s, c][field][i, k]:10.4f}' for k, i in box_columns))
    for name, dmax in zonal_limits.items():
        print(f'{name}.pdf: difference colour range -{dmax:.4f} to +{dmax:.4f} degC')
    print(f'Saved in {out}')
