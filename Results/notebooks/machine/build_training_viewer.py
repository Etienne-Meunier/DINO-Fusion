"""
Build `density_viewer_training.js`, the data of the "Data" tab of `density_viewer.html` : the time evolution of the training
archive, one value per state in archive order (36 states per year), on the 1° grid of the states (1/4° fields coarse-grained).
    series : per-state metrics of METRICS (from T, S and SSH only, the archive has no velocity), each with its group,
             label and unit
    equilibration : per metric, the trend of the annual means over the first and last WINDOW years and over the whole
                    run, in units of the interannual noise (see `equilibration`)
    hovmoller : annual means of the horizontal mean of T, S and sigma_0 of each level, minus their mean over the run
    reference : volume mean T and S, area mean SST and ACC transport of the 1/4° spin-up restart, read from
                density_viewer_aposteriori.js (build_aposteriori_viewer.py, 1/4° grid, transport from the velocities)
    batches : archive positions of the data batches drawn by the analysis scripts from the shuffle seeded with DATA_SEED
The pass over the training archive is cached (viewer_cache) until the archive or the code change.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[3]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')

import json
import random

import numpy as np
import webdataset as wds
import xarray as xr

from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED
from metrics import get_density_at_surface
from build_density_viewer import MLD_DSIGMA, MLD_ZREF, acc_mask, mixed_layer_depth
from viewer_cache import cached, code_hash, file_id, source_hash

HERE = Path(__file__).parent
OUT = HERE / 'density_viewer_training.js'
APOSTERIORI_DATA = HERE / 'density_viewer_aposteriori.js'
MESH = f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc'
STATES_PER_YEAR = 36
CHUNK = 60  # states processed together
WINDOW = 10  # years of the early and late trends
DRIFT_THRESHOLD = 2.  # |drift| / trend_noise below which a window shows no detectable drift
G, RHO0, OMEGA = 9.81, 1026., 7.2921e-5
DEPTH_BANDS = [(0, 200), (200, 1000), (1000, 2000), (2000, 5000)]  # [m], on the depth of the T points
DEEP_CONVECTION = 1000.  # [m], mixed layer depth of the deep convection area
ISOTHERM = 20.  # [°C], thermocline isotherm, mid-thermocline of the tropical mean profile (26 °C at the surface, 12 °C at 450 m)
# (label, first, last) : states first to last - 1 of the DATA_SEED shuffle
BATCHES = [('paper/common.py (200 states)', 0, 200), ('Density and A-priori tabs (8 states)', 0, 8),
           ('a posteriori data run (16 states)', 200, 216)]
# key : (group, label, unit, key of the restart in density_viewer_aposteriori.js or None)
METRICS = {
    'toce': ('Heat and salt', 'Volume mean T', '°C', 'toce'),
    'soce': ('Heat and salt', 'Volume mean S', 'g kg⁻¹', 'soce'),
    'sigma': ('Heat and salt', 'Volume mean σ₀', 'kg m⁻³', None),
    **{f'toce_{lo}_{hi}': ('T by depth', f'Volume mean T, {lo}–{hi} m', '°C', None) for lo, hi in DEPTH_BANDS},
    **{f'soce_{lo}_{hi}': ('S by depth', f'Volume mean S, {lo}–{hi} m', 'g kg⁻¹', None) for lo, hi in DEPTH_BANDS},
    'toce_north': ('T by region', 'Volume mean T, north of the equator', '°C', None),
    'toce_south': ('T by region', 'Volume mean T, south of the equator, out of the channel', '°C', None),
    'toce_acc': ('T by region', 'Volume mean T, ACC channel', '°C', None),
    'sst': ('Surface', 'Area mean SST', '°C', 'sst'),
    'sss': ('Surface', 'Area mean SSS', 'g kg⁻¹', None),
    'ssh': ('Surface', 'Area mean SSH', 'm', None),
    'sst_contrast': ('Surface', 'SST, tropics (|lat| < 15°) minus north of 50°N', '°C', None),
    'mld': ('Mixing and stratification', 'Area mean mixed layer depth', 'm', None),
    'convection': ('Mixing and stratification', f'Ocean area with mixed layer deeper than {DEEP_CONVECTION:.0f} m', '%', None),
    'strat': ('Mixing and stratification', 'σ₀ at 1000 m minus σ₀ at the surface, area mean', 'kg m⁻³', None),
    'thermocline': ('Mixing and stratification', f'Depth of the {ISOTHERM:.0f} °C isotherm, tropical mean profile (|lat| < 15°)', 'm', None),
    'acc_ssh': ('ACC (geostrophy)', 'SSH drop across the channel (north − south edge)', 'm', None),
    'acc_bc': ('ACC (geostrophy)', 'Thermal-wind transport relative to the bottom', 'Sv', None),
    'acc_total': ('ACC (geostrophy)', 'Geostrophic transport referenced to the SSH', 'Sv', 'acc'),
}


class Grid:
    """1° grid of the training states : masks, cell volumes, depth and latitude, from the mesh mask."""
    def __init__(self, path):
        m = xr.open_dataset(path).isel(time_counter=0)
        self.tmask = m.tmask.values.astype(bool)
        self.area = m.e1t.values * m.e2t.values
        self.e3t = m.e3t_0.values
        self.volume = self.area * self.e3t * self.tmask
        self.depth = m.gdept_1d.values
        self.lat = m.gphit.values
        self.acc = acc_mask(self.tmask)
        self.channel_rows = np.flatnonzero(self.acc[0].any(axis=1))


def vmean(f, w):
    """Weighted mean over the last 3 axes of (b, z, y, x) `f`, NaN ignored, weights `w` (z, y, x), (b,)."""
    return np.nansum(f * w, axis=(-3, -2, -1)) / w.sum()


def amean(f, w):
    """Weighted mean over the last 2 axes of (b, y, x) `f`, NaN ignored, weights `w` (y, x), (b,)."""
    return np.nansum(f * w, axis=(-2, -1)) / w.sum()


def describe(t, s, ssh, grid):
    """Metrics of METRICS of a chunk of states, t, s (b, z, y, x), ssh (b, y, x), NaN on land, {key: (b,)}."""
    sigma = np.where(grid.tmask, get_density_at_surface(t, s, grid.tmask) - 1000, np.nan)
    v, sea = grid.volume, grid.tmask[0]
    surf_area = grid.area * sea
    out = {'toce': vmean(t, v), 'soce': vmean(s, v), 'sigma': vmean(sigma, v)}
    for lo, hi in DEPTH_BANDS:
        band = ((grid.depth >= lo) & (grid.depth < hi))[:, None, None]
        out[f'toce_{lo}_{hi}'], out[f'soce_{lo}_{hi}'] = vmean(t, v * band), vmean(s, v * band)
    north = (grid.lat > 0)[None]
    out['toce_north'] = vmean(t, v * north)
    out['toce_south'] = vmean(t, v * ~north * ~grid.acc)
    out['toce_acc'] = vmean(t, v * grid.acc)

    out['sst'], out['sss'], out['ssh'] = amean(t[:, 0], surf_area), amean(s[:, 0], surf_area), amean(ssh, surf_area)
    tropics, polar = np.abs(grid.lat) < 15, grid.lat > 50
    out['sst_contrast'] = amean(t[:, 0], surf_area * tropics) - amean(t[:, 0], surf_area * polar)

    mld = mixed_layer_depth(np.nan_to_num(sigma), grid.tmask, grid.depth)
    out['mld'] = amean(mld, surf_area)
    out['convection'] = 100 * amean((mld > DEEP_CONVECTION).astype(float), surf_area)
    k1000 = int(np.argmin(np.abs(grid.depth - 1000)))
    out['strat'] = amean(sigma[:, k1000] - sigma[:, 0], grid.area * grid.tmask[k1000])
    # Tropical mean profile of T, depth of ISOTHERM linearly interpolated between the levels (T decreases with depth there)
    w = grid.area * tropics * grid.tmask
    profile = np.nansum(t * w, axis=(-2, -1)) / np.where(w.sum(axis=(-2, -1)) > 0, w.sum(axis=(-2, -1)), np.nan)
    out['thermocline'] = isotherm_depth(profile, grid.depth)

    south, north_row = grid.channel_rows[0], grid.channel_rows[-1]
    levels = grid.acc[:, grid.channel_rows].all(axis=(1, 2))
    edge = lambda f, row: np.nanmean(f[..., row, :], axis=-1)
    out.update(acc_geostrophy(edge(sigma, north_row)[:, levels], edge(sigma, south)[:, levels], edge(ssh, north_row), edge(ssh, south),
                              grid.depth[levels], grid.e3t[levels, south, 0], grid.lat[grid.channel_rows].mean()))

    level_mean = lambda f: np.nansum(f * v, axis=(-2, -1)) / np.where(v.sum(axis=(-2, -1)) > 0, v.sum(axis=(-2, -1)), np.nan)
    profiles = {'toce': level_mean(t), 'soce': level_mean(s), 'sigma': level_mean(sigma)}
    return out, profiles


def isotherm_depth(profile, depth, isotherm=ISOTHERM):
    """Depth [m] of `isotherm` in each T profile (b, z), linearly interpolated between the levels, T decreasing with depth."""
    return np.array([np.interp(-isotherm, -p[np.isfinite(p)], depth[np.isfinite(p)]) for p in profile])


def acc_geostrophy(sigma_north, sigma_south, ssh_north, ssh_south, depth, e3t, lat):
    """
    Geostrophic zonal transport through a flat-bottomed reentrant channel, f at latitude `lat` [°], from the zonal means of
    sigma_0 (b, z) and SSH (b,) on its north and south edge rows, on the levels `depth`, `e3t` (z,) of the channel :
        acc_ssh : SSH north minus south [m]
        acc_bc : transport of the thermal wind relative to the bottom [Sv]
        acc_total : transport referenced to the surface geostrophic velocity of the SSH [Sv]
    """
    f0 = 2 * OMEGA * np.sin(np.deg2rad(lat))
    H = e3t.sum()
    drho, dssh = sigma_north - sigma_south, ssh_north - ssh_south
    # u(z) = u(-H) + g / (rho0 f) ∫_{-H}^{z} ∂ρ/∂y, integrated over the depth and across the channel
    bc = 1e-6 * G / (RHO0 * f0) * (drho * depth * e3t).sum(axis=-1)
    # u(z) = -g / f ∂η/∂y - g / (rho0 f) ∫_{z}^{0} ∂ρ/∂y
    total = 1e-6 * (-G / f0 * H * dssh - G / (RHO0 * f0) * (drho * (H - depth) * e3t).sum(axis=-1))
    return {'acc_ssh': dssh, 'acc_bc': bc, 'acc_total': total}


def equilibration(x, window=WINDOW):
    """
    Drift of the annual means of `x` (n_states,) : linear trend times the duration over the first and last `window` years
    (early, late) and over the whole run (full), divided by `trend_noise`, the std of the same `window`-year drift over
    every sliding window of the residual of the annual means about a quadratic fit of the run. The residual keeps the
    low-frequency (autocorrelated) variability, so a window of variability alone scores about ±1.
        noise : std of that residual, seasonal : std of the states about their annual mean
        spread : std of the states about the quadratic fit (seasonal and interannual variability of a single state)
        status : 'equilibrated' (no detectable drift in either window), 'equilibrates' (only in the first), 'late' (only
                 in the last) or 'drifting', with DRIFT_THRESHOLD
    """
    annual = x.reshape(-1, STATES_PER_YEAR).mean(axis=1)
    years = np.arange(len(annual))
    residual = annual - np.polyval(np.polyfit(years, annual, 2), years)
    drift = lambda a, sl: np.polyfit(years[sl], a[sl], 1)[0] * (len(years[sl]) - 1)
    trend_noise = np.std([drift(residual, slice(i, i + window)) for i in range(len(years) - window + 1)])
    early, late = drift(annual, slice(0, window)) / trend_noise, drift(annual, slice(-window, None)) / trend_noise
    calm_early, calm_late = abs(early) < DRIFT_THRESHOLD, abs(late) < DRIFT_THRESHOLD
    status = 'equilibrated' if calm_early and calm_late else 'equilibrates' if calm_late else 'late' if calm_early else 'drifting'
    return {'status': status, 'noise': float(np.std(residual, ddof=3)), 'trend_noise': float(trend_noise),
            'seasonal': float(np.std(x - np.repeat(annual, STATES_PER_YEAR))),
            'spread': float(np.std(x - np.polyval(np.polyfit(years, annual, 2), np.arange(len(x)) / STATES_PER_YEAR - 0.5))),
            'early': float(early), 'late': float(late),
            'full': float(drift(annual, slice(None)) / trend_noise), 'total_drift': float(drift(annual, slice(None)))}


def archive(data_file):
    """Training states of the tar, without the infos member."""
    return wds.WebDataset(data_file).select(lambda x: 'infos' not in x['__key__'])


def shuffled_keys(data_file, n):
    """Keys of the first `n` states of the shuffle of `DataLoader.get_dataloader(shuffle=True, seed=DATA_SEED)`."""
    shuffled = archive(data_file).shuffle(1000, rng=random.Random(DATA_SEED))
    return [s['__key__'] for _, s in zip(range(n), shuffled)]


def reference_series():
    """Metrics of the 1/4° spin-up restart and its path, from density_viewer_aposteriori.js."""
    text = APOSTERIORI_DATA.read_text()
    reference = json.loads(text[text.index('{'):text.rindex('}') + 1])['reference']
    return {k: reference['series'][r] for k, (*_, r) in METRICS.items() if r is not None}, reference['path']


if __name__ == '__main__':
    config = TrainingConfig()
    grid = Grid(MESH)

    def read_archive():
        """Keys, per-state metrics and per-chunk level profiles of every training state, in archive order."""
        keys, series, profiles = [], {k: [] for k in METRICS}, {'toce': [], 'soce': [], 'sigma': []}
        chunk = []

        def flush():
            land = ~grid.tmask
            t = np.where(land, np.nan, np.stack([c['toce.npy'] for c in chunk]).astype(float))
            s = np.where(land, np.nan, np.stack([c['soce.npy'] for c in chunk]).astype(float))
            ssh = np.where(land[0], np.nan, np.stack([c['ssh.npy'] for c in chunk]).astype(float).reshape(len(chunk), *land.shape[1:]))
            out, prof = describe(t, s, ssh, grid)
            for k in series: series[k].extend(out[k].tolist())
            for k in profiles: profiles[k].append(prof[k])
            chunk.clear()
            print(f'{len(keys)} states', end='\r')

        for sample in archive(config.data_file).decode():
            keys.append(sample['__key__'])
            chunk.append(sample)
            if len(chunk) == CHUNK: flush()
        if chunk: flush()
        return keys, series, profiles

    code = [code_hash(__file__), source_hash(get_density_at_surface, mixed_layer_depth, acc_mask, MLD_DSIGMA, MLD_ZREF), file_id(MESH)]
    keys, series, profiles = cached('training', [*code, file_id(config.data_file)], read_archive)

    position = {key: i for i, key in enumerate(keys)}
    order = shuffled_keys(config.data_file, max(last for _, _, last in BATCHES))
    reference, reference_path = reference_series()
    hovmoller = {}
    for k, p in profiles.items():
        annual = np.concatenate(p).reshape(-1, STATES_PER_YEAR, len(grid.depth)).mean(axis=1)
        hovmoller[k] = np.round(annual - np.nanmean(annual, axis=0), 6).tolist()

    out = {'years': (np.arange(len(keys)) / STATES_PER_YEAR).tolist(),
           'metrics': {k: {'group': g, 'label': l, 'unit': u} for k, (g, l, u, _) in METRICS.items()},
           'series': {k: np.round(v, 7).tolist() for k, v in series.items()},
           'equilibration': {k: equilibration(np.array(v)) for k, v in series.items()},
           'window': WINDOW, 'depth': grid.depth.tolist(), 'hovmoller': hovmoller,
           'mld_def': f'σ₀ > σ₀({MLD_ZREF:.0f} m) + {MLD_DSIGMA}',
           'reference': reference, 'reference_path': reference_path,
           'batches': [{'label': label, 'positions': sorted(position[k] for k in order[first:last])} for label, first, last in BATCHES]}
    OUT.write_text(f'window.TRAINING_DATA = {json.dumps(out)};\n')
    for k, e in out['equilibration'].items():
        print(f'{k:16s} early {e["early"]:+7.2f} late {e["late"]:+7.2f} full {e["full"]:+8.2f}  drift {e["total_drift"]:+.4g}  noise {e["noise"]:.3g}  mean {np.mean(series[k]):.4g}')
    print(f'-> {OUT}')
