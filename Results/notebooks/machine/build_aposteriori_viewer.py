"""
Build the data of the "A posteriori" tab of `density_viewer.html` : the 1/4° NEMO runs started from the median member of
each generation of the optimal sweep (`notebooks/a_posteriori/`), for every experiment of EXPERIMENTS (outputs in
`<generation>/aposteriori/<results>/`, named by `experiment_paths`).
    density_viewer_aposteriori.js : for every collected run of every experiment, time series of domain-mean quantities
        at the output frequency of the run, the Hovmöller of the horizontal mean of T and S, and the same quantities
        for the 1/4° spin-up restart (reference)
    density_viewer_aposteriori_<experiment>_<constraint>.js : snapshots of one run, its initial state then every output
        mean, of surface maps and zonal sections, loaded by the page when that run is shown
Frame 0 is the restart the run starts from (instantaneous), frame i >= 1 the i-th output mean (month or day).
Needs density_viewer_data.js (build_density_viewer.py) : the RMSE of the sigma_0 section is computed against its data
batch-mean section, as the RMSE of the Density tab.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[3]}/'
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

import base64
import gzip
import json
import os
from glob import glob

import cmocean.cm as cm
import numpy as np
import xarray as xr

from metrics import get_density_at_surface
from build_density_viewer import MLD_DSIGMA, MLD_ZREF, mixed_layer_depth
from settings import (COLLECT, EXP, EXPERIMENTS, INFERENCE, PREFIX, REFERENCES_REL, RESTART_1_4DEG, aposteriori_dirs,
                      experiment_paths, read_json)

HERE = Path(__file__).parent
OUT = HERE / 'density_viewer_aposteriori.js'
OUT_SNAPSHOTS = 'density_viewer_aposteriori_{}_{}.js'
DENSITY_DATA = HERE / 'density_viewer_data.js'
REFERENCE = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/restart25_arch'
MAP_STRIDE = 2  # maps are stored every MAP_STRIDE rows and columns
ACC_COLUMN = 1  # zonal transport through the meridional line of U points of this column (first interior column)
# Snapshots are stored as uint16 over these ranges, NaN as 0.
MAP_RANGES = {'sst': (-5., 45.), 'sss': (25., 45.), 'ssh': (-3., 3.), 'speed': (0., 4.), 'mld': (0., 4000.), 'ke_column': (-4., 5.)}
SECTION_RANGES = {'sigma': (15., 32.), 'toce': (-5., 45.), 'soce': (25., 45.)}
# Percentiles of the reference bounding the colour scale of each field, (1, 99) if absent. The MLD is deep (~3700 m) only
# where convection reaches the bottom : bounded at the 90th percentile so that the ~80 m of most of the basin show.
DISPLAY_PERCENTILES = {'mld': (1, 90)}
COLORMAPS = {'sst': cm.thermal, 'sss': cm.haline, 'ssh': cm.balance, 'speed': cm.speed, 'mld': cm.deep, 'ke_column': cm.amp, 'sigma': cm.deep_r,
             'toce': cm.thermal, 'soce': cm.haline, 'diff': cm.balance}


def gz64(array):
    """gzip then base64 of the bytes of an array, decoded in the page with DecompressionStream."""
    return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=6, mtime=0)).decode()


def quantize(a, lo, hi):
    """uint16 of `a` over [lo, hi] (values outside are clipped), 0 for NaN."""
    q = np.clip((a - lo) / (hi - lo), 0, 1) * 65534 + 1
    return np.where(np.isfinite(a), q, 0).astype('<u2')


def colorscale(cmap):
    """Plotly colorscale of a matplotlib colormap, 11 stops."""
    return [[float(v), f'rgb{tuple(int(255 * c) for c in cmap(v)[:3])}'] for v in np.linspace(0, 1, 11)]


def density_data():
    """Latitude, depth and data sections (n, z, y) of the Density tab, decoded from density_viewer_data.js."""
    text = DENSITY_DATA.read_text()
    d = json.loads(text[text.index('{'):text.rindex('}') + 1])
    lo, hi = d['sigma_range']
    q = np.frombuffer(base64.b64decode(d['data']['sections']), dtype='<u2').astype(float)
    sections = np.where(q == 0, np.nan, lo + (q - 1) / 65534 * (hi - lo))
    lat, depth = np.array(d['latitude']), np.array(d['depth'])
    return lat, depth, sections.reshape(-1, len(depth), len(lat))


def to_grid(section, lat, depth, lat_to, depth_to):
    """Section (z, y) on (depth, lat) linearly interpolated onto (depth_to, lat_to), NaN where a neighbour is NaN."""
    def interp(a, x, x_to):
        finite = np.isfinite(a)
        v = np.interp(x_to, x, np.where(finite, a, 0.))
        w = np.interp(x_to, x, finite.astype(float))
        return np.where(w > 1 - 1e-6, v, np.nan)
    by_lat = np.stack([interp(row, lat, lat_to) for row in section])
    return np.stack([interp(col, depth, depth_to) for col in by_lat.T]).T


class Mesh:
    """Grid of the 1/4° configuration (mesh_mask.nc of the reference run), (z, y, x) arrays."""

    def __init__(self, path):
        m = xr.open_dataset(path).isel(time_counter=0)
        self.tmask = m.tmask.values.astype(bool)
        self.umask, self.vmask = m.umask.values.astype(bool), m.vmask.values.astype(bool)
        self.area = (m.e1t * m.e2t).values
        self.e3t_0, self.e3u_0, self.e2u = m.e3t_0.values, m.e3u_0.values, m.e2u.values
        self.depth = m.gdept_1d.values
        self.lat = m.gphit.values[:, ACC_COLUMN + 1]
        self.lon = m.glamt.values[1]
        self.acc_rows = self.tmask[0].all(axis=1)  # reentrant channel


def describe(mesh, t, s, u, v, ue3u, e3t, ssh):
    """
    Quantities of one state of the 1/4° run, from T, S, U, V, thickness-weighted U and e3t (z, y, x) and SSH (y, x),
    NaN or any value on land :
        series : volume means of T, S and kinetic energy (u and v averaged onto T points) [cm2 s-2], mean SST, zonal
                 transport through the meridional line of column ACC_COLUMN [Sv]
        profiles : horizontal volume means of T and S (z,)
        sections : volume-weighted zonal means of sigma_0, T and S (z, y)
        maps : SST, SSS, SSH, surface speed and mixed layer depth of the T, S state (criterion of the A-priori tab,
               build_density_viewer.mixed_layer_depth) and kinetic energy averaged over the water column, weighted by
               the cell thickness, as log10 of cm2 s-2 (median ~10, maximum ~6500 in the spin-up restart) (y, x)
    """
    tm = mesh.tmask
    vol = np.where(tm, mesh.area * e3t, 0.)
    t, s = np.where(tm, t, np.nan), np.where(tm, s, np.nan)
    u, v = np.where(mesh.umask, u, 0.), np.where(mesh.vmask, v, 0.)
    u_t = 0.5 * (u + np.roll(u, 1, axis=2))
    v_t = 0.5 * (v + np.concatenate([np.zeros_like(v[:, :1]), v[:, :-1]], axis=1))
    ke = np.where(tm, 0.5 * (u_t ** 2 + v_t ** 2), np.nan)
    sigma = np.where(tm, get_density_at_surface(t, s, tm) - 1000, np.nan)
    vmean = lambda a, axis=None: np.nansum(a * vol, axis=axis) / vol.sum(axis=axis)
    transport = np.nansum(np.where(mesh.umask, ue3u, 0.)[:, :, ACC_COLUMN] * mesh.e2u[:, ACC_COLUMN]) / 1e6
    with np.errstate(invalid='ignore'):
        sections = {k: vmean(a, axis=2) for k, a in [('sigma', sigma), ('toce', t), ('soce', s)]}
        profiles = {k: vmean(a, axis=(1, 2)) for k, a in [('toce', t), ('soce', s)]}
    surface = tm[0]
    maps = {'sst': t[0], 'sss': s[0], 'ssh': np.where(surface, ssh, np.nan),
            'speed': np.where(surface, np.sqrt(u_t[0] ** 2 + v_t[0] ** 2), np.nan),
            'mld': mixed_layer_depth(sigma[None], tm, mesh.depth)[0],
            'ke_column': np.where(surface, np.log10(np.maximum(
                1e4 * np.nansum(ke * vol, axis=0) / np.where(surface, vol.sum(axis=0), 1.), 1e-4)), np.nan)}
    series = {'toce': vmean(t), 'soce': vmean(s), 'ke': 1e4 * vmean(ke),
              'sst': np.nansum(t[0] * mesh.area) / mesh.area[surface].sum(), 'acc': transport}
    f32 = lambda d: {k: a.astype('f4') for k, a in d.items()}
    return {'series': {k: float(x) for k, x in series.items()}, 'profiles': profiles, 'sections': f32(sections), 'maps': f32(maps)}


def restart_state(mesh, path):
    """`describe` of a NEMO restart (now fields), e3t taken as e3t_0."""
    r = xr.open_dataset(path).isel(time_counter=0)
    un = r.un.values
    return describe(mesh, r.tn.values, r.sn.values, un, r.vn.values, un * mesh.e3u_0, mesh.e3t_0, r.sshn.values)


def output_freq(launch):
    """XIOS frequency of the 3D outputs of a launch record, as in their file names ('1m', '1d')."""
    return next(g['output_freq'] for g in launch['outputs'] if g['enabled'] == '.TRUE.' and '_grid_T_3D' in g['files']).replace('mo', 'm')


def output_days(results, freq):
    """Time of the centre of each output mean of a run, in days from its start (360-day calendar)."""
    t = xr.open_dataset(f'{results}/{EXP}_{freq}_grid_T_3D.nc', decode_times=False)
    return ((t.time_centered.values - t.time_counter_bounds.values[0, 0]) / 86400).tolist()


def output_states(mesh, results, freq):
    """`describe` of each output mean (frequency `freq`) of a run, in time order, with the area mean of the mixed layer depth."""
    t3 = xr.open_dataset(f'{results}/{EXP}_{freq}_grid_T_3D.nc', decode_times=False)
    u3 = xr.open_dataset(f'{results}/{EXP}_{freq}_grid_U_3D.nc', decode_times=False)
    v3 = xr.open_dataset(f'{results}/{EXP}_{freq}_grid_V_3D.nc', decode_times=False)
    t2 = xr.open_dataset(f'{results}/{EXP}_{freq}_grid_T_2D.nc', decode_times=False)
    surface = mesh.tmask[0]
    for i in range(t3.sizes['time_counter']):
        get = lambda ds, name: ds[name].isel(time_counter=i).values
        state = describe(mesh, get(t3, 'toce'), get(t3, 'soce'), get(u3, 'uoce'), get(v3, 'voce'), get(u3, 'uoce_e3u'),
                         get(t3, 'e3t'), get(t2, 'ssh'))
        state['series']['mld'] = float(np.nansum(np.where(surface, get(t2, 'mldr10_1'), 0.) * mesh.area) / mesh.area[surface].sum())
        print(f'  {freq} {i + 1} / {t3.sizes["time_counter"]}', end='\r')
        yield state
    print()


def section_rmse(section, mesh, lat, depth, data_mean):
    """Unweighted RMSE of a 1/4° sigma_0 section, interpolated onto the Density tab grid, to its data batch-mean section."""
    on_grid = to_grid(section, mesh.lat, mesh.depth, lat, depth)
    ok = np.isfinite(on_grid) & np.isfinite(data_mean)
    return float(np.sqrt(np.mean((on_grid[ok] - data_mean[ok]) ** 2)))


def snapshots(states):
    """Quantized and compressed maps (frames, y / MAP_STRIDE, x / MAP_STRIDE) and sections (frames, z, y) of `states`."""
    maps = {k: gz64(quantize(np.stack([st['maps'][k][::MAP_STRIDE, ::MAP_STRIDE] for st in states]), *r))
            for k, r in MAP_RANGES.items()}
    sections = {k: gz64(quantize(np.stack([st['sections'][k] for st in states]), *r)) for k, r in SECTION_RANGES.items()}
    return {'frames': len(states), 'maps': maps, 'sections': sections}


def display_ranges(reference, runs):
    """
    Colour range of each field : DISPLAY_PERCENTILES of the reference, and the half range of the difference to the
    initial state : upper DISPLAY_PERCENTILES of |last frame - first frame| over the runs.
    """
    out = {}
    for group, fields in [('maps', MAP_RANGES), ('sections', SECTION_RANGES)]:
        for k in fields:
            ref = reference[group][k]
            diffs = np.concatenate([np.abs(r[-1][group][k] - r[0][group][k]).ravel() for r in runs]) if runs else np.zeros(1)
            out[k] = {'range': np.nanpercentile(ref, DISPLAY_PERCENTILES.get(k, (1, 99))).tolist(), 'diff': float(np.nanpercentile(diffs, DISPLAY_PERCENTILES.get(k, (1, 99))[1]))}
    return out


if __name__ == '__main__':
    mesh = Mesh(f'{REFERENCE}/mesh_mask.nc')
    lat, depth, data_sections = density_data()
    with np.errstate(invalid='ignore'):
        data_mean = np.nanmean(data_sections, axis=0)
    data_rmse = [float(np.sqrt(np.nanmean((d - data_mean) ** 2))) for d in data_sections]

    reference_path = sorted(glob(f'{REFERENCE}/{EXP}_*_restart.nc'))[-1]
    reference = restart_state(mesh, reference_path)
    reference['series']['rmse'] = section_rmse(reference['sections']['sigma'], mesh, lat, depth, data_mean)
    print(f'reference : {reference_path}')

    out = {'experiments': {}}
    initial = {}  # initial state of each constraint, shared by its experiments
    for experiment in EXPERIMENTS:
        paths = experiment_paths(experiment)
        runs, first_last = {}, []
        for constraint, folder in aposteriori_dirs().items():
            results = f'{folder}/{paths["results"]}'
            if not os.path.exists(f'{results}/{COLLECT}'):
                continue
            print(f'{experiment} {constraint} : {folder}')
            freq = output_freq(read_json(f'{folder}/{paths["launch"]}'))
            if constraint not in initial:
                initial[constraint] = restart_state(mesh, f'{folder}/{RESTART_1_4DEG}')
                initial[constraint]['series']['mld'] = None
            states = [initial[constraint]] + list(output_states(mesh, results, freq))
            for st in states:
                st['series']['rmse'] = section_rmse(st['sections']['sigma'], mesh, lat, depth, data_mean)
            selection, collect = read_json(f'{folder}/selection.json'), read_json(f'{results}/{COLLECT}')
            runs[constraint] = {
                'member': selection['member'], 'projection': selection['generation_args'].get('projection'),
                'generation': selection['generation_rel'], 'elapsed': collect.get('elapsed'), 'freq': freq,
                'days': [0.] + output_days(results, freq),
                'series': {k: [st['series'][k] for st in states] for k in states[-1]['series']},
                'profiles': {k: gz64(np.stack([st['profiles'][k] for st in states]).astype('<f4')) for k in ['toce', 'soce']},
            }
            snap = HERE / OUT_SNAPSHOTS.format(experiment, constraint)
            key = json.dumps(f'{experiment}/{constraint}')
            snap.write_text(f'(window.APOSTERIORI_SNAPSHOTS ??= {{}})[{key}] = {json.dumps(snapshots(states))};\n')
            print(f'Saved {snap} ({snap.stat().st_size / 1e6:.1f} MB)')
            first_last.append([states[0], states[-1]])
        if runs:
            out['experiments'][experiment] = {'runs': runs, 'display': None, 'first_last': first_last}

    out['reference'] = {'path': os.path.relpath(reference_path, os.environ['OCEANDATA']), 'series': reference['series'],
                        'profiles': {k: reference['profiles'][k].tolist() for k in ['toce', 'soce']},
                        **snapshots([reference])}
    out['data_rmse'] = data_rmse
    out['depth'] = mesh.depth.tolist()
    out['lat'] = mesh.lat.tolist()
    out['lon'] = mesh.lon.tolist()
    out['map_stride'] = MAP_STRIDE
    out['map_ranges'], out['section_ranges'] = MAP_RANGES, SECTION_RANGES
    for e in out['experiments'].values():
        e['display'] = display_ranges(reference, e.pop('first_last'))
    out['colorscales'] = {k: colorscale(c) for k, c in COLORMAPS.items()}
    out['acc_lat'] = [float(mesh.lat[mesh.acc_rows].min()), float(mesh.lat[mesh.acc_rows].max())]
    out['inference'] = os.path.relpath(f'{INFERENCE}/{PREFIX}', os.environ['OCEANDATA'])
    OUT.write_text(f'window.APOSTERIORI_DATA = {json.dumps(out)};\n')
    print(f'Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)')
