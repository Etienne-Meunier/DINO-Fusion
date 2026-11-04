#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared state of the paper-clean scripts: the DINO grids as dinostics experiments, the data batch and the generations
of one sweep of one run, and the few helpers dinostics has no routine for.

Imported with ``from common import *`` so an interactive session loads the data once: `importlib.reload(common)`
redefines the functions but keeps the loaded data (see `session.py`).

Grids. GRID is the dinostics `NemoExperiment` of the 1° reference run (domain_cfg, mesh_mask and namelist_cfg of
`<REFERENCES>/1deg_restart`): the generated states live on its T grid, their density is the TEOS-10
polynomial of `metrics.get_density_at_surface`, the EOS of the training constraint (`pipelines/State.py`). `aposteriori_experiment` builds the `NemoExperiment` of the 1/4° reference run
(`<REFERENCES>/restart25_arch`) with the outputs of an a posteriori run in `exp.data`. CHANNEL holds the re-entrant
channel latitudes of the analytical DINO configuration of dinostics (`DinoConfiguration`).

Data. `batch` holds the 16 normalized training states of the data run of the sweep DATA_PREFIX
(`<INFERENCE>/optimal/data/<latest timestamp>/states.npy`, described by its args.json, DATA_ARGS), the states the a
posteriori data run starts from. REFERENCE_DENSITY is the batch-mean zonal sigma_0 section (z_c - 1, y_c) of the
200-state data batch, written once by `build_reference_density.py`; it is the reference of the median member of fig1,
as of the a posteriori selection.

Runs. The model and sweep are RUN / PREFIX, overridable with PAPER_RUN / PAPER_PREFIX. DENSITY_RUN is the
increasing-density run every figure shows (PAPER_DENSITY_RUN, see `compare_eta.py`). Figures are saved in FIGURES, the
`figures` folder next to this file (PAPER_FIGURES), and CONFIGS holds the runs each figure shows for the sweep.
"""
from __init__ import PRP; import sys
sys.path += [f'{PRP}/Diffusion_Model/', PRP, f'{PRP}/Results/a_posteriori/dinostics/',
             f'{PRP}/Results/notebooks/a_posteriori/', f'{PRP}/Results/analysis_scripts/']

import inspect
import json
import os
import subprocess
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr
import xnemogcm as xn
import cmocean  # registers the cmo.* colormaps used by name
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)  # invalid escape sequences in the docstrings of diagnostics.py
    from experiment import NemoExperiment
from dino_configuration import DinoConfiguration
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from metrics import get_density_at_surface
from settings import COLLECT, DATA_RUN, DATA_STATES, EXP, REFERENCES_REL, aposteriori_dirs, experiment_paths

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get('PAPER_RUN', 'tav0h83b')
PREFIX = os.environ.get('PAPER_PREFIX', 'optimal')
INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/{RUN}/inference'
FIGURES = os.environ.get('PAPER_FIGURES', f'{HERE}/figures')
DENSITY_RUN = os.environ.get('PAPER_DENSITY_RUN', 'negative_grad_density_huber_border0_30')
DATA_PREFIX = 'optimal'
REFERENCE_DENSITY_FILE = f'{INFERENCE}/{DATA_PREFIX}/reference_density/reference_density.npy'
REFERENCES = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/'
GRID_1DEG, GRID_1_4DEG = '1deg_restart', 'restart25_arch'
_DINO = DinoConfiguration()
CHANNEL = (_DINO.lat_channel_min, _DINO.lat_channel_max)
DEPTH_POWER = 0.625
SIGMA_LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
FIELDS = ['toce', 'soce']
N_CLUSTERS = 5
N_LEVEL_CLUSTERS = 5
TS_T_EDGES = np.arange(-3., 32.01, 0.5)  # values outside the edges fall in the edge bins
TS_S_EDGES = np.arange(32., 38.001, 0.05)
NAMES = {'data': 'Data',
         'none': 'No constraint',
         'none_border0': 'Boundary projection',
         'zero_mean_border0': 'T,S mean constraint',
         'negative_grad_density_huber_border0': 'Increasing density constraint',
         'mean_density_border0': 'Mean density constraint',
         'negative_grad_density_huber_border0_30': 'Increasing density constraint, eta = 30',
         'negative_grad_density_huber_border0_50': 'Increasing density constraint, eta = 50',
         'negative_grad_density_huber_border0_80': 'Increasing density constraint, eta = 80',
         'negative_grad_density_huber_surface_border0': 'Increasing density + surface T,S',
         'negative_grad_density_huber_surface_border0_30': 'Increasing density (eta = 30) + surface T,S',
         'C': 'Centred noise',
         'C_B0': 'Centred, boundary proj.',
         'C_NL': 'Centred, noisy land',
         'C_NL_b8': 'Centred, noisy land, b8',
         'C_NL_zm0.1': 'C+NL, T,S mean',
         'C_NL_C7030_sTS': 'C+NL, C7 30 + surface T,S',
         'C_NL_C7010': 'C+NL, C7 eta = 10',
         'C_NL_C7030': 'C+NL, C7 eta = 30',
         'C_NL_C7050': 'C+NL, C7 eta = 50',
         'C_NL_C7100': 'C+NL, C7 eta = 100',
         'NL': 'Noisy land',
         'NL_C7010': 'NL, C7 eta = 10',
         'NL_C7030': 'NL, C7 eta = 30',
         'NL_C7050': 'NL, C7 eta = 50',
         'NL_C7100': 'NL, C7 eta = 100'}
warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})


def provenance(sources=(), depth=1):
    """Description and inputs of the figure script calling (`depth` frames up): {'Subject': its docstring, 'Creator':
    its path at the current git commit (``-dirty`` if uncommitted), 'Keywords': RUN, PREFIX, the training data, the
    reference grids, the generation folders of the sweep and `sources`, one per line}."""
    frame = inspect.stack()[depth].frame
    script = os.path.abspath(frame.f_globals.get('__file__', '<interactive>'))
    git = lambda *a: subprocess.run(['git', '-C', PRP, *a], capture_output=True, text=True).stdout.strip()
    commit = git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain', '--', script) else '')
    inputs = [f'run {RUN}, sweep {PREFIX}', DATA_DIR, REFERENCE_DENSITY_FILE, f'{REFERENCES}{GRID_1DEG}',
              *generation_dirs().values(), *map(str, sources)]
    return {'Subject': (frame.f_globals.get('__doc__') or '').strip(), 'Creator': f'{script} @ {commit}',
            'Keywords': '\n'.join(inputs)}


def save(fig, path, sources=(), **kwargs):
    """`fig.savefig(path, **kwargs)` with the `provenance` of the calling script as PDF metadata, Title the file name."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, metadata={'Title': os.path.basename(path)} | provenance(sources, depth=2), **kwargs)


def save_tex(text, path, sources=()):
    """Write the LaTeX `text` to `path`, preceded by the `provenance` of the calling script as % comments."""
    header = '\n'.join(f'{k}: {v}' for k, v in provenance(sources, depth=2).items())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(''.join(f'% {line}\n' for line in header.splitlines()) + text)


def by_timestamp(files):
    """`files` (``.../<timestamp>/<name>``) sorted by the date of their ``%d%m%y-%H%M%S`` timestamp folder."""
    return sorted(files, key=lambda f: datetime.strptime(os.path.basename(os.path.dirname(f)), '%d%m%y-%H%M%S'))


def generation_dirs(prefix=PREFIX, inference=INFERENCE):
    """Folder ``<inference>/<prefix>/<constraint>/<timestamp>/`` holding the generated.npy of every run, keyed by
    constraint. A run with several timestamps prints an error and keeps the latest."""
    dirs = {}
    for run_dir in sorted(glob(f'{inference}/{prefix}/*/')):
        constraint = os.path.basename(os.path.normpath(run_dir))
        versions = by_timestamp(glob(f'{run_dir}/*/generated.npy'))
        if not versions:
            continue
        if len(versions) > 1:
            print(f'ERROR : {prefix}/{constraint} has {len(versions)} versions, keeping the latest')
        dirs[constraint] = os.path.dirname(versions[-1])
    return dirs


def load_generations(prefix=PREFIX, inference=INFERENCE):
    """Generation (normalized tensor) and args.json of every run of `generation_dirs`, as two dicts keyed by constraint."""
    gens, args = {}, {}
    for constraint, path in generation_dirs(prefix, inference).items():
        gens[constraint] = torch.from_numpy(np.load(f'{path}/generated.npy'))
        with open(f'{path}/args.json') as f:
            args[constraint] = json.load(f)
    return gens, args


def to_dataset(x):
    """Unnormalized toce, soce (member, z_c, y_c, x_c) and ssh (member, y_c, x_c) of a normalized batch x, as a
    Dataset on the T grid of GRID."""
    with torch.no_grad():
        fields = {k: v.cpu().numpy() for k, v in observer.unnormalized(x).items()}
    dims = {3: ('member', 'y_c', 'x_c'), 4: ('member', 'z_c', 'y_c', 'x_c')}
    return xr.Dataset({k: (dims[v.ndim], v) for k, v in fields.items()},
                      coords={c: GRID.mask[c] for c in ('z_c', 'y_c', 'x_c', 'gphit', 'glamt', 'gdept_0')})


def cell_volume(exp, e3t=None):
    """Volume e1t e2t e3t of the T cells of the dinostics experiment `exp`, 0 on land; e3t defaults to e3t_0."""
    mask = exp.mask
    e3t = mask.e3t_0 if e3t is None else e3t
    return (mask.e1t * mask.e2t * e3t).where(mask.tmask == 1.0, 0.).transpose(*e3t.dims)


def zonal_mean(var, volume):
    """Volume-weighted mean along x_c of `var` over its ocean cells, NaN on rows without any."""
    return (var * volume).sum('x_c') / volume.where(var.notnull()).sum('x_c')


def teos_sigma0(t, s):
    """TEOS-10 sigma_0 = rho - 1000 of temperature t and salinity s (arrays or DataArrays of any shape, unmasked),
    `metrics.get_density_at_surface`."""
    return get_density_at_surface(t, s, np.ones(())) - 1000


def sigma0(ds, exp=None):
    """TEOS-10 sigma_0 (float64) of the toce, soce of `ds`, NaN on the land of the tmask of `exp` (GRID)."""
    # In float32 the + 1000 of rho rounds it to 1.2e-4 kg m-3, which turns the near-zero vertical differences of the
    # data (pairs of levels with almost equal T, S) into ties and hides them from `density_error`.
    return teos_sigma0(ds.toce.astype(np.float64), ds.soce.astype(np.float64)).where((exp or GRID).mask.tmask == 1)


def zonal_sections(x):
    """Volume-weighted zonal mean of sigma_0 of a normalized batch x, (member, z_c - 1, y_c), last (land) level dropped."""
    section = zonal_mean(sigma0(to_dataset(x)), cell_volume(GRID))
    return section.transpose('member', 'z_c', 'y_c').values[:, :-1]


def density_error(x):
    """Percent of the ocean volume with a density inversion, per member of a normalized batch x.

    The vertical difference of sigma_0 is taken with the xgcm grid of GRID, as `Diagnostics.get_N_squared` does, onto
    the W points, and its negative values are weighted by e3w_0 e1t e2t. get_N_squared itself is not used: it clips
    N^2 below 1e-8 to 1e-7, which hides the inversions.
    """
    drho = GRID.grid.diff(sigma0(to_dataset(x)), 'Z', boundary='extend')
    mask = GRID.mask
    volume = (mask.e1t * mask.e2t * mask.e3w_0).where(drho.notnull(), 0.)
    return (100 * ((drho < 0) * volume).sum(('z_f', 'y_c', 'x_c')) / volume.sum(('z_f', 'y_c', 'x_c'))).values


def rmse(sections, reference):
    """Unweighted RMSE of each section (b, z, y) to `reference` (z, y), over the cells that are ocean in both."""
    return np.sqrt(np.nanmean((sections - reference[None]) ** 2, axis=(1, 2)))


def ts_histograms(fields, region, volume):
    """Fraction of the volume of `region` (z, y, x) in each (T, S) bin of TS_T_EDGES, TS_S_EDGES, per member of the
    unnormalized `fields` {'toce', 'soce': (member, z, y, x)}: (member, len(TS_T_EDGES) - 1, len(TS_S_EDGES) - 1)."""
    t = np.clip(fields['toce'][:, region], TS_T_EDGES[0], TS_T_EDGES[-1] - 1e-6)
    s = np.clip(fields['soce'][:, region], TS_S_EDGES[0], TS_S_EDGES[-1] - 1e-6)
    w = volume[region] / volume[region].sum()
    return np.stack([np.histogram2d(tb, sb, bins=[TS_T_EDGES, TS_S_EDGES], weights=w)[0] for tb, sb in zip(t, s)])


def ts_profiles(fields, tmask):
    """(T, S) profile of every column of the unnormalized `fields` {'toce', 'soce': (member, z, y, x)}, float64
    (member, y, x, 2 z), T then S, each level below the bottom (tmask (z, y, x) bool) filled with the deepest ocean
    value so that shallow columns are compared on their own water, NaN on land."""
    profiles = []
    for f in FIELDS:
        v = np.where(tmask[None], fields[f], np.nan)
        for k in range(1, v.shape[1]):
            v[:, k] = np.where(tmask[k][None], v[:, k], v[:, k - 1])
        profiles.append(np.moveaxis(v, 1, -1))
    return np.concatenate(profiles, axis=-1).astype(np.float64)


class ProfileClusters:
    """k-means (N_CLUSTERS) of the (T, S) profiles (`ts_profiles`) of the columns of a reference batch, T and S each
    scaled by their standard deviation over the reference, clusters numbered 1..N_CLUSTERS by increasing surface sigma_0
    (`teos_sigma0`) of their centre.

    fields : unnormalized {'toce', 'soce': (member, z, y, x)} of the reference batch; tmask : (z, y, x) bool.
    `centres` : (N_CLUSTERS, 2 z) profiles of the centres, T then S, in cluster order.
    """
    def __init__(self, fields, tmask):
        self.tmask, self.nz = tmask, tmask.shape[0]
        x = ts_profiles(fields, tmask)[:, tmask[0]]
        self.scale = np.repeat([np.nanstd(x[..., :self.nz]), np.nanstd(x[..., self.nz:])], self.nz)
        self.km = KMeans(N_CLUSTERS, n_init=10, random_state=0).fit(x.reshape(-1, 2 * self.nz) / self.scale)
        centres = self.km.cluster_centers_ * self.scale
        order = np.argsort(teos_sigma0(centres[:, 0], centres[:, self.nz]))
        self.rank = np.empty(N_CLUSTERS, int)
        self.rank[order] = np.arange(N_CLUSTERS)
        self.centres = centres[order]

    def labels(self, fields):
        """Cluster of every column of the unnormalized `fields` (nearest centre): uint8 (member, y, x), 0 on land."""
        x = ts_profiles(fields, self.tmask)
        ocean = self.tmask[0]
        out = np.zeros(x.shape[:3], 'u1')
        out[:, ocean] = self.rank[self.km.predict(x[:, ocean].reshape(-1, 2 * self.nz) / self.scale)].reshape(
            x.shape[0], -1) + 1
        return out

    def mean_profiles(self, fields, labels):
        """Mean (T, S) profile of the ocean cells of each member of `fields` in each cluster of `labels`, float32
        (member, N_CLUSTERS, 2, z - 1), last (land) level dropped, NaN where the cluster has no ocean cell at that level,
        and the number of columns of each member in each cluster (member, N_CLUSTERS)."""
        b, nz = labels.shape[0], self.nz - 1
        means = np.full((b, N_CLUSTERS, 2, nz), np.nan, '<f4')
        counts = np.zeros((b, N_CLUSTERS), int)
        for i in range(b):
            for k in range(N_CLUSTERS):
                columns = labels[i] == k + 1
                counts[i, k] = columns.sum()
                if not counts[i, k]:
                    continue
                ocean = self.tmask[:nz, columns]
                n = ocean.sum(axis=1)
                for j, f in enumerate(FIELDS):
                    v = np.where(ocean, fields[f][i, :nz][:, columns], 0.)
                    means[i, k, j] = np.where(n > 0, v.sum(axis=1) / np.maximum(n, 1), np.nan)
        return means, counts


class LevelClusters:
    """k-means (`k`) of the (T, S) of the ocean cells of each level of `levels`, one fit per level, on the centred values
    of a reference batch, T and S each scaled by their standard deviation over the reference cells of that level.
    Clusters are numbered 1..k by increasing sigma_0 (`teos_sigma0`) of their centre plus the mean of the level.

    anom : {f: (member, z, y, x)} centred values of FIELDS, NaN on land; tmask : (z, y, x) bool; levels : level indices;
    level_mean : {f: (z,)} the means added back to the centres to order them.
    """
    def __init__(self, anom, tmask, levels, level_mean, k=N_LEVEL_CLUSTERS):
        self.tmask, self.levels, self.k = tmask[levels], levels, k
        self.km, self.scale, self.rank = [], [], []
        for kz in levels:
            x = np.stack([anom[f][:, kz][:, tmask[kz]].ravel() for f in FIELDS], axis=1).astype(np.float64)
            scale = x.std(axis=0)
            km = KMeans(k, n_init=10, random_state=0).fit(x / scale)
            centres = km.cluster_centers_ * scale
            rank = np.empty(k, int)
            rank[np.argsort(teos_sigma0(centres[:, 0] + level_mean['toce'][kz],
                                        centres[:, 1] + level_mean['soce'][kz]))] = np.arange(k)
            self.km.append(km)
            self.scale.append(scale)
            self.rank.append(rank)

    def labels(self, values):
        """Cluster of the cells of the centred values {f: (member, len(levels), y, x)}: uint8, same shape, 0 on land."""
        b = values[FIELDS[0]].shape[0]
        out = np.zeros((b, *self.tmask.shape), 'u1')
        for l in range(len(self.levels)):
            m = self.tmask[l]
            x = np.stack([values[f][:, l][:, m].ravel() for f in FIELDS], axis=1).astype(np.float64)
            out[:, l][:, m] = (self.rank[l][self.km[l].predict(x / self.scale[l])] + 1).reshape(b, -1)
        return out


def panel_letter(k):
    """Letter of the k-th panel (0-based): a ... z, then aa, ab, ..."""
    return chr(ord('a') + k) if k < 26 else panel_letter(k // 26 - 1) + chr(ord('a') + k % 26)


def stretch(depth):
    """Stretched depth axis coordinate, depth ** DEPTH_POWER."""
    return np.asarray(depth) ** DEPTH_POWER


def depth_axis(ax, depth_max, ticks, labels=True):
    """Inverted stretched depth axis down to `depth_max`, ticks at `ticks` [m], labelled when `labels`."""
    ax.set_ylim(stretch(depth_max), 0)
    ax.set_yticks(stretch(ticks))
    ax.set_yticklabels([str(d) for d in ticks] if labels else [])


def channel_profile(section, lat):
    """Mean over the CHANNEL latitudes of a (depth, lat) section, NaN on levels with no ocean there."""
    return np.nanmean(section[:, (lat >= CHANNEL[0]) & (lat < CHANNEL[1])], axis=1)


def plot_section(ax, section, lat, depth, title, profile=None, depth_max=3500,
                 ticks=(0, 100, 250, 500, 1000, 1500, 2000, 3000), lines=0.6, profile_width=2.):
    """Contour a (depth, lat) sigma_0 section on the stretched depth axis, with `profile` (default: the mean of
    `section` over the CHANNEL latitudes) as a black inset on the right. Returns the filled contours."""
    y = stretch(depth)
    cf = ax.contourf(lat, y, section, levels=SIGMA_LEVELS, cmap='cmo.deep_r', extend='both')
    ax.contour(lat, y, section, levels=SIGMA_LEVELS, colors='k', linewidths=lines)
    for b in CHANNEL:
        ax.axvline(x=b, color='gray', linestyle='dashed', linewidth=1)
    ax.set_title(title, loc='left')
    depth_axis(ax, depth_max, ticks)
    if profile is None:
        profile = channel_profile(section, lat)
    # Axes.inset_axes, not the axes_grid1 one: its locator misplaces rasterized meshes with bbox_inches='tight'.
    inset = ax.inset_axes([0.85, 0, 0.15, 1])
    inset.plot(profile, y, color='black', linewidth=profile_width)
    inset.set_facecolor('gray')
    inset.patch.set_alpha(0.4)
    inset.set_xlim(PROFILE_RANGE)
    depth_axis(inset, depth_max, ticks, labels=False)
    inset.set_xticklabels([])
    inset.grid(True, linestyle='--', alpha=0.5)
    return cf


def setup_map(ax, left_labels):
    """Coastlines, borders, extent and gridlines of the paper maps (orthographic axes, longitudes shifted by -60)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax.coastlines(alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.left_labels = left_labels
    gl.ylabel_style = {'size': 6}


def aposteriori_experiment(results=None, files=()):
    """dinostics NemoExperiment of the 1/4° reference grid (domain, mesh mask, namelist of GRID_1_4DEG) holding in
    `exp.data` the daily outputs `files` (e.g. 'T_3D') of the a posteriori run `results`.

    The a posteriori runs do not collect their grid files, which are those of the reference. The `EXP` prefix of the
    glob keeps out the ._* AppleDouble files of the external volume, which `NemoExperiment.open_data` would match.
    """
    exp = NemoExperiment(REFERENCES, GRID_1_4DEG, restarts=[''], files=[])
    for name in files:
        exp.data[name] = xn.open_nemo(domcfg=exp.domain, files=sorted(glob(f'{results}/{EXP}_*_grid_{name}.nc')))
    return exp


def restart_state(exp, path):
    """toce, soce (z_c, y_c, x_c) and uoce (t, z_c, y_c, x_f) of the now fields of the 1/4° restart `path`, read with
    `NemoExperiment.open_restart` of `exp` and set on its grid as `GridManipulation.regrid_restart` does the reverse."""
    r = exp.open_restart(path).isel(time_counter=0).drop_vars(['lon', 'lat', 'nav_lev'])
    r = r.rename(nav_lev='z_c', y='y_c', x='x_c').assign_coords({c: exp.domain[c] for c in ('z_c', 'y_c', 'x_c')})
    uoce = r.un.rename(x_c='x_f').assign_coords(x_f=exp.domain.x_f).expand_dims('t')
    return xr.Dataset({'toce': r.tn, 'soce': r.sn, 'uoce': uoce})


def collected_runs(methods, experiment, prefix=PREFIX):
    """aposteriori/ folder of each of `methods` (every run of `prefix` when None) whose `experiment` run is collected,
    warning about the others."""
    folders = aposteriori_dirs(prefix=prefix, constraints=methods)
    runs = {}
    for method in folders if methods is None else methods:
        folder = folders.get(method)
        if folder is None or not os.path.exists(f'{folder}/{experiment_paths(experiment)["results"]}/{COLLECT}'):
            print(f'WARNING : no collected {experiment} run for {method}, skipped')
        else:
            runs[method] = folder
    return runs


# Skipped on importlib.reload(common), which keeps the module globals.
if 'batch' not in globals():
    GRID = NemoExperiment(REFERENCES, GRID_1DEG, restarts=[''], files=[])
    config = TrainingConfig()
    observer = Observer('data', config, device='cpu')
    DATA_DIR = os.path.dirname(by_timestamp(glob(f'{INFERENCE}/{DATA_PREFIX}/{DATA_RUN}/*/{DATA_STATES}'))[-1])
    batch = torch.from_numpy(np.load(f'{DATA_DIR}/{DATA_STATES}'))
    with open(f'{DATA_DIR}/args.json') as f:
        DATA_ARGS = json.load(f)
    REFERENCE_DENSITY = np.load(REFERENCE_DENSITY_FILE)
    gens, gens_args = load_generations()

sources = {'data': batch} | gens

# Runs of each figure for the sweep PREFIX: fig1 and fig2 main and appendix rows, table row order, and for fig4 the glob
# of the eta series, the glob of its paired unconstrained run (latest match kept) and the etas of its frieze.
_OPTIMAL = ['data', 'none', 'none_border0', DENSITY_RUN, 'zero_mean_border0']
_OCTOBER = ['data', 'none', 'C', 'C_B0', 'C_NL', 'C_NL_zm0.1', 'C_NL_C7030_sTS', 'C_NL_C7010']
PRESETS = {
    'optimal': {'fig1_main': ['data', 'none', DENSITY_RUN, 'zero_mean_border0'],
                'fig1_appendix': _OPTIMAL,
                'fig2_main': ['data', DENSITY_RUN, 'zero_mean_border0'],
                'fig2_appendix': _OPTIMAL,
                'table': _OPTIMAL,
                'fig4_sweep': f'{INFERENCE}/sweep_density_huber/negative_grad_density_huber_border0',
                'fig4_none': f'{INFERENCE}/sweep_baselines/none_border0_b8/151026-145216',
                'fig4_frieze': [10, 100, 300]},
    'october': {'fig1_main': ['data', 'none', 'C_NL', 'C_NL_zm0.1', 'C_NL_C7010'],
                'fig1_appendix': _OCTOBER,
                'fig2_main': ['data', 'C_NL', 'C_NL_zm0.1', 'C_NL_C7010'],
                'fig2_appendix': _OCTOBER,
                'table': _OCTOBER + ['C_NL_b8', 'C_NL_C7030', 'C_NL_C7050', 'C_NL_C7100', 'NL', 'NL_C7010', 'NL_C7030',
                                     'NL_C7050', 'NL_C7100'],
                'fig4_sweep': f'{INFERENCE}/october/C_NL_C7[0-9][0-9][0-9]',
                'fig4_none': f'{INFERENCE}/october/C_NL_b8/*',
                'fig4_frieze': [10, 30, 100]},
}
CONFIGS = PRESETS[PREFIX]
