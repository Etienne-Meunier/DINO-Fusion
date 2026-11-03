#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared state for the paper scripts: data batch, losses and the generations of one sweep of one run.

Imported with ``from common import *`` so an interactive session loads the data once: `importlib.reload(common)`
redefines the functions but keeps the loaded data and the `memo` cache (see `session.py`). The model and sweep
are RUN / PREFIX, overridable with the PAPER_RUN / PAPER_PREFIX environment variables. Figures are saved in the
`figures` folder of the sweep (FIGURES, overridable with PAPER_FIGURES), and CONFIGS holds the runs each figure shows for
that sweep. DENSITY_RUN is the increasing-density run every figure shows, a priori (through the 'optimal' preset of
CONFIGS) and a posteriori (imported by the fig5-fig8 scripts): eta = 30 by default, overridable with PAPER_DENSITY_RUN
(see `compare_eta.py`). The 'october' preset has its own runs and ignores it.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import inspect
import json
import os
import subprocess
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, eps_data, get_loss
from metrics import get_density_at_surface

device = 'cpu'
batch_size = 200

RUN = os.environ.get('PAPER_RUN', 'tav0h83b')
PREFIX = os.environ.get('PAPER_PREFIX', 'optimal')
INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/{RUN}/inference'
FIGURES = os.environ.get('PAPER_FIGURES', f'{INFERENCE}/{PREFIX}/figures')
DENSITY_RUN = os.environ.get('PAPER_DENSITY_RUN', 'negative_grad_density_huber_border0_30')
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


_CACHE = globals().get('_CACHE', {})


def memo(f):
    """Cache the results of `f` across runs of the figure scripts in one session, keyed by its file, name and arguments.

    The cache lives in this module, so it survives `%run` of a script redefining `f`; when the body of `f` changes,
    clear it with `forget(f.__name__)`. Tensors, arrays and xarray objects are keyed by identity (the batch, the
    generations, `load_mask()`), other arguments by their repr.
    """
    def token(a):
        return ('id', id(a)) if isinstance(a, (torch.Tensor, np.ndarray, xr.DataArray, xr.Dataset)) else repr(a)

    def wrapped(*args, **kwargs):
        values = list(args) + [v for _, v in sorted(kwargs.items())]
        key = (os.path.basename(f.__code__.co_filename), f.__name__, tuple(map(token, args)),
               tuple((k, token(v)) for k, v in sorted(kwargs.items())))
        if key not in _CACHE:
            # The arguments are stored with the result: an id stays unique while its object is alive.
            _CACHE[key] = (values, f(*args, **kwargs))
        return _CACHE[key][1]
    wrapped.__name__, wrapped.__doc__ = f.__name__, f.__doc__
    return wrapped


def forget(*names):
    """Drop the `memo` entries of the functions `names`, every entry when none is given."""
    for key in [k for k in _CACHE if not names or k[1] in names]:
        del _CACHE[key]


MASK_FILE = f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc'


def provenance(sources=(), depth=1):
    """Description and inputs of the figure script calling (`depth` frames up): {'Subject': its docstring, 'Creator':
    its path at the current git commit (``-dirty`` if the script has uncommitted changes), 'Keywords': RUN, PREFIX,
    the training data, the mask, the generation folders of the sweep and `sources`, one per line}."""
    frame = inspect.stack()[depth].frame
    script = os.path.abspath(frame.f_globals.get('__file__', '<interactive>'))
    git = lambda *a: subprocess.run(['git', '-C', PRP, *a], capture_output=True, text=True).stdout.strip()
    commit = git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain', '--', script) else '')
    inputs = [f'run {RUN}, sweep {PREFIX}', config.data_file, MASK_FILE, *generation_dirs().values(), *map(str, sources)]
    return {'Subject': (frame.f_globals.get('__doc__') or '').strip(), 'Creator': f'{script} @ {commit}',
            'Keywords': '\n'.join(inputs)}


def save(fig, path, sources=(), **kwargs):
    """`fig.savefig(path, **kwargs)` with the `provenance` of the calling script as PDF metadata, Title the file name."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, metadata={'Title': os.path.basename(path)} | provenance(sources, depth=2), **kwargs)


def save_tex(text, path, sources=()):
    """Write the LaTeX `text` to `path`, preceded by the `provenance` of the calling script as % comments."""
    header = '\n'.join(f'{k}: {v}' for k, v in provenance(sources, depth=2).items())
    with open(path, 'w') as f:
        f.write(''.join(f'% {line}\n' for line in header.splitlines()) + text)


def generation_dirs(prefix=PREFIX, inference=INFERENCE):
    """Folder ``<inference>/<prefix>/<constraint>/<timestamp>/`` holding the generated.npy of every run, keyed by constraint.

    A run is expected to have a single timestamp: when it has several, an error is printed and the latest one is kept.
    """
    dirs = {}
    for run_dir in sorted(glob(f'{inference}/{prefix}/*/')):
        constraint = os.path.basename(os.path.normpath(run_dir))
        versions = sorted(glob(f'{run_dir}/*/generated.npy'),
                          key=lambda v: datetime.strptime(os.path.basename(os.path.dirname(v)), '%d%m%y-%H%M%S'))
        if not versions:
            continue
        if len(versions) > 1:
            print(f'ERROR : {prefix}/{constraint} has {len(versions)} versions '
                  f'({", ".join(os.path.basename(os.path.dirname(v)) for v in versions)}), keeping the latest')
        dirs[constraint] = os.path.dirname(versions[-1])
    return dirs


def load_generations(prefix=PREFIX, inference=INFERENCE, device=device):
    """Load the generation of every run of `generation_dirs`.

    Returns (gens, args) dicts keyed by constraint name, `args` holding the matching args.json.
    """
    gens, args = {}, {}
    for constraint, path in generation_dirs(prefix, inference).items():
        gens[constraint] = torch.from_numpy(np.load(f'{path}/generated.npy')).to(device)
        with open(f'{path}/args.json') as f:
            args[constraint] = json.load(f)
    return gens, args


@memo
def load_mask():
    """Return the 1 degree DINO mesh mask with dimensions renamed to (depth, nav_lat, nav_lon)."""
    file_mask_LR = xr.open_dataset(MASK_FILE).sel(time_counter=0)
    return file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})


def compute_density(fields, file_mask_LR):
    """Potential density referenced at the surface, (batch, depth, nav_lat, nav_lon), of unnormalized 'toce', 'soce'."""
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce, toce, tmask = data.copy(), data.copy(), data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce[:] = np.asarray(fields['toce'])
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


@memo
def zonal_sections(x, file_mask_LR):
    """Volume-weighted zonal mean of sigma_0 - 1000 of a normalized batch x, (batch, depth - 1, nav_lat), last (land) level dropped."""
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    section = (compute_density(fields, file_mask_LR) * volume).sum('nav_lon') / volume.sum('nav_lon')
    return (section.transpose('batch', 'depth', 'nav_lat')[:, :-1] - 1000).values


def rmse(sections, reference):
    """Unweighted RMSE of each section (b, z, y) to `reference` (z, y), over the cells that are ocean in both."""
    return np.sqrt(np.nanmean((sections - reference[None]) ** 2, axis=(1, 2)))


def panel_letter(k):
    """Letter of the k-th panel (0-based): a ... z, then aa, ab, ..."""
    return chr(ord('a') + k) if k < 26 else panel_letter(k // 26 - 1) + chr(ord('a') + k % 26)


def zonal_mean_T(file_mask_LR, t):
    """Area-weighted zonal mean (n, k, y) of the first k levels of T (n, k, y, x), land as 0."""
    area = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[:t.shape[1]]
    return (np.nan_to_num(t) * area).sum(-1) / np.maximum(area.sum(-1), 1e-9)


def training_zonal_T(file_mask_LR, k):
    """Zonal-mean T (n, k, y) of every training state in archive order (period 36 states, one year).

    Read once from the training tar and cached next to it as seasonality_zonal_T.npz, with 14 levels.
    """
    cache = f'{os.path.dirname(config.data_file)}/seasonality_zonal_T.npz'
    if not os.path.exists(cache):
        import io, tarfile
        out = {}
        with tarfile.open(config.data_file) as tar:
            for m in tar:
                if m.name.startswith('infos/') or not m.isfile():
                    continue
                key, field, _ = m.name.split('/')[-1].split('.')
                if field == 'toce':
                    out[key] = zonal_mean_T(file_mask_LR, np.load(io.BytesIO(tar.extractfile(m).read()))[None, :14])[0].astype(np.float32)
        np.savez(cache, T=np.stack([out[k] for k in sorted(out)]), keys=np.array(sorted(out)))
    return np.load(cache)['T'][:, :k]


def median_member(errors):
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


# Skipped on importlib.reload(common), which keeps the module globals.
if 'batch' not in globals():
    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
    dataloader = get_dataloader(config.data_file, transform, batch_size=batch_size, shuffle=True, seed=DATA_SEED)
    observer = Observer('data', config, device=device)
    losses = {name: get_loss(name, observer) for name in LOSSES}

    batch = next(iter(dataloader))
    gens, gens_args = load_generations()

sources = {'data': batch} | gens

# Runs of each figure for the sweep PREFIX : fig1 and fig2 main and appendix rows, fig3 panels, zonal bias lines, table1 row order, and for fig4 the
# glob of the eta series, the glob of its paired unconstrained run (latest match kept) and the etas of its frieze
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
                'fig4_frieze': [10, 100, 300],
                'seasons': _OPTIMAL,
                'zonal_bias': _OPTIMAL[1:]},
    'october': {'fig1_main': ['data', 'none', 'C_NL', 'C_NL_zm0.1', 'C_NL_C7010'],
                'fig1_appendix': _OCTOBER,
                'fig2_main': ['data', 'C_NL', 'C_NL_zm0.1', 'C_NL_C7010'],
                'fig2_appendix': _OCTOBER,
                'table': _OCTOBER + ['C_NL_b8', 'C_NL_C7030', 'C_NL_C7050', 'C_NL_C7100', 'NL', 'NL_C7010', 'NL_C7030',
                                     'NL_C7050', 'NL_C7100'],
                'fig4_sweep': f'{INFERENCE}/october/C_NL_C7[0-9][0-9][0-9]',
                'fig4_none': f'{INFERENCE}/october/C_NL_b8/*',
                'fig4_frieze': [10, 30, 100],
                'seasons': _OCTOBER,
                'zonal_bias': _OCTOBER[1:]},
}
CONFIGS = PRESETS[PREFIX]
