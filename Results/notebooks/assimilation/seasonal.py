"""
Shared constants and helpers of the phase 0 scripts of the assimilation study (see plan.md): reading the training tar,
the mesh mask, PDF provenance, the annual harmonic fit over the last N_YEARS years of states and its noise floor, and
the NH-band mean T profile (band_profile) that phase 1 applies to generated states after generated_toce.

The training data are N_STATES time-ordered states, YEAR = 36 per year (10 days apart in a 360-day calendar). Phase is
the state index modulo YEAR; NH summer is near phase SUMMER, NH winter near phase WINTER. The diffusion model sees the
toce levels LEVELS only (TrainingConfig().fields['toce']).

Inputs (read by the helpers):
    DATA_FILE  training tar, unnormalised NNNNN.{toce,soce,ssh}.npy, time-ordered (read, from the open tar)
    MASK_FILE  1 degree DINO mesh mask (load_mask, band_weights, band_profile with w None)
Outputs: none; `save` writes the figures of the calling script.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')

import io
import os
import subprocess

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from configs.base_config import TrainingConfig

YEAR = 36
N_STATES = 1800
N_YEARS = 5
INDICES = np.arange(N_STATES - N_YEARS * YEAR, N_STATES)
SUMMER, WINTER = 24, 7
MASK_FILE = f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc'
OUT = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference/assimilation/phase0'
DATA_FILE = TrainingConfig().data_file
N_TOCE_LEVELS = 36
LEVELS = np.arange(N_TOCE_LEVELS)[TrainingConfig().fields['toce']]
INTERIOR = slice(1, -1)
NOISE_PERIODS = (N_YEARS * YEAR / 4, N_YEARS * YEAR / 6)
NH_BAND = (15, 70)


def save(fig, path, sources=(), **kwargs):
    """`fig.savefig(path, **kwargs)` with PDF metadata: Title the file name, Subject the docstring of the running
    script, Creator that script at the current git commit (``-dirty`` if it or this module has uncommitted changes),
    Keywords the inputs, one per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    main = sys.modules['__main__']
    script = os.path.abspath(main.__file__)
    git = lambda *a: subprocess.run(['git', '-C', PRP, *a], capture_output=True, text=True).stdout.strip()
    dirty = git('status', '--porcelain', '--', script, os.path.abspath(__file__))
    commit = git('rev-parse', '--short', 'HEAD') + ('-dirty' if dirty else '')
    metadata = {'Title': os.path.basename(path), 'Subject': main.__doc__.strip(), 'Creator': f'{script} @ {commit}',
                'Keywords': '\n'.join([DATA_FILE, MASK_FILE, *map(str, sources)])}
    fig.savefig(path, metadata=metadata, **kwargs)


def load_mask(interior=True):
    """Mesh mask fields as numpy arrays: tmask (z, y, x) bool, ocean = tmask[0], area e1t * e2t, lon glamt, lat gphit
    (y, x), depth gdept_1d (z). `interior` drops the halo columns 0 and -1."""
    m = xr.open_dataset(MASK_FILE).isel(time_counter=0)
    if interior:
        m = m.isel(nav_lon=INTERIOR)
    tmask = m.tmask.values.astype(bool)
    return {'tmask': tmask, 'ocean': tmask[0], 'area': (m.e1t * m.e2t).values, 'lon': m.glamt.values,
            'lat': m.gphit.values, 'depth': m.gdept_1d.values}


def read(tar, index, field):
    """Field `field` ('toce', 'soce' or 'ssh') of state `index` from the open training tar, unnormalised."""
    return np.load(io.BytesIO(tar.extractfile(f'{index:05d}.{field}.npy').read()))


def area_mean(field, w):
    """Weighted mean of `field` (..., y, x) over the last two axes with weights `w` (y, x), zero on land."""
    return np.nansum(field * w, axis=(-2, -1)) / w.sum()


def weighted_quantile(values, weights, q):
    """Quantiles `q` of the 1-D `values` under `weights`."""
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return np.interp(q, cdf, values[order])


def harmonic_fit(series, period=YEAR, also=()):
    """Least-squares fit of a + b*(t - mean(t)) + c*cos(w*t) + d*sin(w*t), t = INDICES, w = 2*pi/`period`, to `series`
    (t, ...), jointly with a cos/sin pair at each period of `also` (in states). `a` is the trend value at the window
    centre mean(t), not at t = 0. Points with a non-finite value at any time are left NaN. Returns the dict of
    coefficient arrays a, b, c, d (shape ...), the anomaly series - trend - harmonics at `also` (detrended anomaly if
    `also` is empty), the harmonic c*cos + d*sin at `period` and the residual anomaly - harmonic, all (t, ...)."""
    t = INDICES.astype(float)
    pairs = [[np.cos(2 * np.pi * t / p), np.sin(2 * np.pi * t / p)] for p in (period, *also)]
    X = np.stack([np.ones_like(t), t - t.mean(), *sum(pairs, [])], axis=1)
    flat = series.reshape(len(t), -1)
    valid = np.all(np.isfinite(flat), axis=0)
    coef = np.full((X.shape[1], flat.shape[1]), np.nan)
    coef[:, valid] = np.linalg.lstsq(X, flat[:, valid], rcond=None)[0]
    shape = series.shape[1:]
    a, b, c, d = (k.reshape(shape) for k in coef[:4])
    background = np.concatenate([X[:, :2], X[:, 4:]], axis=1) @ np.concatenate([coef[:2], coef[4:]])
    anomaly = series - background.reshape(series.shape)
    harmonic = (X[:, 2:4] @ coef[2:4]).reshape(series.shape)
    return {'a': a, 'b': b, 'c': c, 'd': d, 'anomaly': anomaly, 'harmonic': harmonic, 'residual': anomaly - harmonic}


def amplitude(fit):
    """Annual harmonic amplitude sqrt(c^2 + d^2) of `fit`."""
    return np.hypot(fit['c'], fit['d'])


def noise_amplitude(series):
    """Amplitude of the harmonic at each non-annual period of NOISE_PERIODS, fitted jointly with the trend and the
    annual harmonic so that the annual signal does not leak into it: (len(NOISE_PERIODS), ...) for `series` (t, ...),
    in the unit of `series`. Its median is the noise floor of the annual amplitude."""
    return np.stack([amplitude(harmonic_fit(series, p, also=(YEAR,))) for p in NOISE_PERIODS])


def peak_phase(fit):
    """Phase (in states, 0 to YEAR) of the maximum of the annual harmonic of `fit`: atan2(d, c) / w modulo YEAR."""
    return (np.arctan2(fit['d'], fit['c']) * YEAR / (2 * np.pi)) % YEAR


def explained_fraction(fit):
    """Fraction of the variance of the detrended anomaly explained by the annual harmonic, per point."""
    return np.var(fit['harmonic'], axis=0) / np.var(fit['anomaly'], axis=0)


def composite(anomaly):
    """Mean of `anomaly` (t, ...) over the phase-SUMMER states minus its mean over the phase-WINTER states."""
    phase = INDICES % YEAR
    return anomaly[phase == SUMMER].mean(0) - anomaly[phase == WINTER].mean(0)


def zonal_mean(field, w):
    """Weighted mean of `field` (..., y, x) along x with weights `w` (..., y, x), NaN on all-land rows."""
    with np.errstate(invalid='ignore'):
        return np.nansum(field * w, axis=-1) / np.where(w.sum(-1) > 0, w.sum(-1), np.nan)


def seasonal_cycle(anomaly):
    """Mean of `anomaly` (t, ...) over the states of each phase 0 .. YEAR - 1, (YEAR, ...)."""
    return anomaly.reshape(N_YEARS, YEAR, *anomaly.shape[1:]).mean(0)


def band_weights(mask=None, band=NH_BAND, levels=LEVELS):
    """Area weights (len(levels), y, x) on the full grid (halo included) of the ocean points of the toce `levels`
    with band[0] <= gphit <= band[1], zero on the halo columns, on land and outside the band. `mask` is
    load_mask(interior=False), loaded if None."""
    mask = load_mask(interior=False) if mask is None else mask
    in_band = (mask['lat'] >= band[0]) & (mask['lat'] <= band[1])
    w = mask['area'] * in_band * mask['tmask'][levels]
    w[..., 0] = w[..., -1] = 0
    return w


def band_profile(toce, w=None):
    """Area-weighted mean T profile over the NH band of each state. `toce` (n, 18, 199, 62) numpy, unnormalised degC,
    at the model levels in LEVELS order, on the unpadded file grid with the halo columns (generated_toce for a
    generated batch); land values are ignored whatever they are (NaN or not). `w` (18, 199, 62) the band_weights,
    best precomputed once with band_weights(): if None, they are recomputed for NH_BAND and LEVELS, reloading
    MASK_FILE at every call. Returns (n, 18) in degC."""
    w = band_weights() if w is None else w
    return np.einsum('nkyx,kyx->nk', np.where(w > 0, toce, 0), w) / w.sum(axis=(-2, -1))


def map_panel(ax, field, mask, cmap, label, vmin=None, vmax=None, symmetric=False, colorbar=True, log=False):
    """Draw `field` (y, x) on `ax` on the lon/lat grid of `mask`, land (mask['ocean'] False or NaN) blank, with a
    colourbar `label` unless `colorbar` is False. `symmetric` centres the colour range at 0, on `vmax` if given, else on
    the largest absolute value. `log` uses a log colour scale between `vmin` and `vmax` (both required, > 0; values
    <= 0 blank). Values beyond the range saturate, flagged by the colourbar extension. Returns the QuadMesh."""
    field = np.where(mask['ocean'], field, np.nan)
    if symmetric:
        vmax = np.nanmax(np.abs(field)) if vmax is None else vmax
        vmin = -vmax
    lo, hi = np.nanmin(field), np.nanmax(field)
    extend = {(False, False): 'neither', (True, False): 'min', (False, True): 'max', (True, True): 'both'}[
        (vmin is not None and lo < vmin, vmax is not None and hi > vmax)]
    scale = dict(norm=LogNorm(vmin, vmax)) if log else dict(vmin=vmin, vmax=vmax)
    pm = ax.pcolormesh(mask['lon'], mask['lat'], field, cmap=cmap, shading='nearest', rasterized=True, **scale)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude (deg)')
    if colorbar:
        plt.colorbar(pm, ax=ax, label=label, shrink=0.8, pad=0.03, extend=extend)
    return pm


def generated_toce(x, observer):
    """T (b, len(LEVELS), y, x) degC numpy on the full file grid, land NaN, of a normalised generated batch `x`
    (b, 37, 208, 64) tensor, through `observer` (Observer('data', TrainingConfig(), device))."""
    return observer.unnormalized(x, result='strided')['toce'].detach().cpu().numpy()
