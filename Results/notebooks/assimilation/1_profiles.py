#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 of the assimilation study (see plan.md): density and T profiles of the states generated conditioned on a fixed
SST (1_generate_sst.py), on whatever runs exist in the folder of one run set (SETS, --set): centred (phase1.OUT1,
sampler centred) or standard (phase1_standard, sampler standard). The figures go to the folder of the set.

    python 1_profiles.py --set standard

Density as `paper/common.compute_density` (get_density_at_surface of `analysis_scripts/metrics.py`): sigma_0 =
potential density referenced at the surface - 1000, of the fields unnormalised to the 36 file levels, its
volume-weighted horizontal mean per level. The data reference (last-year states, phase1.CACHE1) goes through the same
normalisation round trip as the generated states (36 -> 18 model levels -> 36). Inversion diagnostics per member:
E = % of the ocean volume where sigma_0 decreases with depth (metrics.compute_density_error, the E of
`machine/surface_bias/experiments/evaluate.py`), "cols %" the % of columns with at least one strict decrease of sigma_0
with depth among the columns with at least 2 ocean levels, and "mInv" whether the horizontal-mean profile decreases
somewhere; the level means of mInv are over horizontal domains that shrink with depth, so mInv also compares different
regions. The density profile, E and cols % include the halo columns 0 and -1, for comparability with evaluate.py. The
band T profile is seasonal.band_profile (halo columns excluded), as an anomaly to its 5-year mean over the phase 0
cache, compared with the unconditioned runs of the same sampler and configs (SETS uncond): the centred b32 runs of
machine/surface_bias/experiments for the centred set, the latest paper "optimal" runs for the standard set. The
standard set has a fourth config, B0 (no constraint, BorderZero after every step). Runs of mode clean
(1_generate_sst.py --mode clean) are keyed <season>_<config>_clean and drawn with a
distinct line style.

Inputs:
    <set dir>/<run>/generated.npy, args.json  runs of 1_generate_sst.py
    phase1.CACHE1                             last-year data states
    phase0 toce_levels_1620_1799.npz          T at the model levels of the last 5 years (5-year mean band profile)
    SETS[set]['uncond']                       unconditioned runs of the same sampler and configs
Outputs in the set dir (phase1.RUN_SETS):
    density_profiles.pdf       per run (rows summer / winter, columns the configs of the set, none / [B0 /] C7 / TS)
                               the horizontal-mean sigma_0 profile of each member, the source state and the last-year
                               data states, full depth
    density_profiles_top.pdf   the same over the top TOP_DEPTH m
    density_profiles_seasons.pdf      summer and winter runs overlaid per config (rows full depth / top
                                      TOP_DEPTH m): members, member means, source states, data in the background,
                                      and a first column of the data states within SEASON_HALFWIDTH phases of
                                      SUMMER / WINTER coloured by season
    density_profiles_seasons_nh.pdf   the same for the area-weighted sigma_0 profile over the NH band of BANDS
                                      (seasonal.band_profile at the 36 file levels, halo excluded), with a third row
                                      over the top SURFACE_DEPTH m
    temperature_profiles_seasons.pdf  the same for the NH-band T profile (absolute degC), 3 rows
    band_t_profile.pdf         NH-band T anomaly over the first N_BAND levels of the members of each run, the data
                               summer / winter source states and phase means, the unconditioned run of the set
    sst_check.pdf              per run, max |SST - source SST| over the ocean of each member
    t_hist_seasons.pdf         rows the first N_HIST model T levels, columns data / configs of the set: area-weighted
                               density histograms of T over the ocean of the band HIST_BAND, 40-70N (halo
                               excluded), summer (filled) and winter (outline) overlaid, the source states for data
                               and all members pooled for a config, bins and x axes shared within a row (HIST_PERCENTILES of the row,
                               HIST_BINS bins), the mass outside the bins left out of the area
Printed: per run, the inversion diagnostics, the SST check and the band anomaly at BAND_DEPTHS vs the data and the
unconditioned runs, and the x ranges of the rows of t_hist_seasons.pdf.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/analysis_scripts/'); sys.path.append(f'{PRP}/Diffusion_Model/')

import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
import xarray as xr

from configs.base_config import TrainingConfig
from metrics import compute_density_error, get_density_at_surface
from pipelines.State import Observer

from phase1 import CACHE1, RUN_SETS, SOURCES, load_states
from seasonal import (INDICES, LEVELS, MASK_FILE, N_TOCE_LEVELS, NH_BAND, OUT, SUMMER, WINTER, YEAR, band_profile,
                      band_weights, generated_toce, save, weighted_quantile)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

UNCOND = f'{PRP}/Results/notebooks/machine/surface_bias/experiments/runs'
OPTIMAL = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference/optimal'
SETS = {'centred': dict(dir=RUN_SETS['centred'],
                        labels={'none': 'none', 'c7': 'C7 (Huber density, eta 10)', 'ts': 'TS (zero_mean, eta 1)'},
                        fig1_labels={'none': ('no constraint',) * 2,
                                     'c7': ('C7 (Huber density, eta 10)', 'C7 (Huber, eta 10)'),
                                     'ts': ('TS (T,S mean)',) * 2},
                        uncond={'none': f'{UNCOND}/centred_none_b32', 'c7': f'{UNCOND}/centred_c7010_b32',
                                'ts': f'{UNCOND}/centred_zm_e1_b32'},
                        uncond_label='unconditioned centred'),
        'standard': dict(dir=RUN_SETS['standard'],
                         labels={'none': 'none (no projection)', 'b0': 'B0 (no constraint, BorderZero)',
                                 'c7': 'C7 (Huber density, eta 30)', 'ts': 'TS (zero_mean, eta 0.1)'},
                         fig1_labels={'none': ('no constraint, no projection', 'no constraint'),
                                      'b0': ('B0 (no constraint, BorderZero)', 'B0 (BorderZero)'),
                                      'c7': ('C7 (Huber density, eta 30)', 'C7 (Huber, eta 30)'),
                                      'ts': ('TS (T,S mean, eta 0.1)', 'TS (T,S mean)')},
                         uncond={'none': f'{OPTIMAL}/none/*', 'b0': f'{OPTIMAL}/none_border0/*', 'c7': f'{OPTIMAL}/negative_grad_density_huber_border0_30/*',
                                 'ts': f'{OPTIMAL}/zero_mean_border0/*'},
                         uncond_label='unconditioned optimal')}
SEASONS = {'summer': ('tab:red', SUMMER), 'winter': ('tab:blue', WINTER)}
MODES = {'': ('-', ''), '_clean': ('--', ' clean')}
T_CACHE = f'{OUT}/toce_levels_{INDICES[0]}_{INDICES[-1]}.npz'
DEPTH_POWER, DEPTH_TICKS = 0.625, [0, 100, 250, 500, 1000, 1500, 2000, 3000]
TOP_DEPTH = 500
SURFACE_DEPTH = 100
SEASON_ROWS = (None, TOP_DEPTH, SURFACE_DEPTH)
BANDS = {'nh': (NH_BAND, f'NH band {NH_BAND[0]}-{NH_BAND[1]}N')}
SEASON_HALFWIDTH = 3
SIGMA_LABEL = r'$\sigma_0$ [kg m$^{-3}$ - 1000]'
N_BAND = 8
BAND_DEPTHS = [5, 49, 74, 102]
N_HIST = 5
HIST_BINS = 60
HIST_PERCENTILES = (0.5, 99.5)
HIST_BAND = (50, 70)

observer = Observer('data', TrainingConfig(), 'cpu')
mask = xr.open_dataset(MASK_FILE).sel(time_counter=0).rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
tmask = mask.tmask.values.astype(bool)
volume = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values
depth = mask.gdept_1d.values.squeeze()
OCEAN_LEVELS = volume.sum((1, 2)) > 0
stretch = lambda d: np.asarray(d) ** DEPTH_POWER
BAND_WEIGHTS = {k: band_weights(band=b, levels=np.arange(N_TOCE_LEVELS)) for k, (b, _) in BANDS.items()}


def sigma0(fields):
    """sigma_0 (b, 36, y, x) of unnormalised 'toce', 'soce' (b, 36, y, x), NaN on land (common.compute_density)."""
    b = fields['soce'].shape[0]
    data = mask.e3t_0.copy().expand_dims({'batch': b})
    toce, soce, tm = data.copy(), data.copy(), data.copy()
    toce[:], soce[:] = np.asarray(fields['toce']), np.asarray(fields['soce'])
    tm[:] = np.repeat(tmask[None], b, 0)
    rho = get_density_at_surface(toce, soce, tm).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values
    return np.where(tmask[None], rho - 1000, np.nan)


def diagnostics(x):
    """Diagnostics of a normalised batch x (b, 37, 208, 64): dict of the volume-weighted horizontal-mean sigma_0
    profile (b, 36) in kg m-3 - 1000, the same horizontal mean of T (b, 36) in degC, E (b,) in %, the % of columns with a strict inversion among the columns with at
    least 2 ocean levels (b,), the mean-profile inversion flag (b,) (module docstring), all with the halo columns, T at
    the model levels (b, 18, 199, 62) in degC (generated_toce), and per key k of BANDS the area-weighted band profiles
    'profile_k' and 'T_profile_k' (b, 36) at the 36 file levels (band_profile, halo excluded)."""
    f = {k: v.detach().numpy().astype(np.float64) for k, v in observer.unnormalized(x).items() if k != 'ssh'}
    sig = sigma0(f)
    hmean = lambda a: np.nansum(a * volume, axis=(2, 3)) / np.where(OCEAN_LEVELS, volume.sum((1, 2)), np.nan)
    prof = hmean(sig)
    dsig = np.diff(sig, axis=1)
    col_ocean = tmask[1:].any(0)
    inverted_col = (np.nan_to_num(dsig, nan=0.) < 0).any(1)
    p = prof[:, OCEAN_LEVELS]
    bands = {}
    for k, w in BAND_WEIGHTS.items():
        # the last file level has no ocean point: 0/0 -> NaN there, a level outside OCEAN_LEVELS, never plotted
        with np.errstate(invalid='ignore'):
            bands |= {f'profile_{k}': band_profile(sig, w), f'T_profile_{k}': band_profile(f['toce'], w)}
    return {'profile': prof, 'T_profile': hmean(np.where(tmask[None], f['toce'], np.nan)), 'E': compute_density_error(f, mask), 'columns': 100 * inverted_col[:, col_ocean].mean(1),
            'mean_inv': (np.diff(p, axis=1) < 0).any(1), 'T': generated_toce(x, observer)} | bands


def parse_set():
    """Entry of SETS named by the --set argument of the command line (default centred), with its name under 'name'."""
    parser = argparse.ArgumentParser(description=sys.modules['__main__'].__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', choices=list(SETS), default='centred')
    name = parser.parse_args().set
    return SETS[name] | {'name': name}


def latest_run(pattern):
    """Folder matching `pattern` (glob) with a generated.npy, the latest by its ddmmyy-HHMMSS name if several (folders
    with another name sort first, by name), None if there is none."""
    def stamp(d):
        try:
            return datetime.strptime(os.path.basename(d), '%d%m%y-%H%M%S')
        except ValueError:
            return datetime.min
    folders = [os.path.dirname(p) for p in glob.glob(f'{pattern}/generated.npy')]
    return max(folders, key=lambda d: (stamp(d), d), default=None)


def runs(folder):
    """{name: (args, normalised batch)} of the runs of `folder` with a generated.npy."""
    out = {}
    for path in sorted(glob.glob(f'{folder}/*/generated.npy')):
        d = os.path.dirname(path)
        out[os.path.basename(d)] = (json.load(open(f'{d}/args.json')), torch.from_numpy(np.load(path)))
    return out


def config_key(args):
    """Config key (none, b0, c7 or ts, the keys of SETS labels) of a run from its args.json: b0 for an unconstrained
    standard-sampler run with the BorderZero projection, none for the other unconstrained runs."""
    if args['variant'] == 'none' and args.get('sampler') == 'standard' and args['projection'] == 'border_zero':
        return 'b0'
    return {'none': 'none', 'c7': 'c7', 'zero_mean': 'ts'}[args['variant']]


def run_key(args):
    """Key <season>_<config> of a run in the results, with the suffix _clean for a clean-mode run."""
    return f'{args["season"]}_{config_key(args)}' + ('_clean' if args.get('mode', 'noisy') == 'clean' else '')


def profile_axes(ax, top):
    """Stretched, inverted depth axis of a density profile panel, down to `top` m or the full depth."""
    if top:
        ax.set_ylim(top, 0)
    else:
        ax.set_ylim(stretch(depth[OCEAN_LEVELS].max()), 0)
        ax.set_yticks(stretch(DEPTH_TICKS))
        ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_density(results, data, rs, top=None):
    """Grid seasons x configs of the run set `rs` (SETS entry) of the horizontal-mean sigma_0 profiles of the members of
    the noisy and clean runs (MODES line styles), the source state and the data, save density_profiles[_top].pdf in
    the set dir."""
    n = len(rs['labels'])
    fig, axs = plt.subplots(2, n, figsize=(7.6 * n / 3, 6.6), sharex=True, sharey=True, squeeze=False)
    lv = OCEAN_LEVELS & (depth <= (top or np.inf) * 1.3)
    y = depth if top else stretch(depth)
    for i, (season, (colour, _)) in enumerate(SEASONS.items()):
        source = data['profile'][list(data['indices']).index(SOURCES[season])]
        for j, (key, label) in enumerate(rs['labels'].items()):
            ax = axs[i, j]
            for k, p in enumerate(data['profile']):
                ax.plot(p[lv], y[lv], color='0.45', lw=0.6, alpha=0.8, label='data, last year' if k == 0 else None)
            title = [f'{season}, {label}']
            for suffix, (ls, mode) in MODES.items():
                r = results.get(f'{season}_{key}{suffix}')
                if r is None:
                    continue
                for k, p in enumerate(r['profile']):
                    ax.plot(p[lv], y[lv], color=colour, lw=0.6, ls=ls, alpha=0.7,
                            label=f'generated {season}{mode} (n={len(r["profile"])})' if k == 0 else None)
                title.append(f'{mode.strip() + ": " if mode else ""}E {r["E"].mean():.2f}%, inv. cols '
                             f'{r["columns"].mean():.0f}%, mean inv. {r["mean_inv"].sum()}/{len(r["mean_inv"])}')
            ax.set_title('\n'.join(title if len(title) > 1 else [*title, '(not run yet)']), loc='left', fontsize=6.5)
            ax.plot(source[lv], y[lv], color='k', lw=1.5, label=f'source state {SOURCES[season]}')
            profile_axes(ax, top)
            if i == 1:
                ax.set_xlabel(SIGMA_LABEL)
        axs[i, 0].set_ylabel('depth [m]')
        axs[i, 0].legend(loc='lower left')
    fig.tight_layout()
    path = f'{rs["dir"]}/density_profiles{"_top" if top else ""}.pdf'
    save(fig, path, sources=[rs['dir'], *results])
    plt.close(fig)
    return path


def season_distance(indices, phase):
    """Circular distance in phase (YEAR) of each data state index to `phase`."""
    d = np.abs(np.asarray(indices) % YEAR - phase)
    return np.minimum(d, YEAR - d)


def plot_seasons(results, data, rs, field, xlabel, name, rows=(None, TOP_DEPTH), band=None, legend_loc='lower left'):
    """Grid rows x (data, configs of the run set `rs`, a SETS entry) of the profiles `field` of `results` and `data` (diagnostics), the global
    horizontal means if `band` is None, else the `field`_`band` profiles over that key of BANDS (named in the x label),
    with the summer and winter noisy runs overlaid: members, member means, source states and the last-year data in the
    background; the data column colours the data states within SEASON_HALFWIDTH phases of SUMMER / WINTER. `rows` are
    the depth ranges of the rows, each None for the full depth on the stretched axis or the bottom depth in m of a
    linear axis from the surface; the x limits are shared across a row and fitted to the lines drawn in it. The figure
    height scales with len(rows). The E in the titles is the global one whatever `band`. `legend_loc` places the
    legend of the data column. Save `name`.pdf in the set dir."""
    dark = {'summer': 'darkred', 'winter': 'darkblue'}
    if band is not None:
        field, xlabel = f'{field}_{band}', f'{xlabel}, {BANDS[band][1]}'
    prof = data[field]
    sources = {s: prof[list(data['indices']).index(SOURCES[s])] for s in SEASONS}
    near = {s: season_distance(data['indices'], p) <= SEASON_HALFWIDTH for s, (_, p) in SEASONS.items()}
    fig, axs = plt.subplots(len(rows), 1 + len(rs['labels']), figsize=(9 / 4 * (1 + len(rs['labels'])), 3.7 * len(rows)),
                            sharey='row', squeeze=False, layout='constrained')
    for i, top in enumerate(rows):
        lv = OCEAN_LEVELS & (depth <= (top or np.inf) * 1.3)
        y = depth if top else stretch(depth)
        ax = axs[i, 0]
        other = ~near['summer'] & ~near['winter']
        for k, p in enumerate(prof[other]):
            ax.plot(p[lv], y[lv], color='0.75', lw=0.6, label='other phases' if k == 0 else None)
        for season, (colour, phase) in SEASONS.items():
            for k, p in enumerate(prof[near[season]]):
                ax.plot(p[lv], y[lv], color=colour, lw=0.8,
                        label=f'phase {phase} +- {SEASON_HALFWIDTH}' if k == 0 else None)
        ax.set_title(f'data, last year (n={len(prof)})', loc='left')
        ax.legend(loc=legend_loc, fontsize=6)
        for j, (key, label) in enumerate(rs['labels'].items(), 1):
            ax = axs[i, j]
            for k, p in enumerate(prof):
                ax.plot(p[lv], y[lv], color='0.82', lw=0.5, zorder=1)
            title = [label]
            runs_ = {s: results[f'{s}_{key}'] for s in SEASONS if f'{s}_{key}' in results}
            for season, r in runs_.items():
                for p in r[field]:
                    ax.plot(p[lv], y[lv], color=SEASONS[season][0], lw=0.5, alpha=0.35)
            for season in SEASONS:
                r = runs_.get(season)
                if r is None:
                    title.append(f'{season}: not run yet')
                    continue
                ax.plot(r[field].mean(0)[lv], y[lv], color=SEASONS[season][0], lw=1.8, alpha=0.9)
                title.append(f'{season} n={len(r[field])}, E {r["E"].mean():.2f}%')
            for season in SEASONS:
                ax.plot(sources[season][lv], y[lv], color=dark[season], lw=1.2, ls='--')
            ax.set_title('\n'.join(title), loc='left', fontsize=6.5)
        xs = np.concatenate([line.get_xdata() for ax in axs[i] for line in ax.get_lines()])
        lo, hi = np.nanmin(xs), np.nanmax(xs)
        pad = 0.03 * (hi - lo)
        for ax in axs[i]:
            ax.set_xlim(lo - pad, hi + pad)
            profile_axes(ax, top)
            ax.set_xlabel(xlabel)
        axs[i, 0].set_ylabel('depth [m]' + ('' if top else ' (stretched)'))
    legend = {f'{s} member': dict(color=c, lw=1.2, alpha=0.6) for s, (c, _) in SEASONS.items()}
    legend |= {f'{s} member mean': dict(color=c, lw=1.8, alpha=0.9) for s, (c, _) in SEASONS.items()}
    legend |= {f'{s} source {SOURCES[s]}': dict(color=dark[s], lw=1.2, ls='--') for s in SEASONS}
    legend['data, last year'] = dict(color='0.82', lw=0.8)
    fig.legend([Line2D([], [], **kw) for kw in legend.values()], list(legend), loc='outside lower center', ncol=4,
               frameon=False)
    path = f'{rs["dir"]}/{name}.pdf'
    save(fig, path, sources=[rs['dir'], *results])
    plt.close(fig)
    return path


def plot_band(results, uncond, band, rs):
    """Per config of the run set `rs` (SETS entry), NH-band T anomaly profile (degC) over the first N_BAND levels of the
    members and mean of the noisy and clean runs (MODES line styles), the data source states and phase means and the
    unconditioned run `uncond` {config: (folder, anomaly (n, 18))}, save band_t_profile.pdf in the set dir."""
    d = depth[LEVELS[:N_BAND]]
    n = len(rs['labels'])
    fig, axs = plt.subplots(1, n, figsize=(7.2 * n / 3, 3.2), sharex=True, sharey=True)
    for ax, (key, label) in zip(axs, rs['labels'].items()):
        if key in uncond:
            u = uncond[key][1][:, :N_BAND]
            ax.fill_betweenx(d, u.min(0), u.max(0), color='0.8', label=f'{rs["uncond_label"]} n={len(u)} (min-max)')
            ax.plot(u.mean(0), d, color='0.4', lw=1.2, label=f'{rs["uncond_label"]} mean')
        for season, (colour, _) in SEASONS.items():
            ax.plot(band['phase'][season][:N_BAND], d, color=colour, lw=1.5, ls='--', label=f'data {season} phase mean')
            ax.plot(band['source'][season][:N_BAND], d, color=colour, lw=0.8, marker='o', ms=2.5,
                    label=f'data {season} source {SOURCES[season]}')
            for suffix, (ls, mode) in MODES.items():
                name = f'{season}_{key}{suffix}'
                if name in results:
                    a = results[name]['band'][:, :N_BAND]
                    for m in a:
                        ax.plot(m, d, color=colour, lw=0.4, ls=ls, alpha=0.5)
                    ax.plot(a.mean(0), d, color=colour, lw=2, ls=ls, label=f'{season}{mode} conditioned mean')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_ylim(d.max() * 1.05, 0)
        ax.set_title(label, loc='left')
        ax.set_xlabel('NH-band T anomaly [degC]')
        ax.grid(True, linestyle='--', alpha=0.5)
    axs[0].set_ylabel('depth [m]')
    legend = {}
    for ax in axs:
        legend.update({l: h for h, l in zip(*ax.get_legend_handles_labels())})
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(legend.values(), legend.keys(), loc='lower center', ncol=3, frameon=False)
    path = f'{rs["dir"]}/band_t_profile.pdf'
    save(fig, path, sources=[rs['dir'], T_CACHE, *(f for f, _ in uncond.values()), *results])
    plt.close(fig)
    return path


def plot_t_hist(results, states, rs):
    """Grid of the first N_HIST model T levels (rows) x (data, configs of the run set `rs`, a SETS entry) (columns) of
    the area-weighted histograms of T over the ocean cells of the latitude band HIST_BAND (halo columns excluded),
    summer and winter overlaid. The data column holds the source states SOURCES of `states` (load_states) at the model levels; a
    config column pools all the members of its noisy run in `results` (diagnostics 'T'). Within a row the HIST_BINS
    bins span the HIST_PERCENTILES weighted percentiles of all its histograms pooled, each histogram with total weight
    1, and the x axes are shared. Each histogram is the bin weight divided by the total weight of the histogram
    (including the values outside the bins) and by the bin width, so it integrates to the in-range fraction: the mass
    outside the row range is missing, not renormalised away. Summer is a filled step, winter a thicker unfilled step.
    Save t_hist_seasons.pdf in the set dir. Returns the path and {depth (m): (lo, hi)} of the row ranges."""
    w = band_weights(band=HIST_BAND, levels=LEVELS[:N_HIST])
    cells = w > 0
    columns = {'data': {s: states['toce'][list(states['indices']).index(SOURCES[s])][None, LEVELS[:N_HIST]]
                        for s in SEASONS}}
    columns |= {key: {s: results[f'{s}_{key}']['T'][:, :N_HIST] for s in SEASONS if f'{s}_{key}' in results}
                for key in rs['labels']}
    titles = ['data, source states'] + list(rs['labels'].values())
    fig, axs = plt.subplots(N_HIST, len(columns), figsize=(2.1 * len(columns), 1.7 * N_HIST), sharex='row',
                            sharey='row',
                            squeeze=False, layout='constrained')
    style = {'summer': dict(fill=True, color=SEASONS['summer'][0], alpha=0.4, lw=0),
             'winter': dict(fill=False, color='navy', lw=1.4)}
    ranges = {}
    for i in range(N_HIST):
        panels = {(key, s): (t[:, i][:, cells[i]].ravel(), np.tile(w[i][cells[i]], len(t)))
                  for key, by_season in columns.items() for s, t in by_season.items()}
        pooled = np.concatenate([v for v, _ in panels.values()])
        weights = np.concatenate([m / m.sum() for _, m in panels.values()])
        lo, hi = weighted_quantile(pooled, weights, np.array(HIST_PERCENTILES) / 100)
        edges = np.linspace(lo, hi, HIST_BINS + 1)
        d = depth[LEVELS[i]]
        ranges[round(float(d))] = (lo, hi)
        for j, key in enumerate(columns):
            ax = axs[i, j]
            for s in SEASONS:
                if (key, s) not in panels:
                    continue
                v, m = panels[key, s]
                h, _ = np.histogram(v, edges, weights=m)
                ax.stairs(h / m.sum() / np.diff(edges), edges, **style[s])
            ax.set_xlim(lo, hi)
            ax.grid(True, linestyle='--', alpha=0.5)
            if i == 0:
                ax.set_title(titles[j], loc='left', fontsize=7)
            if i == N_HIST - 1:
                ax.set_xlabel('T (degC)')
        axs[i, 0].set_ylabel(f'{d:.0f} m\ndensity')
    handles = [Patch(facecolor=style['summer']['color'], alpha=style['summer']['alpha'], label='summer'),
               Line2D([], [], color=style['winter']['color'], lw=style['winter']['lw'], label='winter')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle(f'T over the ocean of the band {HIST_BAND[0]}-{HIST_BAND[1]}N, area-weighted', fontsize=8)
    path = f'{rs["dir"]}/t_hist_seasons.pdf'
    save(fig, path, sources=[rs['dir'], CACHE1, *results])
    plt.close(fig)
    return path, ranges


def plot_sst(results, rs):
    """Per run, ordered by season then by the configs of the run set `rs` (SETS entry), max |SST - source SST| over the
    ocean of each member, save sst_check.pdf in the dir of `rs`."""
    order = {k: i for i, k in enumerate(rs['labels'])}
    results = dict(sorted(results.items(), key=lambda kv: (kv[0].split('_')[0], order[kv[0].split('_')[1]])))
    fig, ax = plt.subplots(figsize=(max(4.5, 0.75 * len(results)), 2.4))
    for i, (name, r) in enumerate(results.items()):
        ax.plot(np.full(len(r['sst_err']), i), r['sst_err'], 'o', ms=3, color='k')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(list(results), rotation=30, ha='right')
    ax.set_yscale('log')
    ax.set_ylim(1e-8, 1)
    ax.set_ylabel('max |SST - source|\non ocean [degC]')
    fig.tight_layout()
    path = f'{rs["dir"]}/sst_check.pdf'
    save(fig, path, sources=[rs['dir'], *results])
    plt.close(fig)
    return path


if __name__ == '__main__':
    rs = parse_set()
    states = load_states()
    xd = torch.stack([observer.normalizer({f: torch.from_numpy(states[f][k].copy()) for f in ['toce', 'soce', 'ssh']})
                      for k in range(len(states['indices']))]).float()
    data = diagnostics(xd) | {'indices': states['indices']}

    w = band_weights()
    t5 = np.load(T_CACHE)['toce']
    band5 = band_profile(t5, w)
    ref_mean = band5.mean(0)
    phase = INDICES % YEAR
    band = {'phase': {s: band5[phase == p].mean(0) - ref_mean for s, (_, p) in SEASONS.items()},
            'source': {s: band5[list(INDICES).index(SOURCES[s])] - ref_mean for s in SEASONS}}
    uncond = {}
    for key, pattern in rs['uncond'].items():
        folder = latest_run(pattern)
        if folder is not None:
            x = torch.from_numpy(np.load(f'{folder}/generated.npy'))
            uncond[key] = folder, band_profile(generated_toce(x, observer), w) - ref_mean

    sst_true = {s: states['toce'][list(states['indices']).index(SOURCES[s])][0] for s in SEASONS}
    results = {}
    for args, x in runs(rs['dir']).values():
        r = diagnostics(x)
        r['band'] = band_profile(r['T'], w) - ref_mean
        r['sst_err'] = np.abs(r['T'][:, 0] - sst_true[args['season']])[:, tmask[0]].max(1)
        r['args'] = args
        results[run_key(args)] = r

    paths = [plot_density(results, data, rs), plot_density(results, data, rs, TOP_DEPTH),
             plot_seasons(results, data, rs, 'profile', SIGMA_LABEL, 'density_profiles_seasons'),
             plot_seasons(results, data, rs, 'profile', SIGMA_LABEL, 'density_profiles_seasons_nh', SEASON_ROWS, 'nh',
                          'upper right'),
             plot_seasons(results, data, rs, 'T_profile', 'T [degC]', 'temperature_profiles_seasons', SEASON_ROWS,
                          'nh', 'lower right'),
             plot_band(results, uncond, band, rs), plot_sst(results, rs)]
    hist_path, hist_ranges = plot_t_hist(results, states, rs)
    paths.append(hist_path)
    print('\n'.join(paths))
    print('\n'.join(f'unconditioned {k}: {f}' for k, (f, _) in uncond.items()))
    print('t_hist_seasons x ranges: ' + ', '.join(f'{d} m [{lo:.2f}, {hi:.2f}]' for d, (lo, hi) in hist_ranges.items()))

    lev = [int(np.argmin(np.abs(depth[LEVELS] - z))) for z in BAND_DEPTHS]
    print(f'\ndata last year: E {data["E"].mean():.3f}% (max {data["E"].max():.3f}), inverted cols '
          f'{data["columns"].mean():.2f}%, mean-profile inversions {data["mean_inv"].sum()}/{len(data["mean_inv"])}')
    print(f'{"run":14s} {"n":>3s} {"min":>5s} {"E%":>7s} {"Emax%":>7s} {"cols%":>6s} {"mInv":>5s} {"SSTerr":>8s}  '
          + '  '.join(f'T{depth[LEVELS][l]:.0f}m' for l in lev))
    for s in SEASONS:
        print(f'{"data " + s + " src":14s}' + ' ' * 47 + '  '.join(f'{band["source"][s][l]:+6.2f}' for l in lev))
        print(f'{"data " + s + " ph":14s}' + ' ' * 47 + '  '.join(f'{band["phase"][s][l]:+6.2f}' for l in lev))
    for key, (_, u) in uncond.items():
        print(f'{"uncond " + key:14s}' + ' ' * 47 + '  '.join(f'{u[:, l].mean():+6.2f}' for l in lev))
    for name, r in results.items():
        print(f'{name:14s} {len(r["E"]):3d} {r["args"]["minutes"]:5.1f} {r["E"].mean():7.3f} {r["E"].max():7.3f} '
              f'{r["columns"].mean():6.2f} {r["mean_inv"].sum():2d}/{len(r["mean_inv"]):<2d} {r["sst_err"].max():8.2g}  '
              + '  '.join(f'{r["band"][:, l].mean():+6.2f}' for l in lev))
