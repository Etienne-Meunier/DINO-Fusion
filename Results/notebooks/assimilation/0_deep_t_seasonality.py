#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 of the assimilation study (see plan.md): how deep does the seasonal cycle of T reach in the training data?

Reference for phase 1, where states are generated conditioned on the SST of a summer or winter state: it shows the
seasonal signal of T below the surface and down to which depth it carries. Only the toce levels LEVELS seen by the
diffusion model are used, at their gdept_1d depths. Over the last N_YEARS years of states, each ocean point is fitted
with a + b*t + c*cos(w*t) + d*sin(w*t), w = 2*pi/YEAR, t the state index (seasonal.harmonic_fit). The noise floor of
a level is the area-weighted median amplitude of a harmonic at the non-annual periods NOISE_PERIODS, each fitted jointly
with the trend and the annual harmonic (seasonal.noise_amplitude), both periods pooled, over the NH band NH_BAND
(15-70N). The halo columns 0 and -1 are dropped everywhere.

Inputs:
    TrainingConfig().data_file  training tar, unnormalised NNNNN.toce.npy, time-ordered
    MASK_FILE                   1 degree DINO mesh mask (tmask, e1t, e2t, glamt, gphit, gdept_1d)
    seasonal.py                 shared constants and helpers, band_profile
Outputs in OUT:
    toce_levels_<first>_<last>.npz  cache of T at the model levels of the states, full grid, float32
    t_amplitude_depth.pdf           per level, NH-band median and p90 of the annual amplitude, the noise floor and
                                    noise p90, and the circular mean phase of maximum over the points above twice the
                                    floor
    t_composite_section.pdf         zonal mean of the detrended T composite, phase SUMMER minus phase WINTER, latitude
                                    vs depth at two colour ranges
    t_composite_maps.pdf            the same composite at MAP_COUNT levels picked from the amplitude profile
    t_band_profile.pdf              NH-band mean T profile minus its N_YEARS-year time mean (not detrended), per phase
                                    averaged over the years, min-max over the states shaded
Printed: per level, the amplitude median, p90, floor and noise p90 (p90 of the amplitude at NOISE_PERIODS), the band
area fraction above twice the floor and its phase; the first level where the median reaches the floor and the p90 the
noise p90; the map levels; per level, the band-mean profile anomaly of phases SUMMER and WINTER and its min / max.
"""
import os
import tarfile

import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from seasonal import (DATA_FILE, INDICES, INTERIOR, LEVELS, N_YEARS, NH_BAND, NOISE_PERIODS, OUT, SUMMER, WINTER, YEAR,
                      amplitude, band_profile, band_weights, composite, harmonic_fit, load_mask, map_panel,
                      noise_amplitude, peak_phase, read, save, seasonal_cycle, weighted_quantile, zonal_mean)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7})

CACHE = f'{OUT}/toce_levels_{INDICES[0]}_{INDICES[-1]}.npz'
MAP_COUNT = 5


def load_toce():
    """T of the INDICES states at the toce LEVELS, (t, len(LEVELS), y, x) float32 with the halo columns, NaN on land,
    from CACHE or else the tar."""
    if not os.path.exists(CACHE):
        with tarfile.open(DATA_FILE) as tar:
            toce = np.stack([read(tar, i, 'toce')[LEVELS] for i in INDICES]).astype(np.float32)
        os.makedirs(OUT, exist_ok=True)
        np.savez(CACHE, indices=INDICES, levels=LEVELS, toce=toce)
    data = np.load(CACHE)
    assert np.array_equal(data['indices'], INDICES) and np.array_equal(data['levels'], LEVELS)
    return data['toce']


def circular_stats(phase, weights):
    """Weighted circular mean (in states, 0 to YEAR) and circular std sqrt(-2 ln R) (in states) of the 1-D `phase`."""
    z = np.sum(weights * np.exp(2j * np.pi * phase / YEAR)) / weights.sum()
    return (np.angle(z) * YEAR / (2 * np.pi)) % YEAR, np.sqrt(-2 * np.log(np.abs(z))) * YEAR / (2 * np.pi)


def level_stats(series, w):
    """Annual harmonic statistics of one level over the band points: `series` (t, n) T at the n points of area
    weights `w` (n). Returns the weighted median and p90 of the amplitude, the floor (weighted median of
    noise_amplitude, both NOISE_PERIODS pooled) and the p90 of that noise amplitude, the area fraction of the points
    above twice the floor, and the circular mean and std of their phase of maximum (NaN if none)."""
    fit = harmonic_fit(series)
    amp = amplitude(fit)
    noise = noise_amplitude(series).ravel()
    floor, noise_p90 = weighted_quantile(noise, np.tile(w, len(NOISE_PERIODS)), [0.5, 0.9])
    median, p90 = weighted_quantile(amp, w, [0.5, 0.9])
    above = amp > 2 * floor
    phase, spread = circular_stats(peak_phase(fit)[above], w[above]) if above.any() else (np.nan, np.nan)
    return {'median': median, 'p90': p90, 'floor': floor, 'noise_p90': noise_p90, 'fraction': w[above].sum() / w.sum(),
            'phase': phase, 'spread': spread}


def first_below(values, floor):
    """Index of the first level where `values` <= `floor`, or None."""
    below = np.flatnonzero(values <= floor)
    return int(below[0]) if below.size else None


def log_depth_axis(ax, depth):
    """Log depth y axis, inverted, from the shallowest to the deepest of `depth`, with ticks at plain numbers."""
    ax.set_yscale('log')
    ax.set_ylim(depth[-1] * 1.15, depth[0] / 1.15)
    ticks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel('Depth (m)')


def depth_edges(depth):
    """Cell edges (len(depth) + 1) of `depth` on a log scale: geometric midpoints, extrapolated at both ends."""
    mid = np.sqrt(depth[1:] * depth[:-1])
    return np.concatenate([[depth[0] ** 2 / mid[0]], mid, [depth[-1] ** 2 / mid[-1]]])


mask_full = load_mask(interior=False)
mask = load_mask(interior=True)
depth = mask['depth'][LEVELS]
tmask = mask['tmask'][LEVELS]
w_full = band_weights(mask_full)
w = w_full[..., INTERIOR]
toce_full = load_toce()
toce = np.where(tmask, toce_full[..., INTERIOR], np.nan).astype(float)

stats = [level_stats(toce[:, k][:, w[k] > 0], w[k][w[k] > 0]) for k in range(len(LEVELS))]
stats = {key: np.array([s[key] for s in stats]) for key in stats[0]}
k_median, k_p90 = first_below(stats['median'], stats['floor']), first_below(stats['p90'], stats['noise_p90'])
print(f'NH band {NH_BAND[0]}-{NH_BAND[1]}N, states {INDICES[0]}-{INDICES[-1]}, annual amplitude (degC):')
for k, lev in enumerate(LEVELS):
    print(f'  level {lev:2d} {depth[k]:7.1f} m: median {stats["median"][k]:.4f}, p90 {stats["p90"][k]:.4f}, floor '
          f'{stats["floor"][k]:.4f}, noise p90 {stats["noise_p90"][k]:.4f}, ratio median/floor '
          f'{stats["median"][k] / stats["floor"][k]:.2f}, area fraction '
          f'> 2 floor {stats["fraction"][k]:.2f}, phase {stats["phase"][k]:.1f} +- {stats["spread"][k]:.1f}')
for name, k in (('median', k_median), ('p90', k_p90)):
    print(f'{name} reaches the noise {name} at '
          + ('no level' if k is None else f'level {LEVELS[k]} ({depth[k]:.1f} m)'))

k_floor = len(LEVELS) - 2 if k_median is None else min(k_median, len(LEVELS) - 2)
map_k = np.unique(np.round(np.linspace(0, k_floor, MAP_COUNT - 1)).astype(int).tolist() + [k_floor + 1])
print('map levels: ' + ', '.join(f'{LEVELS[k]} ({depth[k]:.0f} m)' for k in map_k))

fig, axs = plt.subplots(1, 2, figsize=(7.5, 4.6), sharey=True, layout='constrained')
ax = axs[0]
ax.plot(stats['median'], depth, 'o-', color='C0', ms=3, lw=1, label='median')
ax.plot(stats['p90'], depth, 's--', color='C3', ms=3, lw=1, label='90th percentile')
ax.plot(stats['floor'], depth, '^:', color='grey', ms=3, lw=1, label='noise floor')
ax.plot(2 * stats['floor'], depth, color='grey', lw=0.6, label='2 x noise floor')
ax.plot(stats['noise_p90'], depth, 'v--', color='0.6', ms=2.5, lw=0.8, label='noise 90th percentile')
for k, ls in ((k_median, '-'), (k_p90, '--')):
    if k is not None:
        ax.axhline(depth[k], color='k', lw=0.5, ls=ls)
ax.set_xscale('log')
ax.set_xlabel('Annual T amplitude (degC)')
ax.legend(fontsize=6, loc='lower right')
ax.set_title('Amplitude; horizontal lines: first level where\nthe median (solid) / p90 (dashed) reach\n'
             'the noise median (floor) / noise p90')
log_depth_axis(ax, depth)
ax = axs[1]
ok = np.isfinite(stats['phase'])
phase = np.unwrap(stats['phase'][ok], period=YEAR)
ax.errorbar(phase, depth[ok], xerr=stats['spread'][ok], fmt='o-', color='k', ms=3, lw=1, elinewidth=0.6, capsize=1.5)
for p, ls in ((SUMMER, '-'), (WINTER, '--')):
    for x in p + YEAR * np.arange(np.floor((phase.min() - p) / YEAR), np.ceil((phase.max() - p) / YEAR) + 1):
        ax.axvline(x, color='grey', lw=0.5, ls=ls)
for x, d, f in zip(phase, depth[ok], stats['fraction'][ok]):
    ax.annotate(f'{f:.2f}', (x, d), xytext=(4, 2), textcoords='offset points', fontsize=5, color='C0')
ax.set_xlabel(f'Phase of maximum (state index mod {YEAR}, unwrapped)')
ax.set_title(f'Phase of maximum, circular mean +- circular std\nover the points above 2 x floor (blue: their area '
             f'fraction)\nvertical lines: phase {SUMMER} (solid), {WINTER} (dashed)')
fig.suptitle(f'Annual harmonic of T at the model levels, area-weighted over {NH_BAND[0]}-{NH_BAND[1]}N, '
             f'states {INDICES[0]}-{INDICES[-1]}')
save(fig, f'{OUT}/t_amplitude_depth.pdf', [CACHE], bbox_inches='tight')

anomaly = harmonic_fit(toce)['anomaly']
diff = composite(anomaly)
del anomaly
w_ocean = mask['area'] * tmask
section = zonal_mean(np.nan_to_num(diff), w_ocean)
lat = mask['lat'][:, 0]
lat_edges = np.concatenate([[1.5 * lat[0] - 0.5 * lat[1]], (lat[1:] + lat[:-1]) / 2,
                            [1.5 * lat[-1] - 0.5 * lat[-2]]])
clip_k = k_floor if k_median is not None else len(LEVELS) // 2
clip = np.nanmax(np.abs(section[clip_k:]))
fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.4), sharex=True, layout='constrained')
for ax, vmax, title in ((axs[0], np.nanmax(np.abs(section)), 'full colour range'),
                        (axs[1], clip, f'colour range of the levels from {depth[clip_k]:.0f} m down')):
    pm = ax.pcolormesh(lat_edges, depth_edges(depth), section, cmap=cm.balance, vmin=-vmax, vmax=vmax, shading='flat',
                       rasterized=True)
    plt.colorbar(pm, ax=ax, label='degC', pad=0.02, extend='both' if np.nanmax(np.abs(section)) > vmax else 'neither')
    for b in NH_BAND:
        ax.axvline(b, color='k', lw=0.4, ls=':')
    log_depth_axis(ax, depth)
    ax.set_title(title)
axs[1].set_xlabel('Latitude (deg)')
fig.suptitle(f'Zonal mean of detrended T, phase {SUMMER} minus phase {WINTER} (NH summer minus NH winter), mean over '
             f'{N_YEARS} years\ndotted: NH band {NH_BAND[0]}-{NH_BAND[1]}N')
save(fig, f'{OUT}/t_composite_section.pdf', [CACHE], bbox_inches='tight', dpi=200)

fig, axs = plt.subplots(1, len(map_k), figsize=(2.4 * len(map_k), 4.6), sharex=True, sharey=True,
                        layout='constrained')
for ax, k in zip(axs, map_k):
    ocean = tmask[k]
    vmax = weighted_quantile(np.abs(diff[k][ocean]), mask['area'][ocean], 0.99)
    map_panel(ax, diff[k], dict(mask, ocean=ocean), cm.balance, 'degC', vmax=vmax, symmetric=True)
    ax.set_title(f'level {LEVELS[k]}, {depth[k]:.0f} m\nband median amplitude\n/ floor '
                 f'{stats["median"][k] / stats["floor"][k]:.1f}')
axs[0].set_ylabel('Latitude (deg)')
fig.suptitle(f'Detrended T, phase {SUMMER} minus phase {WINTER} (NH summer minus NH winter), mean over {N_YEARS} '
             f'years\none colour range per level, clipped at the 99th percentile of |diff|')
save(fig, f'{OUT}/t_composite_maps.pdf', [CACHE], bbox_inches='tight', dpi=200)

profile = band_profile(toce_full, w_full)
profile_anomaly = profile - profile.mean(0)
cycle = seasonal_cycle(profile_anomaly)
lo, hi = profile_anomaly.min(0), profile_anomaly.max(0)
print(f'band-mean T minus its {N_YEARS}-year mean (degC), per level: phase {SUMMER} / phase {WINTER} mean, min / max '
      'of the states')
for k, lev in enumerate(LEVELS):
    print(f'  level {lev:2d} {depth[k]:7.1f} m: {cycle[SUMMER, k]:+.4f} / {cycle[WINTER, k]:+.4f}, '
          f'{lo[k]:+.4f} / {hi[k]:+.4f}')
fig, axs = plt.subplots(1, 2, figsize=(7.5, 4.6), layout='constrained')
norm = Normalize(0, YEAR)
for ax in axs:
    ax.fill_betweenx(depth, lo, hi, color='0.85', lw=0, label=f'min-max of the {len(INDICES)} states')
    for p in range(YEAR):
        ax.plot(cycle[p], depth, color=cm.phase(norm(p)), lw=0.6)
    for p, ls in ((SUMMER, '-'), (WINTER, '--')):
        ax.plot(cycle[p], depth, color='k', lw=1.6, ls=ls, label=f'phase {p}')
    ax.axvline(0, color='k', lw=0.4)
    log_depth_axis(ax, depth)
    ax.set_xlabel(f'Band-mean T minus its {N_YEARS}-year time mean (degC)')
axs[0].legend(fontsize=6, loc='lower left')
axs[0].set_title('all levels')
deep = slice(k_floor, None)
xlim = 1.1 * np.abs(np.concatenate([lo[deep], hi[deep]])).max()
axs[1].set_xlim(-xlim, xlim)
axs[1].set_ylim(depth[-1] * 1.15, depth[k_floor] / 1.15)
axs[1].set_yticks([*axs[1].get_yticks(), depth[k_floor]],
                  [*(label.get_text() for label in axs[1].get_yticklabels()), f'{depth[k_floor]:.0f}'])
axs[1].set_title(f'zoom from {depth[k_floor]:.0f} m down (median amplitude at the floor)')
plt.colorbar(ScalarMappable(norm, cm.phase), ax=axs[1], label=f'phase (state index mod {YEAR}), mean over {N_YEARS} '
             'years', pad=0.02)
fig.suptitle(f'Area-mean T over {NH_BAND[0]}-{NH_BAND[1]}N per level (seasonal.band_profile), states '
             f'{INDICES[0]}-{INDICES[-1]}, not detrended')
save(fig, f'{OUT}/t_band_profile.pdf', [CACHE], bbox_inches='tight')
plt.close('all')
