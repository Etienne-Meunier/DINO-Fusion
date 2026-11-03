#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 of the assimilation study (see plan.md): does the SSH of the training data carry a seasonal cycle?

Absolute SSH is dominated by the mean circulation (about -2.2 to 1.3 m), so a cm-scale seasonal signal is invisible in
snapshots (0_seasonal_snapshots.py). Here the last N_YEARS years of states are fitted, at each ocean point, with
a + b*t + c*cos(w*t) + d*sin(w*t), w = 2*pi/YEAR, t the state index, and the annual harmonic is shown against the
detrended anomaly. SST (toce level 0) is the reference for a clear seasonal signal. Phase is the state index modulo
YEAR; NH summer is near phase SUMMER, NH winter near phase WINTER. The halo columns 0 and -1 are dropped everywhere.

The numbers are printed for SSH and for SSH minus its area-weighted domain mean per state, which would remove a uniform
seasonal offset. The domain mean is conserved in this run (printed amplitude and residual std near 0), so the maps
are drawn for SSH only. Harmonics at the non-annual periods NOISE_PERIODS, fitted jointly with the trend and the
annual harmonic (seasonal.noise_amplitude), give the amplitude that the non-seasonal variability alone produces (noise
floor, both periods pooled).

Inputs:
    TrainingConfig().data_file  training tar, unnormalised NNNNN.{toce,ssh}.npy, time-ordered
    MASK_FILE                   1 degree DINO mesh mask (tmask, e1t, e2t, glamt, gphit)
    seasonal.py                 shared constants and helpers
Outputs in OUT:
    ssh_sst_<first>_<last>.npz  cache of the SSH and SST of the states
    ssh_domain_mean.pdf         area-weighted domain, NH and SH mean SSH time series, raw and detrended
    ssh_hovmoller.pdf           zonal-mean detrended SSH, latitude vs time at two colour ranges, and its mean
                                seasonal cycle (average over the years per phase)
    ssh_composite.pdf           detrended SSH, mean of phase SUMMER minus mean of phase WINTER states, two colour ranges
    ssh_harmonic.pdf            SSH annual harmonic amplitude, phase of maximum and fraction of detrended variance
    sst_seasonality.pdf         zonal-mean detrended SST Hovmoller and annual harmonic amplitude
Printed: domain-mean seasonal amplitudes, and the area-weighted median / 90th percentile of the seasonal amplitude of
SSH, SSH minus domain mean and SST, against the spatial std of the time mean, the non-seasonal residual std and the
noise floor.
"""
import os
import tarfile

import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from seasonal import (DATA_FILE, INDICES, INTERIOR, N_YEARS, NOISE_PERIODS, OUT, SUMMER, WINTER, YEAR, amplitude,
                      area_mean, composite, explained_fraction, harmonic_fit, load_mask, map_panel, noise_amplitude,
                      peak_phase, read, save, seasonal_cycle, weighted_quantile, zonal_mean)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7})

CACHE = f'{OUT}/ssh_sst_{INDICES[0]}_{INDICES[-1]}.npz'


def load_states():
    """SSH and SST (toce level 0) of the INDICES states, (t, y, x) with the halo columns, from CACHE or else the tar."""
    if not os.path.exists(CACHE):
        with tarfile.open(DATA_FILE) as tar:
            ssh = np.stack([read(tar, i, 'ssh') for i in INDICES])
            sst = np.stack([read(tar, i, 'toce')[0] for i in INDICES])
        os.makedirs(OUT, exist_ok=True)
        np.savez(CACHE, indices=INDICES, ssh=ssh, sst=sst)
    data = np.load(CACHE)
    assert np.array_equal(data['indices'], INDICES)
    return data['ssh'], data['sst']


def hovmoller_panel(ax, time, field, lat, label, vmax=None, marks=()):
    """Draw `field` (t, y) on `ax`, `time` (t) vs latitude `lat` (y), colour range symmetric about 0 on `vmax` (default
    the largest absolute value), with vertical lines at the times `marks` (pairs of position and line style)."""
    vmax = np.nanmax(np.abs(field)) if vmax is None else vmax
    pm = ax.pcolormesh(time, lat, field.T, cmap=cm.balance, vmin=-vmax, vmax=vmax, shading='nearest',
                       rasterized=True)
    for x, ls in marks:
        ax.axvline(x, color='k', lw=0.4, ls=ls)
    ax.set_ylabel('Latitude (deg)')
    plt.colorbar(pm, ax=ax, label=label, pad=0.02, extend='both' if np.nanmax(np.abs(field)) > vmax else 'neither')


def season_marks(n_years):
    """Vertical line positions (in years) and styles of the phases SUMMER (solid) and WINTER (dashed) of `n_years`."""
    return [(k + p / YEAR, ls) for k in range(n_years) for p, ls in ((SUMMER, '-'), (WINTER, '--'))]


def print_numbers(name, series, w, scale=100, unit='cm'):
    """Print, for the annual harmonic fit of `series` (t, y, x), the area-weighted median and 90th percentile of the
    seasonal amplitude, the spatial std of the time mean, the median / RMS of the per-point residual std and the
    median / 90th percentile of the explained variance fraction, over the ocean points of weights `w`, times `scale`
    in `unit`. The noise floor is the weighted median of noise_amplitude, both NOISE_PERIODS pooled."""
    fit = harmonic_fit(series)
    time_mean = series.mean(0)
    ocean = w > 0
    amp = scale * amplitude(fit)[ocean]
    res_std = scale * np.std(fit['residual'], axis=0)[ocean]
    wo = w[ocean]
    mean_std = scale * np.sqrt(np.average((time_mean[ocean] - np.average(time_mean[ocean], weights=wo)) ** 2, weights=wo))
    q50, q90 = weighted_quantile(amp, wo, [0.5, 0.9])
    r50, r90 = weighted_quantile(res_std, wo, [0.5, 0.9])
    frac = weighted_quantile(explained_fraction(fit)[ocean], wo, [0.5, 0.9])
    floor = scale * weighted_quantile(noise_amplitude(series)[:, ocean].ravel(), np.tile(wo, len(NOISE_PERIODS)), 0.5)
    print(f'{name}: seasonal amplitude median {q50:.2f} {unit}, p90 {q90:.2f} {unit}, max {amp.max():.2f} {unit} | '
          f'spatial std of time mean {mean_std:.2f} {unit} | residual std median {r50:.2f} {unit}, p90 {r90:.2f} '
          f'{unit}, RMS {np.sqrt(np.average(res_std ** 2, weights=wo)):.2f} {unit} | '
          f'explained fraction median {frac[0]:.2f}, p90 {frac[1]:.2f} | noise floor median amplitude {floor:.2f} '
          f'{unit} at periods {NOISE_PERIODS} states, pooled')


mask = load_mask()
ssh, sst = (np.where(mask['ocean'], f[:, :, INTERIOR], np.nan) for f in load_states())
w = mask['area'] * mask['ocean']
lat = mask['lat'][:, 0]
years = (INDICES - INDICES[0]) / YEAR

regions = {'domain': w, 'NH': w * (mask['lat'] > 0), 'SH': w * (mask['lat'] < 0)}
means = {k: area_mean(ssh, r) for k, r in regions.items()}
mean_fits = {k: harmonic_fit(m) for k, m in means.items()}
for k, f in mean_fits.items():
    print(f'{k} mean SSH: seasonal amplitude {100 * amplitude(f):.2f} cm, peak phase {peak_phase(f):.1f}, trend '
          f'{100 * f["b"] * YEAR:.3f} cm/yr, explained fraction {explained_fraction(f):.2f}, residual std '
          f'{100 * f["residual"].std():.2f} cm')

variants = {'SSH': ssh, 'SSH - domain mean': ssh - means['domain'][:, None, None]}
for k, v in variants.items():
    print_numbers(k, v, w)
print_numbers('SST', sst, w, scale=1, unit='degC')
fit = harmonic_fit(ssh)
sst_fit = harmonic_fit(sst)
ocean = w > 0

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True, layout='constrained')
for (k, m), color in zip(means.items(), ['k', 'C3', 'C0']):
    axs[0].plot(years, 100 * (m - m.mean()), color=color, lw=1, label=f'{k} (mean {100 * m.mean():.1f} cm)')
    axs[1].plot(years, 100 * mean_fits[k]['anomaly'], color=color, lw=1,
                label=f'{k}, harmonic amplitude {100 * amplitude(mean_fits[k]):.2f} cm')
    axs[1].plot(years, 100 * mean_fits[k]['harmonic'], color=color, lw=1.2, ls=':')
axs[0].set_ylabel(f'Mean SSH minus\nits {N_YEARS}-year mean (cm)')
axs[1].set_ylabel('Detrended (cm)\ndotted: annual harmonic')
axs[1].set_xlabel(f'Years since state {INDICES[0]}')
for ax in axs:
    ax.legend(fontsize=6, loc='upper left')
    for x, ls in season_marks(N_YEARS):
        ax.axvline(x, color='grey', lw=0.4, ls=ls)
fig.suptitle(f'Area-weighted mean SSH, states {INDICES[0]}-{INDICES[-1]} '
             f'(solid: phase {SUMMER}, dashed: phase {WINTER})')
save(fig, f'{OUT}/ssh_domain_mean.pdf', [CACHE], bbox_inches='tight')

zonal = 100 * zonal_mean(fit['anomaly'], w)
cycle = seasonal_cycle(zonal)
fig, axs = plt.subplots(3, 1, figsize=(6.5, 8.4), layout='constrained')
hovmoller_panel(axs[0], years, zonal, lat, 'cm', marks=season_marks(N_YEARS))
axs[0].set_title('Zonal mean of detrended SSH, full colour range')
vmax = np.nanmax(np.abs(cycle))
hovmoller_panel(axs[1], years, zonal, lat, 'cm', vmax=vmax, marks=season_marks(N_YEARS))
axs[1].set_title('Same, colour range of the mean seasonal cycle below')
for ax in axs[:2]:
    ax.set_xlabel(f'Years since state {INDICES[0]} (solid: phase {SUMMER}, dashed: phase {WINTER})')
hovmoller_panel(axs[2], np.arange(YEAR), cycle, lat, 'cm', vmax=vmax, marks=season_marks(1))
axs[2].set_xlabel(f'Phase (state index mod {YEAR})')
axs[2].set_title(f'Mean seasonal cycle: zonal mean of detrended SSH averaged over the {N_YEARS} years per phase')
save(fig, f'{OUT}/ssh_hovmoller.pdf', [CACHE], bbox_inches='tight', dpi=200)

diff = 100 * composite(fit['anomaly'])
fig, axs = plt.subplots(1, 2, figsize=(4.6, 5.2), sharex=True, sharey=True, layout='constrained')
map_panel(axs[0], diff, mask, cm.balance, 'cm', symmetric=True)
axs[0].set_title('full colour range')
clip = weighted_quantile(np.abs(diff[ocean]), w[ocean], 0.9)
map_panel(axs[1], diff, mask, cm.balance, 'cm', vmax=clip, symmetric=True)
axs[1].set_title('range clipped at the\n90th percentile of |diff|')
axs[0].set_ylabel('Latitude (deg)')
fig.suptitle(f'Detrended SSH, phase {SUMMER} minus phase {WINTER}\n(NH summer minus NH winter), mean over {N_YEARS} '
             f'years')
save(fig, f'{OUT}/ssh_composite.pdf', [CACHE], bbox_inches='tight', dpi=200)

amp = 100 * amplitude(fit)
fig, axs = plt.subplots(1, 3, figsize=(6.9, 5.2), sharex=True, sharey=True, layout='constrained')
map_panel(axs[0], amp, mask, cm.amp, 'cm', vmin=0, vmax=weighted_quantile(amp[ocean], w[ocean], 0.95))
map_panel(axs[1], peak_phase(fit), mask, cm.phase, f'phase (state index mod {YEAR})', vmin=0, vmax=YEAR)
map_panel(axs[2], explained_fraction(fit), mask, cm.dense, 'fraction', vmin=0, vmax=1)
axs[0].set_title('annual amplitude\n(range: 95th percentile)')
axs[1].set_title('phase of maximum')
axs[2].set_title('variance fraction of\nthe annual harmonic')
axs[0].set_ylabel('Latitude (deg)')
fig.suptitle(f'Annual harmonic of SSH, states {INDICES[0]}-{INDICES[-1]}')
save(fig, f'{OUT}/ssh_harmonic.pdf', [CACHE], bbox_inches='tight', dpi=200)

fig = plt.figure(figsize=(7.5, 4.2), layout='constrained')
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(gs[0])
hovmoller_panel(ax, years, zonal_mean(sst_fit['anomaly'], w), lat, 'degC', marks=season_marks(N_YEARS))
ax.set_xlabel(f'Years since state {INDICES[0]}\n(solid: phase {SUMMER}, dashed: phase {WINTER})')
ax.set_title('Zonal mean of detrended SST')
ax = fig.add_subplot(gs[1])
map_panel(ax, amplitude(sst_fit), mask, cm.amp, 'degC', vmin=0)
ax.set_title('SST annual amplitude')
save(fig, f'{OUT}/sst_seasonality.pdf', [CACHE], bbox_inches='tight', dpi=200)
plt.close('all')
