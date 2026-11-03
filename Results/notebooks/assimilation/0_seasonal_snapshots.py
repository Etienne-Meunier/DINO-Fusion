#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 of the assimilation study (see plan.md): a first look at seasonality in the training data, SSH and the
temperature of the upper levels in one summer and one winter state of each of the last two years of the dataset.

The calendar phase of state 00000 is unknown, so the seasons are defined from the data: within each year (YEAR = 36
states, 10 days apart in a 360-day calendar), summer is the state of maximal area-weighted Northern Hemisphere mean SST
(toce level 0 over nav_lat > 0) and winter the minimal one. Summer and winter therefore mean NH summer and NH winter.
The halo columns 0 and -1 are dropped everywhere.

Inputs:
    TrainingConfig().data_file  training tar, unnormalised NNNNN.{toce,ssh}.npy, time-ordered
    MASK_FILE                   1 degree DINO mesh mask (tmask, e1t, e2t, glamt, gphit, gdept_1d)
    seasonal.py                 shared constants and helpers
Outputs in OUT, one pair per year (first state of the year in the name):
    ssh_<start>.pdf   absolute SSH of the summer and winter states side by side, one sequential colour range
    toce_<start>.pdf  T of the first N_LEVELS file levels, summer and winter side by side, one row and colour range per
                      level
"""
import tarfile

import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from seasonal import DATA_FILE, INTERIOR, N_STATES, OUT, YEAR, load_mask, read, save

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7})

YEARS = [N_STATES - 2 * YEAR, N_STATES - YEAR]
N_LEVELS = 7


def nh_sst(tar, indices, mask):
    """Area-weighted mean of toce level 0 over the Northern Hemisphere ocean (nav_lat > 0) of each state of `indices`,
    on the interior grid of `mask` (load_mask())."""
    w = mask['area'] * mask['ocean'] * (mask['lat'] > 0)
    return np.array([np.nansum(read(tar, i, 'toce')[0][:, INTERIOR] * w) / w.sum() for i in indices])


def masked(field, tmask):
    """`field` with land (tmask == 0) set to NaN, so it is drawn blank."""
    return np.where(tmask, field, np.nan)


def pair_panels(axs, fields, mask, cmap, label, titles=None):
    """Draw the two `fields` (y, x) on `axs` with one colour range and one colourbar `label`, titled `titles` if
    given."""
    vmin, vmax = np.nanmin(fields), np.nanmax(fields)
    for k, (ax, f) in enumerate(zip(axs, fields)):
        pm = ax.pcolormesh(mask['lon'], mask['lat'], f, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest',
                           rasterized=True)
        ax.set_aspect('equal')
        if titles is not None:
            ax.set_title(titles[k])
    axs[1].tick_params(labelleft=False)
    plt.colorbar(pm, ax=list(axs), label=label, shrink=0.9, pad=0.03)


mask = load_mask()
with tarfile.open(DATA_FILE) as tar:
    for start in YEARS:
        indices = np.arange(start, start + YEAR)
        sst = nh_sst(tar, indices, mask)
        seasons = {'summer': int(indices[np.argmax(sst)]), 'winter': int(indices[np.argmin(sst)])}
        print(f'year {start}-{start + YEAR - 1}: ' + ', '.join(
            f'{s} {i} (phase {i % YEAR}, NH SST {sst[i - start]:.3f} degC)' for s, i in seasons.items()))
        titles = [f'NH {s}\nstate {i}\nphase {i % YEAR}' for s, i in seasons.items()]

        ssh = np.stack([masked(read(tar, i, 'ssh')[:, INTERIOR], mask['ocean']) for i in seasons.values()])
        fig, axs = plt.subplots(1, 2, figsize=(4.6, 5.2), sharex=True, sharey=True, layout='constrained')
        pair_panels(axs, ssh, mask, cm.haline, 'SSH (m)', titles)
        axs[0].set_ylabel('Latitude (deg)')
        for ax in axs:
            ax.set_xlabel('Longitude (deg)')
        fig.suptitle(f'SSH, year of states {start}-{start + YEAR - 1}')
        save(fig, f'{OUT}/ssh_{start}.pdf', bbox_inches='tight', dpi=200)

        toce = np.stack([read(tar, i, 'toce')[:N_LEVELS, :, INTERIOR] for i in seasons.values()])
        fig, axs = plt.subplots(N_LEVELS, 2, figsize=(3.1, 2.3 * N_LEVELS), sharex=True, sharey=True,
                                layout='constrained')
        for k in range(N_LEVELS):
            pair_panels(axs[k], masked(toce[:, k], mask['tmask'][k]), mask, cm.thermal, 'T (degC)',
                        titles if k == 0 else None)
            axs[k, 0].set_ylabel(f'level {k}, {mask["depth"][k]:.0f} m\nLatitude (deg)')
        for ax in axs[-1]:
            ax.set_xlabel('Longitude (deg)')
        fig.suptitle(f'Temperature, year of states {start}-{start + YEAR - 1}')
        save(fig, f'{OUT}/toce_{start}.pdf', bbox_inches='tight', dpi=200)
        plt.close('all')
