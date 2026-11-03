"""
Phase 1 of the assimilation study (see plan.md): zonal sigma_0 sections (latitude x depth) of the states generated
conditioned on a fixed SST (1_generate_sst.py), in the style of paper/fig1_density_profile.py, for one run set
(1_profiles.SETS, --set centred or standard).

    python 1_density_sections.py --set standard

Grid: columns summer / winter, rows the source data state, then the runs of the configs of the set. Each panel is the
volume-weighted zonal mean of sigma_0 - 1000 (paper common.zonal_sections, last land level dropped), with as an inset
on the right the mean over the band latitudes of the batch-average section of the row (the source state for the first
row). The source states (phase1.SOURCES) go through the same normalisation round trip as the generated states
(phase1.normalised_state). The median member of a run is the member at the median of the RMSE of its section to the
source section (paper common.median_member).

Inputs:
    <set dir>/<run>/generated.npy   runs of 1_generate_sst.py, normalised (16, 37, 208, 64), one per season and config
    phase1.CACHE1                   last-year data states, source states SOURCES
Outputs in <set dir>/fig1 (set dir in phase1.RUN_SETS):
    fig1_phase1_SH.pdf       batch-average sections, inset band BANDS['SH'] (the paper's band)
    fig1_phase1_NH.pdf       batch-average sections, inset band BANDS['NH']
    fig1_phase1_median.pdf   median-member sections, inset band BANDS['SH']
Printed: per run, the median member and its RMSE to the source section.
"""
import importlib

import cmocean.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from phase1 import CACHE1, SOURCES, load_states, normalised_state
from seasonal import save

profiles = importlib.import_module('1_profiles')
observer, mask, volume = profiles.observer, profiles.mask, profiles.volume

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

SEASONS = ['summer', 'winter']
BANDS = {'SH': (-65, -45), 'NH': (15, 70)}
LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]

depth = mask.gdept_1d.values.squeeze()[:-1]
latitude = mask.gphit.isel(nav_lon=10).values
stretch = lambda d: np.asarray(d) ** DEPTH_POWER


def zonal_sections(x):
    """Volume-weighted zonal mean of sigma_0 - 1000 of a normalised batch x (b, 37, 208, 64), (b, 35, nav_lat), NaN
    on land, last (land) level dropped."""
    f = {k: v.detach().numpy().astype(np.float64) for k, v in observer.unnormalized(x).items() if k != 'ssh'}
    sig = profiles.sigma0(f)
    with np.errstate(invalid='ignore'):
        section = np.nansum(sig * volume, axis=-1) / volume.sum(-1)
    return section[:, :-1]


def band_profile(section, band):
    """Mean over the `band` latitudes of a (depth, lat) section."""
    return np.nanmean(section[:, (latitude >= band[0]) & (latitude < band[1])], axis=1)


def rmse(sections, reference):
    """Unweighted RMSE of each section (b, z, y) to `reference` (z, y), over the cells that are ocean in both."""
    return np.sqrt(np.nanmean((sections - reference[None]) ** 2, axis=(1, 2)))


def median_member(errors):
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


def plot_main_and_inset(ax, section, profile, title, band):
    """Contour a (depth, lat) section on the stretched depth axis and inset `profile`, the mean over the `band`
    latitudes."""
    y = stretch(depth)
    cf = ax.contourf(latitude, y, section, levels=LEVELS, cmap=cm.deep_r, extend='both')
    ax.contour(latitude, y, section, levels=LEVELS, colors='k', linewidths=0.6)
    for lat in band:
        ax.axvline(x=lat, color='gray', linestyle='dashed', linewidth=1)
    ax.set_title(title, loc='left')
    ax.set_ylim(stretch(DEPTH_MAX), 0)
    ax.set_yticks(stretch(DEPTH_TICKS))
    ax.set_yticklabels([str(d) for d in DEPTH_TICKS])

    inset_ax = inset_axes(ax, width='15%', height='100%', loc='right', borderpad=0)
    inset_ax.plot(profile, y, color='black', linewidth=2)
    inset_ax.set_facecolor('gray')
    inset_ax.patch.set_alpha(0.4)
    inset_ax.set_xlim(PROFILE_RANGE)
    inset_ax.set_ylim(ax.get_ylim())
    inset_ax.set_yticks(ax.get_yticks())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    return cf


def plot_grid(panels, band, path, sources):
    """Grid of `panels` [[(section, profile, title), ...] per row], one column per season, letters a), b), ... row by
    row, shared inverted colour bar, save as `path` with `sources` as provenance."""
    nrows, ncols = len(panels), len(panels[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.8, 1.55 * nrows), sharex=True, sharey=True, squeeze=False,
                            gridspec_kw={'hspace': 0.35, 'wspace': 0.05})
    for i, row in enumerate(panels):
        for j, (section, profile, title) in enumerate(row):
            cf = plot_main_and_inset(axs[i, j], section, profile, f'{chr(ord("a") + i * ncols + j)}) {title}', band)
        axs[i, 0].set_ylabel('depth [m]')
    for ax in axs[-1]:
        ax.set_xlabel('latitude [°N]')
    cbar = fig.colorbar(cf, ax=axs, label=r'$\sigma_0$ [kg m$^{-3}$ - 1000]', fraction=0.03 / ncols, pad=0.02 / ncols)
    cbar.ax.invert_yaxis()
    save(fig, path, sources=sources, bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    rs = profiles.parse_set()
    configs, out = rs['fig1_labels'], f'{rs["dir"]}/fig1'
    states = load_states()
    source = {s: zonal_sections(normalised_state(states, SOURCES[s], observer)[None])[0] for s in SEASONS}
    runs = {profiles.run_key(args): (args['name'], x) for args, x in profiles.runs(rs['dir']).values()}
    sections = {f'{s}_{c}': zonal_sections(runs[f'{s}_{c}'][1]) for s in SEASONS for c in configs}
    sources = [CACHE1, *(f'{rs["dir"]}/{runs[k][0]}/generated.npy' for k in sections)]

    medians = {}
    for k, sec in sections.items():
        errors = rmse(sec, source[k.split('_')[0]])
        medians[k] = median_member(errors)
        print(f'{k:12s} median member {medians[k]:2d} / {len(errors)}, RMSE {errors[medians[k]]:.3f} '
              f'(mean {errors.mean():.3f})')

    def grid(band, median):
        """Panels of the grid: the source state row, then one row per config, batch mean or median member (short config
        label in the titles)."""
        rows = [[(source[s], band_profile(source[s], band), f'{s.capitalize()}, data state {SOURCES[s]}')
                 for s in SEASONS]]
        for key, (label, short) in configs.items():
            row = []
            for s in SEASONS:
                sec = sections[f'{s}_{key}']
                mean = np.nanmean(sec, axis=0)
                row.append((sec[medians[f'{s}_{key}']] if median else mean, band_profile(mean, band),
                            f'{s.capitalize()}, {short + ", median" if median else label}'))
            rows.append(row)
        return rows

    paths = [plot_grid(grid(band, False), band, f'{out}/fig1_phase1_{b}.pdf', sources) for b, band in BANDS.items()]
    paths.append(plot_grid(grid(BANDS['SH'], True), BANDS['SH'], f'{out}/fig1_phase1_median.pdf', sources))
    print('\n'.join(paths))
