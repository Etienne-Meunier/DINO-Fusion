"""Figure 6: depth-space meridional overturning stream function of the a posteriori runs in the 1/4° NEMO model, one
panel per configuration of `NAMES` whose EXPERIMENT run is collected, averaged over the last YEARS years of the run.

    psi(z, y) = sum_{k' <= k} sum_x <v e3v> e1v  [Sv]

<v e3v> is `voce_e3v`, the time mean of the thickness-weighted meridional velocity written by XIOS, multiplied by e1v
as `Diagnostics.get_MOC` does before its transform to density classes; here the sum over x is cumulated over the
levels from the surface down instead, on the dinostics `NemoExperiment` of the 1/4° reference grid. psi is NaN on the
levels with no wet V point along the zonal line. The y axis is the nominal level depth gdept_1d: DINO is in
s-coordinates, so below ~1000 m the actual depth of a level is shallower where the bottom is shallower.
Also saved in the `aposteriori` subfolder of FIGURES: amoc, the methods of ROW_METHODS side by side. The averaging,
colour scale and panel layout are shared with `fig6_moc_density.py`.
"""
from common import *

YEARS = 1 / 12            # length of the average, taken at the end of each run (1 / 12: last month)
EXPERIMENT = 'daily_2m'
NCOLS = 4
ROW_METHODS = ['data', 'none', DENSITY_RUN, 'zero_mean_border0']
PSI_PERCENTILE = 99       # half range of the colour scale: this percentile of |psi| over all the runs
PSI_STEP = 4.             # contour interval [Sv], zero contour left out (noisy in weak-flow regions)
DEPTH_MAX, DEPTH_TICKS = 4000, [0, 100, 250, 500, 1000, 2000, 3000, 4000]


def last_outputs(data, years, days_per_year=360):
    """Number of the last outputs of `data` (an `exp.data` Dataset) within its last `years` years, and the length in
    years of that window, the whole run when it is shorter (360_day calendar of DINO)."""
    end, start = data['t_bounds'].values[-1, 1], data['t_bounds'].values[0, 0]
    window = min(years * days_per_year * 86400, (end - start).total_seconds())
    if window < years * days_per_year * 86400:
        print(f'WARNING : the run covers {window / 86400:g} days only, averaged over its full length')
    return int(sum((end - t).total_seconds() < window for t in data['t'].values)), window / (days_per_year * 86400)


def period(lengths, days_per_year=360):
    """The shortest of the averaging `lengths` [years], written in years, or in days below one year."""
    length = min(lengths)
    return f'{length:g} year{"s" if length > 1 else ""}' if length >= 1 else f'{length * days_per_year:g} days'


def mean_moc(results, years=YEARS):
    """Depth-space stream function [Sv] (z_c, y_f) of the V outputs of `results` averaged over the last `years` years,
    and the length of the average in years."""
    exp = aposteriori_experiment(results, files=('V_3D',))
    V = exp.data['V_3D']
    last, length = last_outputs(V, years)
    transport = (V.voce_e3v * exp.domain.e1v).isel(t=slice(-last, None)).mean('t')
    psi = transport.sum('x_c').cumsum('z_c') / 1e6
    return psi.where(exp.mask.vmask.any('x_c')).transpose('z_c', 'y_f').compute(), length


def scale(mocs):
    """Half range of the colour scale and contour levels of the stream functions `mocs`."""
    half_range = float(np.nanpercentile(np.abs(np.stack([m.values for m in mocs])), PSI_PERCENTILE))
    half_range = PSI_STEP * np.ceil(half_range / PSI_STEP)
    levels = np.arange(-half_range, half_range + PSI_STEP / 2, PSI_STEP)
    return half_range, levels[np.abs(levels) > PSI_STEP / 2]


def panel_grid(n):
    """Figure of `n` panels on rows of NCOLS sharing their axes, and its axes (rows, NCOLS): the slots past `n` are
    hidden, the last panel of each column gets the latitude labels."""
    nrows = -(-n // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(11, 1.9 * nrows + 0.5), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.3))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
        # The last panel of a column may sit above an empty slot of the last row.
        elif i + NCOLS >= n:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('latitude [°N]')
    return fig, axes


def panel(ax, psi, lat, depth, half_range, levels, title):
    """Pseudocolour and contours of the stream function `psi` (z, y) on the stretched depth axis."""
    pm = ax.pcolormesh(lat, stretch(depth), psi.values, cmap='cmo.balance', vmin=-half_range, vmax=half_range,
                       shading='nearest', rasterized=True)
    ax.contour(lat, stretch(depth), psi.values, levels=levels, colors='k', linewidths=0.3)
    ax.set_title(title, loc='left')
    return pm


def compute():
    """Depth-space stream function of the collected EXPERIMENT runs of NAMES, averaged over their last YEARS years:
    {'folders', 'mocs': {method: (z_c, y_f) [Sv]}, 'lengths': {method: years}, 'lat', 'depth'}."""
    grid = aposteriori_experiment()
    lat, depth = grid.domain.gphiv.mean('x_c').values, grid.mask.gdept_1d.values
    folders = collected_runs(NAMES, EXPERIMENT)
    mocs, lengths = {}, {}
    for method, folder in folders.items():
        results = f'{folder}/{experiment_paths(EXPERIMENT)["results"]}'
        mocs[method], lengths[method] = mean_moc(results)
        print(f'{method} : {results}, mean of {lengths[method]:g} years, '
              f'psi in [{float(mocs[method].min()):.1f}, {float(mocs[method].max()):.1f}] Sv')
    return {'folders': folders, 'mocs': mocs, 'lengths': lengths, 'lat': lat, 'depth': depth}


def plot(computed):
    """fig6_amoc and aposteriori/amoc of `computed` (`compute`): {file name in FIGURES, without extension: figure}."""
    mocs, lengths, lat, depth = computed['mocs'], computed['lengths'], computed['lat'], computed['depth']
    label = rf'$\psi_z$ [Sv], mean of the last {period(lengths.values())}'

    half_range, levels = scale(mocs.values())
    fig, axes = panel_grid(len(mocs))
    for i, (ax, (method, psi)) in enumerate(zip(axes.flat, mocs.items())):
        pm = panel(ax, psi, lat, depth, half_range, levels, NAMES[method])
        ax.title.set_fontsize(7)
        depth_axis(ax, DEPTH_MAX, DEPTH_TICKS)
        if i % NCOLS == 0:
            ax.set_ylabel('depth [m]')
    fig.colorbar(pm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.12, aspect=50, label=label)
    figs = {'fig6_amoc': fig}

    row = [m for m in ROW_METHODS if m in mocs]
    half_range, levels = scale(mocs[m] for m in row)
    fig, axs = plt.subplots(1, len(row), figsize=(2.75 * len(row), 1.9), sharey=True, squeeze=False,
                            gridspec_kw={'wspace': 0.05})
    for j, method in enumerate(row):
        pm = panel(axs[0, j], mocs[method], lat, depth, half_range, levels, f'{panel_letter(j)}) {NAMES[method]}')
        depth_axis(axs[0, j], DEPTH_MAX, DEPTH_TICKS)
        axs[0, j].set_xlabel('latitude [°N]')
    axs[0, 0].set_ylabel('depth [m]')
    fig.colorbar(pm, ax=axs, fraction=0.03, pad=0.02, label=label)
    figs['aposteriori/amoc'] = fig
    return figs


def main():
    """Compute, plot and save the figures in FIGURES."""
    computed = compute()
    figs, folders = plot(computed), computed['folders']
    save(figs['fig6_amoc'], f'{FIGURES}/fig6_amoc.pdf', sources=list(folders.values()), bbox_inches='tight')
    save(figs['aposteriori/amoc'], f'{FIGURES}/aposteriori/amoc.pdf',
         sources=[folders[m] for m in ROW_METHODS if m in folders], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
