"""Figure 8: map of the temperature drift over the first days of the a posteriori runs of fig5 and fig7 (experiment
EXPERIMENT, 1/4° NEMO run started from the median member of each generation of the sweep PREFIX).

For each method of METHODS, side by side: at every water column, the thickness-weighted RMSE over depth of T to the
initial state (day 0, the restart the run starts from), with the time-varying e3t of each output,

    sqrt(sum_z e3t (T(x, y, z) - T_init(x, y, z))^2 / sum_z e3t),

averaged over the first FIRST_DAYS daily outputs, divided by its area-weighted mean over the domain (`xgcm.Grid.average`
of the 1/4° experiment over X and Y, given in the title of each panel). One logarithmic colour scale for all the panels,
centred on 1 (the domain mean). A method whose run is not collected is skipped.
"""
from common import *

import cartopy.crs as ccrs
from matplotlib.colors import LogNorm

EXPERIMENT = 'daily_2m'
METHODS = {'data': 'Data',
           'none': 'No constraint',
           DENSITY_RUN: rf'Increasing density, $\eta = {gens_args[DENSITY_RUN]["eta"]:g}$',
           'zero_mean_border0': 'T,S mean constraint'}
FIRST_DAYS = 10
RATIO_MAX = 10  # the colour scale spans [1 / RATIO_MAX, RATIO_MAX] times the domain mean


def column_rmse(folder, experiment, n_days):
    """Thickness-weighted RMSE over depth (y_c, x_c) of T to the restart the `experiment` run in `folder` starts from,
    averaged over its first `n_days` daily outputs, NaN on land columns, and its area-weighted domain mean."""
    paths = experiment_paths(experiment)
    exp = aposteriori_experiment(f'{folder}/{paths["results"]}', files=('T_3D',))
    T = exp.data['T_3D'].isel(t=slice(0, n_days))
    initial = restart_state(exp, f'{folder}/{paths["restart_1_4deg"]}')
    e3t = T.e3t.where(exp.mask.tmask == 1.0, 0.)
    rmse_map = np.sqrt(((T.toce - initial.toce) ** 2).weighted(e3t).mean('z_c')).mean('t').compute()
    return rmse_map.values, float(exp.grid.average(rmse_map, ['X', 'Y']))


def plot_maps(maps, means, titles, ncols):
    """Figure of the maps (j, i) over their domain mean, `titles` above them, on rows of `ncols` panels."""
    grid = aposteriori_experiment().domain
    lon, lat = grid.glamt.values - 60, grid.gphit.values
    nrows = -(-len(maps) // ncols)
    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 4.2 * nrows), subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(wspace=0.15), squeeze=False)
    for i, (ax, m, mean, title) in enumerate(zip(axes.flat, maps, means, titles)):
        pm = ax.pcolormesh(lon, lat, m / mean, cmap='cmo.balance', norm=LogNorm(1 / RATIO_MAX, RATIO_MAX),
                           shading='nearest', rasterized=True, transform=ccrs.PlateCarree())
        setup_map(ax, i % ncols == 0)
        ax.set_title(f'{title}\ndomain mean {mean:.2g} °C', loc='left')
    for ax in axes.flat[len(maps):]:
        ax.set_visible(False)
    ticks = [1 / RATIO_MAX, 1, RATIO_MAX]
    cb = fig.colorbar(pm, ax=axes, orientation='vertical', fraction=0.03, pad=0.02, ticks=ticks, extend='both',
                      label=f'RMSE over depth to day 0 / domain mean\n$T$, mean of days 1 to {FIRST_DAYS}')
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([f'{t:g}' for t in ticks])
    return fig


def compute(methods=METHODS):
    """Column RMSE of T to day 0 of each method of `methods` whose EXPERIMENT run is collected: {'folders', 'maps':
    {method: (y, x)}, 'means': {method: domain mean}}."""
    folders = collected_runs(methods, EXPERIMENT)
    maps, means = {}, {}
    for method, folder in folders.items():
        maps[method], means[method] = column_rmse(folder, EXPERIMENT, FIRST_DAYS)
        print(f'{method} : {folder}, domain mean {means[method]:.3f} °C')
    return {'folders': folders, 'maps': maps, 'means': means}


def plot(computed, methods=METHODS):
    """Maps of `computed` (`compute`) side by side over their domain mean, titled by `methods` (method -> title)."""
    maps, means = computed['maps'], computed['means']
    return plot_maps(list(maps.values()), list(means.values()), [methods[m] for m in maps], len(maps))


def main(methods=METHODS, out_name='fig8_drift_spatial'):
    """Compute and plot the maps of `methods`, save the figure as `{FIGURES}/{out_name}.pdf`."""
    computed = compute(methods)
    save(plot(computed, methods), f'{FIGURES}/{out_name}.pdf', sources=list(computed['folders'].values()),
         bbox_inches='tight', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
