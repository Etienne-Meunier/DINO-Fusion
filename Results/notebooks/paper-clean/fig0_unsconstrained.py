"""Figure 0: one data state against one unconstrained state, as panel A of the "No constraint" tab of
`machine/density_viewer.html`, drawn in the style of the paper maps.

Four figures in the `fig0_unconstrained` subfolder of FIGURES, each a pair of maps, data then unconstrained: T at the
level nearest to 5 m, S at the level nearest to 136 m, and the level k-means clusters (`common.LevelClusters`, fit
on the first N_FIT data states) of the same states at each of these levels, offset removed: the basin mean of the
level, area-weighted (`xgcm.Grid.average` of GRID over X and Y)."""
from common import *

import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap

SOURCES = ['data', 'none']
MEMBERS = {'data': 0, 'none': 0}
PANELS = [('toce', 5.), ('soce', 136.)]  # (field, depth [m]), the nearest level is used
N_FIT = 16
STYLE = {'toce': dict(levels=np.linspace(0, 27, 28), cmap='cmo.thermal', label='conservative temperature [°C]'),
         'soce': dict(levels=np.linspace(32.2, 36.8, 24), cmap='cmo.haline', label=r'absolute salinity [g kg$^{-1}$]')}
LEVEL_COLORS = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']


def anomalies(ds):
    """{field: (member, z, y, x)} of FIELDS of `ds` minus their area-weighted basin mean on each level, NaN on land, and
    {field: (member, z)} of these means."""
    means = {f: GRID.grid.average(ds[f], ['X', 'Y']) for f in FIELDS}
    return {f: (ds[f] - means[f]).values for f in FIELDS}, {f: means[f].values for f in FIELDS}


def compute():
    """Fit the level clusters on the first N_FIT data states and label the MEMBERS of SOURCES at the levels of PANELS:
    {'levels': level indices, 'depth': gdept_1d, 'maps': {source: {field: (z, y, x)}}, 'labels': {source:
    (len(levels), y, x)}, 'k': number of clusters}."""
    tmask = GRID.mask.tmask.values.astype(bool)
    depth = GRID.mask.gdept_1d.values
    levels = [int(np.abs(depth - d).argmin()) for _, d in PANELS]
    states = {s: to_dataset(sources[s][MEMBERS[s]:MEMBERS[s] + 1]) for s in SOURCES}
    fit, fit_means = anomalies(to_dataset(batch[:N_FIT]))
    clusters = LevelClusters(fit, tmask, levels, {f: m.mean(axis=0) for f, m in fit_means.items()})
    labels = {s: clusters.labels({f: a[:, levels] for f, a in anomalies(states[s])[0].items()})[0] for s in SOURCES}
    return {'levels': levels, 'depth': depth, 'labels': labels, 'k': clusters.k,
            'maps': {s: {f: states[s][f].values[0] for f in FIELDS} for s in SOURCES}}


def plot(computed):
    """T, S and cluster map pairs of `computed` (`compute`): {file name in the `fig0_unconstrained` subfolder, without
    extension: figure}."""
    levels, depth, labels, k = computed['levels'], computed['depth'], computed['labels'], computed['k']
    lon, lat = GRID.mask.glamt.values - 60, GRID.mask.gphit.values
    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    cluster_cmap = ListedColormap(LEVEL_COLORS[:k])
    cluster_norm = BoundaryNorm(np.arange(0.5, k + 1), k)

    def pair_figure():
        """Figure with one map per source of SOURCES side by side, and its axes."""
        fig, axs = plt.subplots(1, len(SOURCES), figsize=(1.75 * len(SOURCES), 2.6), subplot_kw={'projection': proj},
                                gridspec_kw={'wspace': 0.05})
        for s, ax in enumerate(axs):
            setup_map(ax, s == 0)
        return fig, axs

    figs = {}
    for p, (f, _) in enumerate(PANELS):
        kz = levels[p]
        fig, axs = pair_figure()
        for ax, source in zip(axs, SOURCES):
            cf = ax.contourf(lon, lat, computed['maps'][source][f][kz], levels=STYLE[f]['levels'],
                             cmap=STYLE[f]['cmap'], extend='both', transform=ccrs.PlateCarree())
        cbar = fig.colorbar(cf, ax=axs, orientation='horizontal', fraction=0.05, pad=0.04, aspect=30)
        cbar.set_label(f'{STYLE[f]["label"]} at {depth[kz]:.0f} m')
        cbar.set_ticks(STYLE[f]['levels'][::len(STYLE[f]['levels']) // 5])
        figs[f'{f[0].upper()}_{depth[kz]:.0f}m'] = fig

        fig, axs = pair_figure()
        for ax, source in zip(axs, SOURCES):
            ax.pcolormesh(lon, lat, np.where(labels[source][p] > 0, labels[source][p], np.nan), cmap=cluster_cmap,
                          norm=cluster_norm, shading='nearest', rasterized=True, transform=ccrs.PlateCarree())
        figs[f'clusters_{depth[kz]:.0f}m'] = fig
    return figs


def main():
    """Compute, plot and save the figures in the `fig0_unconstrained` subfolder of FIGURES."""
    for name, fig in plot(compute()).items():
        save(fig, f'{FIGURES}/fig0_unconstrained/{name}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
