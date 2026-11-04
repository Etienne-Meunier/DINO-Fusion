"""Figure 2: the ocean state of each source as the A-priori tab of `machine/density_viewer.html`, maps drawn in the
style of the paper maps.

One row per source, batch average. Columns:
    SST : mean over the batch of the surface temperature.
    Clusters : most frequent k-means cluster of the (T, S) profile of each column over the batch
        (`common.ProfileClusters`, fit on the first N_FIT data states, every state assigned to the nearest centre).
    Clusters in T-S : centres of the k-means (dashed) and mean profile of the columns of the batch in each cluster
        (solid, members weighted by their column count), one marker per level, sigma_0 isopycnals behind.
    T-S diagram : volume fraction (e1t e2t e3t_0 of GRID) of each (T, S) bin averaged over the batch, re-entrant channel
        (ocean cells between the CHANNEL latitudes of `DinoConfiguration`) in solar_r and the rest of the basin in
        greys, log10 scale, each normalised by its own volume, sigma_0 isopycnals.
Both T-S panels have S on the x axis and T on the y axis; the isopycnals are TEOS-10 sigma_0 (`teos_sigma0`).

Main figure: sources of CONFIGS['fig2_main']. Appendix: sources of CONFIGS['fig2_appendix']."""
from common import *

import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

N_FIT = 8
CHUNK = 25  # members processed at once, the float64 profiles of a whole batch do not fit in memory
SST_LEVELS = np.linspace(0, 27, 28)
TS_LOG_RANGE = (-6., -1.)
TS_RANGE = {'S': (33.5, 37.5), 'T': (-2., 28.)}
CLUSTER_COLORS = ['#d7301f', '#fc8d59', '#4daf4a', '#377eb8', '#6a3d9a']
COLUMNS = ['SST', 'profile clusters', 'clusters in T–S', 'T–S diagram']


def fields(x):
    """Unnormalized toce, soce numpy arrays (member, z, y, x) of a normalized batch x, NaN on land."""
    ds = to_dataset(x)
    return {f: ds[f].values for f in FIELDS}


def batch_state(x, clusters):
    """
    Batch-average state of a normalized batch x:
        sst : mean SST (y, x), NaN on land
        cluster : most frequent cluster 1..N_CLUSTERS of `clusters` (y, x), NaN on land
        T, S : mean profile of the columns of each cluster (N_CLUSTERS, z - 1), members weighted by their column count
        ts : mean volume fraction in each (T, S) bin of the channel and of the rest of the basin (2, nT, nS)
    """
    tmask = GRID.mask.tmask.values.astype(bool)
    channel = tmask & ((GRID.mask.gphit >= CHANNEL[0]) & (GRID.mask.gphit <= CHANNEL[1])).values[None]
    volume = cell_volume(GRID).values
    sst, counts, ts, profile_sum, profile_weight = 0., 0., 0., 0., 0.
    for i in range(0, len(x), CHUNK):
        f = fields(x[i:i + CHUNK])
        sst = sst + f['toce'][:, 0].sum(axis=0)
        labels = clusters.labels(f)
        counts = counts + np.stack([(labels == k + 1).sum(axis=0) for k in range(N_CLUSTERS)])
        means, columns = clusters.mean_profiles(f, labels)
        w = np.where(np.isfinite(means), columns[:, :, None, None], 0)
        profile_sum = profile_sum + (np.nan_to_num(means) * w).sum(axis=0)
        profile_weight = profile_weight + w.sum(axis=0)
        ts = ts + np.stack([ts_histograms(f, channel, volume), ts_histograms(f, tmask & ~channel, volume)],
                           axis=1).sum(axis=0)
    profiles = np.where(profile_weight > 0, profile_sum / np.maximum(profile_weight, 1), np.nan)
    return {'sst': np.where(tmask[0], sst / len(x), np.nan),
            'cluster': np.where(tmask[0], counts.argmax(axis=0) + 1., np.nan),
            'T': profiles[:, 0], 'S': profiles[:, 1], 'ts': ts / len(x)}


def bottom_colorbar(fig, ax, mappable, label, offset, **kwargs):
    """Horizontal colorbar of the width of `ax`, its top `offset` [inch] below `ax`, positions read after a draw."""
    p, height = ax.get_position(), fig.get_size_inches()[1]
    cbar = fig.colorbar(mappable, cax=fig.add_axes([p.x0, p.y0 - (offset + 0.1) / height, p.width, 0.1 / height]),
                        orientation='horizontal', **kwargs)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)


def plot_grid(configs, states, centres):
    """Figure of one row per source of `configs` and the COLUMNS, from the `batch_state` of each source `states` and
    the cluster centres `centres` (N_CLUSTERS, 2 z)."""
    nz = GRID.mask.sizes['z_c']
    lon, lat = GRID.mask.glamt.values - 60, GRID.mask.gphit.values
    centres = {'T': centres[:, :nz - 1], 'S': centres[:, nz:2 * nz - 1]}
    s_grid, t_grid = np.meshgrid(np.linspace(*TS_RANGE['S'], 200), np.linspace(*TS_RANGE['T'], 200))
    sigma_grid = teos_sigma0(t_grid, s_grid)

    def isopycnals(ax):
        """sigma_0 isopycnals on (S, T) axes, labelled."""
        cs = ax.contour(s_grid, t_grid, sigma_grid, levels=SIGMA_LEVELS, colors='#999', linewidths=0.6)
        ax.clabel(cs, cs.levels[::2], fontsize=5, colors='#777')

    configs = [s for s in configs if s in sources]
    nrows, ncols = len(configs), len(COLUMNS)
    fig = plt.figure(figsize=(7.5, 2.3 * nrows))
    gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1, 1.5, 1.5], hspace=0.3, wspace=0.45)
    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    axs = np.empty((nrows, ncols), object)
    for i, s in enumerate(configs):
        state = states[s]
        axs[i, 0], axs[i, 1] = (fig.add_subplot(gs[i, j], projection=proj) for j in range(2))
        axs[i, 2], axs[i, 3] = (fig.add_subplot(gs[i, j]) for j in range(2, 4))
        cf = axs[i, 0].contourf(lon, lat, state['sst'], levels=SST_LEVELS, cmap='cmo.thermal', extend='both',
                                transform=ccrs.PlateCarree())
        cl = axs[i, 1].pcolormesh(lon, lat, state['cluster'], cmap=ListedColormap(CLUSTER_COLORS),
                                  norm=BoundaryNorm(np.arange(0.5, N_CLUSTERS + 1), N_CLUSTERS), shading='nearest',
                                  rasterized=True, transform=ccrs.PlateCarree())
        for j in range(2):
            setup_map(axs[i, j], j == 0)

        ax = axs[i, 2]
        isopycnals(ax)
        for k in range(N_CLUSTERS):
            ax.plot(centres['S'][k], centres['T'][k], '--o', color=CLUSTER_COLORS[k], linewidth=1, markersize=1.5)
            ax.plot(state['S'][k], state['T'][k], '-o', color=CLUSTER_COLORS[k], linewidth=1.5, markersize=2)
        ax.set_xlim(centres['S'].min() - 0.15, centres['S'].max() + 0.15)
        ax.set_ylim(centres['T'].min() - 1, centres['T'].max() + 1)

        ax = axs[i, 3]
        for r, cmap, alpha in [(1, 'Greys', 1.), (0, 'cmo.solar_r', 0.8)]:
            h = np.ma.masked_less_equal(state['ts'][r], 0)
            ax.pcolormesh(TS_S_EDGES, TS_T_EDGES, np.log10(h), cmap=cmap, vmin=TS_LOG_RANGE[0], vmax=TS_LOG_RANGE[1],
                          alpha=alpha, rasterized=True)
        isopycnals(ax)
        ax.set_xlim(TS_RANGE['S'])
        ax.set_ylim(TS_RANGE['T'])
        for j in range(2, 4):
            axs[i, j].set_ylabel('T [°C]')
            axs[i, j].tick_params(labelsize=7)
        for j, column in enumerate(COLUMNS):
            axs[i, j].set_title(f'{panel_letter(i * ncols + j)}) {column}')
        axs[i, 0].text(-0.45, 0.5, NAMES[s], transform=axs[i, 0].transAxes, rotation=90, ha='center', va='center',
                       fontsize=8)
    for j in range(2, 4):
        axs[-1, j].set_xlabel(r'S [g kg$^{-1}$]')

    fig.canvas.draw()
    bottom_colorbar(fig, axs[-1, 0], cf, 'SST [°C]', 0.1, ticks=SST_LEVELS[::9])
    bottom_colorbar(fig, axs[-1, 1], cl, r'cluster, by increasing $\sigma_0$', 0.1, ticks=np.arange(1, N_CLUSTERS + 1))
    p, height = axs[-1, 2].get_position(), fig.get_size_inches()[1]
    fig.legend([Line2D([], [], color='k', linestyle='--', marker='o', markersize=1.5, linewidth=1),
                Line2D([], [], color='k', marker='o', markersize=2, linewidth=1.5)],
               ['k-means centre (data)', 'mean profile of the cluster'], loc='upper center', fontsize=7, frameon=False,
               bbox_to_anchor=(p.x0 + p.width / 2, p.y0 - 0.3 / height))
    return fig


def compute():
    """Fit the profile clusters on the first N_FIT data states, batch-average state of every source of the main and
    appendix figures: {'centres': (N_CLUSTERS, 2 z), 'states': {source: `batch_state`}}."""
    clusters = ProfileClusters(fields(batch[:N_FIT]), GRID.mask.tmask.values.astype(bool))
    shown = [s for s in dict.fromkeys(CONFIGS['fig2_main'] + CONFIGS['fig2_appendix']) if s in sources]
    states = {s: batch_state(sources[s], clusters) for s in shown}
    return {'centres': clusters.centres, 'states': states}


def plot(computed):
    """Main and appendix figures of `computed` (`compute`): {file name, without extension: figure}."""
    return {name: plot_grid(CONFIGS[key], computed['states'], computed['centres'])
            for name, key in [('fig2_state', 'fig2_main'), ('fig2_state_appendix', 'fig2_appendix')]}


def main():
    """Compute, plot and save the figures in FIGURES."""
    for name, fig in plot(compute()).items():
        save(fig, f'{FIGURES}/{name}.pdf', bbox_inches='tight', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
