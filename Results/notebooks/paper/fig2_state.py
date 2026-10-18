"""Figure 2: the ocean state of each source as the A-priori tab of `machine/density_viewer.html`, maps drawn in the
style of the paper maps (`Results/a_priori/spatial_structure.py`).

One row per source, batch average as the default view of the tab. Columns:
    SST : mean over the batch of the surface temperature.
    Clusters : most frequent k-means cluster of the (T, S) profile of each column over the batch. The k-means is fit on
        the first N_FIT data states (as the tab), every state of every source assigned to the nearest centre.
    Clusters in T-S : centres of the k-means (dashed) and mean profile of the columns of the batch in each cluster
        (solid, members weighted by their column count), one marker per level, sigma_0 isopycnals behind.
    T-S diagram : volume fraction of each (T, S) bin averaged over the batch, ACC (reentrant channel) in blue and the
        rest of the basin in red, each normalised by its own volume, sigma_0 isopycnals.
Both T-S panels have S on the x axis and T on the y axis.

Main figure: sources of CONFIGS_MAIN. Appendix: every source of `sources`."""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/machine/')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from metrics import get_density_at_surface
from build_density_viewer import N_CLUSTERS, N_DATA, TS_S_EDGES, TS_T_EDGES, ProfileClusters, acc_mask, ts_histograms

CONFIGS_MAIN = ['data', 'zero_mean_border0', 'negative_grad_density_huber_border0']
CONFIGS_APPENDIX = list(dict.fromkeys(['data', 'none', 'none_border0', 'zero_mean_border0', 'mean_density_border0',
                                       'negative_grad_density_huber_border0'] + list(sources)))
N_FIT = N_DATA
CHUNK = 25  # members processed at once, the float64 profiles of a whole batch do not fit in memory
SST_LEVELS = np.linspace(0, 27, 28)
SIGMA_LEVELS = np.arange(22., 28.51, 0.5)
TS_LOG_RANGE = (-6., -1.)
TS_RANGE = {'S': (33.5, 37.5), 'T': (-2., 28.)}
CLUSTER_COLORS = ['#d7301f', '#fc8d59', '#4daf4a', '#377eb8', '#6a3d9a']
COLUMNS = ['SST', 'profile clusters', 'clusters in T–S', 'T–S diagram']

file_mask_LR = load_mask()
tmask = file_mask_LR.tmask.transpose('depth', 'nav_lat', 'nav_lon').values.astype(bool)
volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0).transpose('depth', 'nav_lat', 'nav_lon').values
acc = acc_mask(tmask)
lon = file_mask_LR.glamt.values - 60
lat = file_mask_LR.gphit.values
nz = tmask.shape[0]


def unnormalized(x):
    """Unnormalized 'toce', 'soce' numpy arrays (b, z, y, x) of a normalized batch x."""
    with torch.no_grad():
        return {k: v.cpu().numpy() for k, v in observer.unnormalized(x).items()}


clusters = ProfileClusters({f: v[:N_FIT] for f, v in unnormalized(sources['data']).items()}, tmask)


def batch_state(x):
    """
    Batch-average a-priori state of a normalized batch x, as the batch-average view of the A-priori tab :
        sst : mean SST (y, x), NaN on land
        cluster : most frequent cluster 1..N_CLUSTERS (y, x), NaN on land
        T, S : mean profile of the columns of each cluster (N_CLUSTERS, z - 1), members weighted by their column count
        ts : mean volume fraction in each (T, S) bin of the ACC and of the rest of the basin (2, nT, nS)
    """
    sst, counts, ts, profile_sum, profile_weight = 0., 0., 0., 0., 0.
    n = len(x)
    for i in range(0, n, CHUNK):
        fields = unnormalized(x[i:i + CHUNK])
        sst = sst + fields['toce'][:, 0].sum(axis=0)
        labels = clusters.labels(fields)
        counts = counts + np.stack([(labels == k + 1).sum(axis=0) for k in range(N_CLUSTERS)])
        means, columns = clusters.mean_profiles(fields, labels)
        w = np.where(np.isfinite(means), columns[:, :, None, None], 0)
        profile_sum = profile_sum + (np.nan_to_num(means) * w).sum(axis=0)
        profile_weight = profile_weight + w.sum(axis=0)
        ts = ts + np.stack([ts_histograms(fields, acc, volume), ts_histograms(fields, tmask & ~acc, volume)], axis=1).sum(axis=0)
    profiles = np.where(profile_weight > 0, profile_sum / np.maximum(profile_weight, 1), np.nan)
    return {'sst': np.where(tmask[0], sst / n, np.nan),
            'cluster': np.where(tmask[0], counts.argmax(axis=0) + 1., np.nan),
            'T': profiles[:, 0], 'S': profiles[:, 1],
            'ts': ts / n}


states = {s: batch_state(sources[s]) for s in dict.fromkeys(CONFIGS_MAIN + CONFIGS_APPENDIX) if s in sources}
centres = {'T': clusters.centres[:, :nz - 1], 'S': clusters.centres[:, nz:2 * nz - 1]}
s_grid, t_grid = np.meshgrid(np.linspace(*TS_RANGE['S'], 200), np.linspace(*TS_RANGE['T'], 200))
sigma_grid = get_density_at_surface(t_grid, s_grid, np.ones_like(t_grid)) - 1000


def setup_map(ax, left_labels):
    """Coastlines, borders, extent and gridlines of the paper maps."""
    ax.coastlines(alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.left_labels = left_labels


def isopycnals(ax):
    """sigma_0 isopycnals on (S, T) axes, labelled."""
    cs = ax.contour(s_grid, t_grid, sigma_grid, levels=SIGMA_LEVELS, colors='#999', linewidths=0.6)
    ax.clabel(cs, cs.levels[::2], fontsize=7, colors='#777')


def plot_cluster_ts(ax, state):
    """Centres of the k-means (dashed) and mean cluster profiles of `state` (solid) in (S, T), axes on the span of the centres."""
    isopycnals(ax)
    for k in range(N_CLUSTERS):
        ax.plot(centres['S'][k], centres['T'][k], '--o', color=CLUSTER_COLORS[k], linewidth=1, markersize=1.5)
        ax.plot(state['S'][k], state['T'][k], '-o', color=CLUSTER_COLORS[k], linewidth=1.5, markersize=2)
    ax.set_xlim(centres['S'].min() - 0.15, centres['S'].max() + 0.15)
    ax.set_ylim(centres['T'].min() - 1, centres['T'].max() + 1)


def plot_ts_diagram(ax, state):
    """
    log10 volume fraction of the rest of the basin (red) and of the ACC (blue, on top) in (S, T), sigma_0 isopycnals.
    Returns the meshes of the ACC and of the rest.
    """
    meshes = {}
    for r, cmap, alpha in [(1, cm.amp, 1.), (0, cm.ice_r, 0.8)]:
        h = np.ma.masked_less_equal(state['ts'][r], 0)
        meshes[r] = ax.pcolormesh(TS_S_EDGES, TS_T_EDGES, np.log10(h), cmap=cmap, vmin=TS_LOG_RANGE[0],
                                  vmax=TS_LOG_RANGE[1], alpha=alpha, rasterized=True)
    isopycnals(ax)
    ax.set_xlim(TS_RANGE['S'])
    ax.set_ylim(TS_RANGE['T'])
    return meshes[0], meshes[1]


def bottom_colorbar(fig, ax, mappable, label, offset, **kwargs):
    """Horizontal colorbar of the width of `ax`, its top `offset` [inch] below `ax`, positions read after a draw."""
    p, height = ax.get_position(), fig.get_size_inches()[1]
    cbar = fig.colorbar(mappable, cax=fig.add_axes([p.x0, p.y0 - (offset + 0.1) / height, p.width, 0.1 / height]),
                        orientation='horizontal', **kwargs)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def plot_grid(configs, name):
    """Plot one row per source of `configs` and the COLUMNS of the A-priori tab, save as `name`.pdf."""
    for s in configs:
        if s not in sources:
            print(f'WARNING : {s} has no run in {PREFIX}, skipped')
    configs = [s for s in configs if s in sources]
    nrows, ncols = len(configs), len(COLUMNS)
    fig = plt.figure(figsize=(13, 4 * nrows))
    gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1, 1.5, 1.5], hspace=0.3, wspace=0.3)
    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    axs = np.empty((nrows, ncols), object)
    cluster_cmap = ListedColormap(CLUSTER_COLORS)
    cluster_norm = BoundaryNorm(np.arange(0.5, N_CLUSTERS + 1), N_CLUSTERS)
    for i, s in enumerate(configs):
        state = states[s]
        axs[i, 0], axs[i, 1] = (fig.add_subplot(gs[i, j], projection=proj) for j in range(2))
        axs[i, 2], axs[i, 3] = (fig.add_subplot(gs[i, j]) for j in range(2, 4))
        cf = axs[i, 0].contourf(lon, lat, state['sst'], levels=SST_LEVELS, cmap=cm.thermal, extend='both',
                                transform=ccrs.PlateCarree())
        cl = axs[i, 1].pcolormesh(lon, lat, state['cluster'], cmap=cluster_cmap, norm=cluster_norm, shading='nearest',
                                  rasterized=True, transform=ccrs.PlateCarree())
        for j in range(2):
            setup_map(axs[i, j], j == 0)
        plot_cluster_ts(axs[i, 2], state)
        acc_mesh, rest_mesh = plot_ts_diagram(axs[i, 3], state)
        for j in range(2, 4):
            axs[i, j].set_ylabel('T [°C]')
            axs[i, j].tick_params(labelsize=8)
        for j, column in enumerate(COLUMNS):
            axs[i, j].set_title(f'{chr(ord("a") + i * ncols + j)}) {column}', fontsize=10)
        axs[i, 0].text(-0.45, 0.5, NAMES[s], transform=axs[i, 0].transAxes, rotation=90, ha='center', va='center',
                       fontsize=11)
    for j in range(2, 4):
        axs[-1, j].set_xlabel(r'S [g kg$^{-1}$]')

    fig.canvas.draw()
    bottom_colorbar(fig, axs[-1, 0], cf, 'SST [°C]', 0.15, ticks=SST_LEVELS[::9])
    bottom_colorbar(fig, axs[-1, 1], cl, r'cluster, by increasing $\sigma_0$', 0.15, ticks=np.arange(1, N_CLUSTERS + 1))
    p, height = axs[-1, 2].get_position(), fig.get_size_inches()[1]
    fig.legend([Line2D([], [], color='k', linestyle='--', marker='o', markersize=1.5, linewidth=1),
                Line2D([], [], color='k', marker='o', markersize=2, linewidth=1.5)],
               ['k-means centre (data)', 'mean profile of the cluster'], loc='upper center', fontsize=8, frameon=False,
               bbox_to_anchor=(p.x0 + p.width / 2, p.y0 - 0.5 / height))
    bottom_colorbar(fig, axs[-1, 3], acc_mesh, r'log$_{10}$ volume fraction, ACC', 0.5)
    bottom_colorbar(fig, axs[-1, 3], rest_mesh, r'log$_{10}$ volume fraction, rest of the basin', 1.0)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f'{FIGURES}/{name}.pdf', bbox_inches='tight', dpi=200)
    return fig


plot_grid(CONFIGS_MAIN, 'fig2_state')
plot_grid(CONFIGS_APPENDIX, 'fig2_state_appendix')
plt.show()
