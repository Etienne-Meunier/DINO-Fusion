"""Figure 0: one data state against one unconstrained state, as panel A of the "No constraint" tab of
`machine/density_viewer.html`, drawn in the style of the paper maps (`Results/a_priori/spatial_structure.py`).

Four figures in the `fig0_unconstrained` subfolder of FIGURES, each a pair of maps, data then unconstrained: T at the
level nearest to 5 m, S at the level nearest to 136 m, and the level k-means clusters of the same states at each of these
levels, offset (basin mean of the level) removed."""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/machine/')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})
from build_noconstraint_viewer import FIELDS, LevelClusters, anomalies, level_means

SOURCES = ['data', 'none']
MEMBERS = {'data': 0, 'none': 0}
PANELS = [('toce', 5.), ('soce', 136.)]  # (field, depth [m]), the nearest level is used
N_FIT = 32  # data states the k-means is fit on, as in build_noconstraint_viewer.py
STYLE = {'toce': dict(levels=np.linspace(0, 27, 28), cmap=cm.thermal, label='conservative temperature [°C]'),
         'soce': dict(levels=np.linspace(32.2, 36.8, 24), cmap=cm.haline, label=r'absolute salinity [g kg$^{-1}$]')}
LEVEL_COLORS = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']

file_mask_LR = load_mask()
tmask = file_mask_LR.tmask.transpose('depth', 'nav_lat', 'nav_lon').values.astype(bool)
depth = file_mask_LR.gdept_1d.values
lon = file_mask_LR.glamt.values - 60
lat = file_mask_LR.gphit.values
levels = [int(np.abs(depth - d).argmin()) for _, d in PANELS]

with torch.no_grad():
    fields = {s: {k: v.cpu().numpy() for k, v in observer.unnormalized(sources[s]).items()} for s in SOURCES}

fit = {f: fields['data'][f][:N_FIT] for f in FIELDS}
fit_means = level_means(fit, tmask)
level_clusters = LevelClusters(anomalies(fit, tmask, fit_means), tmask, levels,
                               {f: m.mean(axis=0) for f, m in fit_means.items()})


def member_labels(source):
    """Level cluster labels of the member MEMBERS[source] with its own level means removed, uint8 (len(levels), y, x), 0 on land."""
    member = {f: fields[source][f][MEMBERS[source]][None] for f in FIELDS}
    anom = anomalies(member, tmask, level_means(member, tmask))
    return level_clusters.labels({f: anom[f][:, levels] for f in FIELDS})[0]


labels = {s: member_labels(s) for s in SOURCES}


def setup_map(ax, left_labels):
    """Coastlines, borders, extent and gridlines of the paper maps."""
    ax.coastlines(alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.left_labels = left_labels
    gl.ylabel_style = {'size': 6}


proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
cluster_cmap = ListedColormap(LEVEL_COLORS[:level_clusters.k])
cluster_norm = BoundaryNorm(np.arange(0.5, level_clusters.k + 1), level_clusters.k)
OUT = f'{FIGURES}/fig0_unconstrained'
os.makedirs(OUT, exist_ok=True)


def pair_figure():
    """Figure with one map per source of SOURCES side by side, and its axes."""
    fig, axs = plt.subplots(1, len(SOURCES), figsize=(1.75 * len(SOURCES), 2.6), subplot_kw={'projection': proj},
                            gridspec_kw={'wspace': 0.05})
    for s, ax in enumerate(axs):
        setup_map(ax, s == 0)
    return fig, axs


for p, (f, _) in enumerate(PANELS):
    kz = levels[p]
    fig, axs = pair_figure()
    for ax, source in zip(axs, SOURCES):
        field = np.where(tmask[kz], fields[source][f][MEMBERS[source], kz], np.nan)
        cf = ax.contourf(lon, lat, field, levels=STYLE[f]['levels'], cmap=STYLE[f]['cmap'], extend='both',
                         transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax=axs, orientation='horizontal', fraction=0.05, pad=0.04, aspect=30)
    cbar.set_label(f'{STYLE[f]["label"]} at {depth[kz]:.0f} m')
    cbar.set_ticks(STYLE[f]['levels'][::len(STYLE[f]['levels']) // 5])
    save(fig, f'{OUT}/{f[0].upper()}_{depth[kz]:.0f}m.pdf', bbox_inches='tight')

    fig, axs = pair_figure()
    for ax, source in zip(axs, SOURCES):
        lab = np.where(labels[source][p] > 0, labels[source][p], np.nan)
        ax.pcolormesh(lon, lat, lab, cmap=cluster_cmap, norm=cluster_norm, shading='nearest', rasterized=True,
                      transform=ccrs.PlateCarree())
    save(fig, f'{OUT}/clusters_{depth[kz]:.0f}m.pdf', bbox_inches='tight')
plt.show()
