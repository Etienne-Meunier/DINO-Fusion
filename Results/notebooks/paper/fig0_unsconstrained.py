"""Figure 0: one data state against one unconstrained state, as panel A of the "No constraint" tab of
`machine/density_viewer.html`, drawn in the style of the paper maps (`Results/a_priori/spatial_structure.py`).

Top row: T at the level nearest to 5 m and S at the level nearest to 136 m, data then unconstrained. Bottom row: the
level k-means clusters of the same states at the same levels, offset (basin mean of the level) removed."""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/machine/')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
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


proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
ncols = len(PANELS) * len(SOURCES)
fig, axs = plt.subplots(2, ncols, figsize=(3.5 * ncols, 10), subplot_kw={'projection': proj},
                        gridspec_kw={'hspace': 0.3, 'wspace': 0.05})
cluster_cmap = ListedColormap(LEVEL_COLORS[:level_clusters.k])
cluster_norm = BoundaryNorm(np.arange(0.5, level_clusters.k + 1), level_clusters.k)
for p, (f, _) in enumerate(PANELS):
    kz = levels[p]
    for s, source in enumerate(SOURCES):
        c = p * len(SOURCES) + s
        field = np.where(tmask[kz], fields[source][f][MEMBERS[source], kz], np.nan)
        cf = axs[0, c].contourf(lon, lat, field, levels=STYLE[f]['levels'], cmap=STYLE[f]['cmap'], extend='both',
                                transform=ccrs.PlateCarree())
        axs[0, c].set_title(f'{chr(ord("a") + c)}) {NAMES[source]}\n{f[0].upper()} at {depth[kz]:.0f} m', fontsize=10)
        setup_map(axs[0, c], c == 0)

        lab = np.where(labels[source][p] > 0, labels[source][p], np.nan)
        cl = axs[1, c].pcolormesh(lon, lat, lab, cmap=cluster_cmap, norm=cluster_norm, shading='nearest', rasterized=True,
                                  transform=ccrs.PlateCarree())
        axs[1, c].set_title(f'{chr(ord("a") + ncols + c)}) {NAMES[source]}\nclusters at {depth[kz]:.0f} m', fontsize=10)
        setup_map(axs[1, c], c == 0)
    cbar = fig.colorbar(cf, ax=axs[0, p * len(SOURCES):(p + 1) * len(SOURCES)], orientation='horizontal',
                        fraction=0.04, pad=0.02, aspect=30)
    cbar.set_label(STYLE[f]['label'])

cbar = fig.colorbar(cl, ax=axs[1, :], orientation='horizontal', fraction=0.04, pad=0.02, aspect=60,
                    ticks=np.arange(1, level_clusters.k + 1))
cbar.set_label(r'(T, S) cluster of the level, offset removed, by increasing $\sigma_0$ of its centre')

os.makedirs(FIGURES, exist_ok=True)
fig.savefig(f'{FIGURES}/fig0_unconstrained.pdf', bbox_inches='tight')
plt.show()
