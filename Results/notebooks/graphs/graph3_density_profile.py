from common import *

import cmocean.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metrics import get_density_at_surface

file_mask_LR = load_mask()


def compute_density(fields, file_mask_LR):
    """Potential density referenced at the surface of a batch.

    `fields` holds unnormalized 'toce' and 'soce' of shape (b, z, h, w). Returns an
    xarray DataArray with dims (batch, depth, nav_lat, nav_lon).
    """
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce = data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce = data.copy()
    toce[:] = np.asarray(fields['toce'])
    tmask = data.copy()
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


def zonal_mean(field, file_mask_LR):
    """Volume-weighted mean over longitude of a (batch, depth, nav_lat, nav_lon) field."""
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    return (field * volume).sum('nav_lon') / volume.sum('nav_lon')


# Batch-averaged zonal section of sigma_0, last (land) level dropped
sections = {}
for source, x in sources.items():
    fields = {k: v.detach().cpu() for k, v in observer.unnormalized(x).items()}
    sections[source] = (zonal_mean(compute_density(fields, file_mask_LR), file_mask_LR).mean('batch')[:-1] - 1000).values

depth = file_mask_LR.gdept_1d.values[:-1]
latitude = file_mask_LR.gphit.isel(nav_lon=10).values
levels = np.linspace(22., 28.5, 14)
band = (-65, -45)


def plot_main_and_inset(ax, section, title):
    """Contour a (depth, lat) section and inset its mean profile over the `band` latitudes."""
    cf = ax.contourf(latitude, depth, section, levels=levels, cmap=cm.deep_r)
    ax.contour(latitude, depth, section, levels=levels, colors='k', linewidths=0.75)
    for lat in band:
        ax.axvline(x=lat, color='gray', linestyle='dashed')
    ax.set_title(title, loc='left')
    ax.set_ylim(3500, 0)

    profile = np.nanmean(section[:, (latitude >= band[0]) & (latitude < band[1])], axis=1)
    inset_ax = inset_axes(ax, width='15%', height='100%', loc='right', borderpad=2)
    inset_ax.plot(profile, depth, color='black', linewidth=2)
    inset_ax.patch.set_alpha(0.75)
    inset_ax.set_ylim(ax.get_ylim())
    inset_ax.set_yticks(ax.get_yticks())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    return cf


ncols = 2
nrows = int(np.ceil(len(sections) / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), sharex=True, sharey=True, squeeze=False,
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.1})
for i, ax in enumerate(axs.flatten()):
    if i >= len(sections):
        ax.axis('off')
        continue
    source, section = list(sections.items())[i]
    cf = plot_main_and_inset(ax, section, f'{chr(ord("a") + i)}) {source}')

cbar = fig.colorbar(cf, ax=axs, label=r'Potential Density Anomaly $\sigma_0$ [kg m$^{-3}$ - 1000]')
cbar.ax.invert_yaxis()
for ax in axs[:, 0]:
    ax.set_ylabel('depth [m]')
for ax in axs[-1, :]:
    ax.set_xlabel('latitude [°N]')
plt.show()
