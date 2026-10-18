"""Figure 1: zonal sigma_0 sections as the Density tab of `machine/density_viewer.html`, with the batch-average profile
over the BAND latitudes as a black inset.

Main figure: median member by RMSE of each source in CONFIGS_MAIN, one per row. Appendix: every source in
CONFIGS_APPENDIX, one per row, median member on the left and batch-average section on the right."""
from common import *

import cmocean.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metrics import get_density_at_surface


CONFIGS_MAIN = ['data', 'none', 'zero_mean_border0', 'negative_grad_density_huber_border0']
CONFIGS_APPENDIX = ['data', 'none', 'none_border0', 'zero_mean_border0', 'mean_density_border0',
                    'negative_grad_density_huber_border0']
BAND = (-65, -45)
LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]

file_mask_LR = load_mask()


def compute_density(fields, file_mask_LR):
    """Potential density referenced at the surface, (batch, depth, nav_lat, nav_lon), of unnormalized 'toce', 'soce'."""
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce, toce, tmask = data.copy(), data.copy(), data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce[:] = np.asarray(fields['toce'])
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


def zonal_sections(x, file_mask_LR):
    """Volume-weighted zonal mean of sigma_0 - 1000 of a normalized batch x, (batch, depth - 1, nav_lat), last (land) level dropped."""
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    section = (compute_density(fields, file_mask_LR) * volume).sum('nav_lon') / volume.sum('nav_lon')
    return (section.transpose('batch', 'depth', 'nav_lat')[:, :-1] - 1000).values


def rmse(sections, reference):
    """Unweighted RMSE of each section (b, z, y) to `reference` (z, y), over the cells that are ocean in both."""
    return np.sqrt(np.nanmean((sections - reference[None]) ** 2, axis=(1, 2)))


def median_member(errors):
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


def band_profile(section):
    """Mean over the BAND latitudes of a (depth, lat) section."""
    return np.nanmean(section[:, (latitude >= BAND[0]) & (latitude < BAND[1])], axis=1)


stretch = lambda d: np.asarray(d) ** DEPTH_POWER
depth = file_mask_LR.gdept_1d.values[:-1]
latitude = file_mask_LR.gphit.isel(nav_lon=10).values

sections = {source: zonal_sections(sources[source], file_mask_LR) for source in dict.fromkeys(CONFIGS_MAIN + CONFIGS_APPENDIX)}
data_mean = np.nanmean(sections['data'], axis=0)
medians = {}
for source, s in sections.items():
    errors = rmse(s, data_mean)
    m = median_member(errors)
    medians[source] = m
    print(f'{source} : median member {m} / {len(errors)}, RMSE {errors[m]:.3f} (mean {errors.mean():.3f})')


def plot_main_and_inset(ax, section, profile, title):
    """Contour a (depth, lat) section on the stretched depth axis and inset `profile`, the mean over the `BAND` latitudes."""
    y = stretch(depth)
    cf = ax.contourf(latitude, y, section, levels=LEVELS, cmap=cm.deep_r, extend='both')
    ax.contour(latitude, y, section, levels=LEVELS, colors='k', linewidths=0.6)
    for lat in BAND:
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


def plot_grid(rows, name):
    """Plot one row per source and one column per entry of `rows` [(source, [(section, title), ...]), ...], save as `name`.pdf."""
    nrows, ncols = len(rows), len(rows[0][1])
    fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 3 * nrows), sharex=True, sharey=True, squeeze=False,
                            gridspec_kw={'hspace': 0.25, 'wspace': 0.05})
    for i, (source, panels) in enumerate(rows):
        profile = band_profile(np.nanmean(sections[source], axis=0))
        for j, (section, title) in enumerate(panels):
            cf = plot_main_and_inset(axs[i, j], section, profile, f'{chr(ord("a") + i * ncols + j)}) {title}')
        axs[i, 0].set_ylabel('depth [m]')
    for ax in axs[-1]:
        ax.set_xlabel('latitude [°N]')
    cbar = fig.colorbar(cf, ax=axs, label=r'$\sigma_0$ [kg m$^{-3}$ - 1000]', fraction=0.03 / ncols, pad=0.02 / ncols)
    cbar.ax.invert_yaxis()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f'{FIGURES}/{name}.pdf', bbox_inches='tight')
    return fig


plot_grid([(s, [(sections[s][medians[s]], NAMES[s])]) for s in CONFIGS_MAIN], 'fig1_density_profile')
plot_grid([(s, [(sections[s][medians[s]], f'{NAMES[s]}, median member'),
                (np.nanmean(sections[s], axis=0), f'{NAMES[s]}, batch average')]) for s in CONFIGS_APPENDIX],
          'fig1_density_profile_appendix')
plt.show()
