"""Figure 5: a posteriori behaviour of the generated states in the 1/4° NEMO model, recomputed from the NEMO outputs of
the runs of `notebooks/a_posteriori/` (experiment EXPERIMENT, run started from the median member of each generation).

Left, one row per method of METHODS:
    zonal mean sigma_0 after 2 months (mean of the last daily output), with its BAND profile as inset (as fig4),
    its difference to the initial state (day 0, the converted generated state),
    its difference to the reference density, the mean zonal sigma_0 section of the data batch (1° grid, the 1/4°
    section interpolated onto its latitudes, both grids share their levels).
Right, over time: a) zonal transport through the meridional line of U points of column ACC_COLUMN [Sv], b) RMSE of the
zonal sigma_0 section to the reference density (grey band: RMSE of each data state to it, min-max), c) empty.
Dashed: the 1/4° spin-up restart. A method whose run is not collected yet is skipped.
"""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

import cmocean.cm as cm
from nemo_cookbook import NEMODataTree
from settings import COLLECT, EXP, REFERENCES_REL, RESTART_1_4DEG, aposteriori_dirs, experiment_paths
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

EXPERIMENT = 'daily_2m'
METHODS = {'data': 'Data',
           'none': 'No constraint',
           'negative_grad_density_huber_border0': r'Increasing density, $\eta = 10$',
           'zero_mean_border0': 'T,S mean constraint'}
COLORS = ['k', 'C0', 'C1', 'C2']
REFERENCE = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/restart25_arch'
DOMAIN = f'{REFERENCE}/domain_cfg_out.nc'
ACC_COLUMN = 1  # first interior column
BAND = (-65, -45)
LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
DIFF_PERCENTILE = 99  # half range of each difference colour scale: this percentile of |difference| over all the runs
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000]
stretch = lambda d: np.asarray(d) ** DEPTH_POWER


def output_tree(results):
    """NEMODataTree of the daily T and U outputs of a run, e3t time-varying (written by XIOS), times in seconds."""
    paths = {'parent': {'domain': DOMAIN, 'gridT': f'{results}/{EXP}_1d_grid_T_3D.nc',
                        'gridU': f'{results}/{EXP}_1d_grid_U_3D.nc'}}
    return NEMODataTree.from_paths(paths, iperio=True, decode_times=False)


def restart_tree(path):
    """NEMODataTree of the now fields of a NEMO restart, with e3t = e3t_0 and uoce_e3u = un * e3u_0."""
    r = xr.open_dataset(path)
    domain = xr.open_dataset(DOMAIN)
    grid_t = r[['tn', 'sn']].rename(tn='toce', sn='soce', nav_lev='deptht')
    grid_u = (r.un * domain.e3u_0.values).rename('uoce_e3u').to_dataset().rename(nav_lev='depthu')
    return NEMODataTree.from_datasets({'parent': {'domain': domain, 'gridT': grid_t, 'gridU': grid_u}},
                                      iperio=True, linssh=True)


def zonal_sigma(nemo):
    """Volume-weighted zonal mean of sigma_0 - 1000 (time, k, j) of the T grid of `nemo`, NaN on land rows."""
    sigma = get_density_at_surface(nemo['gridT/toce'].data, nemo['gridT/soce'].data, nemo['gridT/tmask'].data) - 1000
    vol = nemo.cell_volume('gridT').data
    return ((sigma * vol).sum('i') / vol.sum('i')).values


def transport(nemo):
    """Zonal transport [Sv] (time) through column ACC_COLUMN of the thickness-weighted zonal velocity of `nemo`."""
    return nemo['gridU/uoce_e3u'].integral(dims=['j']).data.isel(i=ACC_COLUMN).sum('k').values / 1e6


def restart(path):
    """Zonal sigma_0 section (k, j) and transport of a NEMO restart."""
    nemo = restart_tree(path)
    return zonal_sigma(nemo)[0], float(transport(nemo)[0])


def run(folder):
    """Days from the start, zonal sigma_0 sections (frames, z, y) and transports of the EXPERIMENT run in `folder`.

    Frame 0 is the restart the run starts from (instantaneous), frame i >= 1 the i-th daily mean.
    """
    nemo = output_tree(f'{folder}/{experiment_paths(EXPERIMENT)["results"]}')
    grid = nemo['gridT']
    days = (grid['time_centered'].values - grid['time_counter_bounds'].values[0, 0]) / 86400
    section0, transport0 = restart(f'{folder}/{RESTART_1_4DEG}')
    return np.r_[0., days], np.concatenate([section0[None], zonal_sigma(nemo)]), np.r_[transport0, transport(nemo)]


def to_lat(section, lat, lat_to):
    """Section (z, y) on `lat` linearly interpolated onto `lat_to`, NaN where a neighbour is NaN."""
    def interp(a):
        finite = np.isfinite(a)
        v = np.interp(lat_to, lat, np.where(finite, a, 0.))
        w = np.interp(lat_to, lat, finite.astype(float))
        return np.where(w > 1 - 1e-6, v, np.nan)
    return np.stack([interp(row) for row in section])


def plot_section(ax, section, lat, depth, title):
    """Contour a (depth, lat) sigma_0 section on the stretched depth axis, with its BAND profile as a black inset."""
    y = stretch(depth)
    cf = ax.contourf(lat, y, section, levels=LEVELS, cmap=cm.deep_r, extend='both')
    ax.contour(lat, y, section, levels=LEVELS, colors='k', linewidths=0.4)
    for b in BAND:
        ax.axvline(x=b, color='gray', linestyle='dashed', linewidth=1)
    ax.set_title(title, loc='left')
    # Axes.inset_axes, not the axes_grid1 one: its locator misplaces the rasterized meshes with bbox_inches='tight'.
    inset_ax = ax.inset_axes([0.85, 0, 0.15, 1])
    band = (lat >= BAND[0]) & (lat < BAND[1])
    with np.errstate(invalid='ignore'):
        profile = np.nanmean(section[:, band], axis=1)
    inset_ax.plot(profile, y, color='black', linewidth=1.5)
    inset_ax.set_facecolor('gray')
    inset_ax.patch.set_alpha(0.4)
    inset_ax.set_xlim(PROFILE_RANGE)
    inset_ax.set_ylim(stretch(DEPTH_MAX), 0)
    inset_ax.set_yticks(stretch(DEPTH_TICKS))
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    return cf


def plot_diff(ax, diff, lat, depth, half_range, title):
    """Pseudocolour of a (depth, lat) difference on the stretched depth axis, symmetric scale of half range `half_range`."""
    pm = ax.pcolormesh(lat, stretch(depth), diff, cmap=cm.balance, vmin=-half_range, vmax=half_range,
                       shading='nearest', rasterized=True)
    ax.set_title(title, loc='left')
    return pm


file_mask_LR = load_mask()
lat_1 = file_mask_LR.gphit.isel(nav_lon=10).values
data_sections = zonal_sections(batch, file_mask_LR)
with np.errstate(invalid='ignore'):
    reference = np.nanmean(data_sections, axis=0)
data_rmse = rmse(data_sections, reference)

spinup_path = sorted(glob(f'{REFERENCE}/{EXP}_*_restart.nc'))[-1]
spinup = restart_tree(spinup_path)
lat, depth = spinup['gridT']['gphit'].isel(i=ACC_COLUMN + 1).values, spinup['gridT']['deptht'].values
nz = reference.shape[0]
depth_1 = depth[:nz]
rmse_to_reference = lambda section: float(rmse(to_lat(section[:nz], lat, lat_1)[None], reference)[0])
spinup_section, spinup_transport = zonal_sigma(spinup)[0], float(transport(spinup)[0])
spinup_rmse = rmse_to_reference(spinup_section)
print(f'spin-up {spinup_path} : transport {spinup_transport:.1f} Sv, RMSE {spinup_rmse:.3f}')

folders = aposteriori_dirs(constraints=METHODS)
runs = {}
for method in METHODS:
    folder = folders.get(method)
    if folder is None or not os.path.exists(f'{folder}/{experiment_paths(EXPERIMENT)["results"]}/{COLLECT}'):
        print(f'WARNING : no collected {EXPERIMENT} run for {method}, skipped')
        continue
    print(f'{method} : {folder}')
    days, sections, transports = run(folder)
    runs[method] = dict(days=days, sections=sections, transport=transports,
                        rmse=np.array([rmse_to_reference(s) for s in sections]))

fig = plt.figure(figsize=(11, 1.75 * len(runs) + 0.4))
outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.15], wspace=0.2)
left = outer[0].subgridspec(len(runs) + 2, 3, height_ratios=[1] * len(runs) + [0.12, 0.06], wspace=0.08, hspace=0.25)
right = outer[1].subgridspec(3, 1, hspace=0.7)
axes = np.array([[fig.add_subplot(left[i, j]) for j in range(3)] for i in range(len(runs))])
COLUMNS = [rf'$\sigma_0$, day {next(iter(runs.values()))["days"][-1]:g}', 'minus day 0', 'minus reference density']

vs_day0 = {m: r['sections'][-1] - r['sections'][0] for m, r in runs.items()}
vs_reference = {m: to_lat(r['sections'][-1][:nz], lat, lat_1) - reference for m, r in runs.items()}
half_range = lambda diffs: float(np.nanpercentile(np.abs(np.stack(list(diffs.values()))), DIFF_PERCENTILE))
range_day0, range_reference = half_range(vs_day0), half_range(vs_reference)
print(f'difference half ranges : minus day 0 {range_day0:.3f}, minus reference {range_reference:.3f} kg m-3')

for i, (method, r) in enumerate(runs.items()):
    final = r['sections'][-1]
    titles = COLUMNS if i == 0 else [''] * 3
    cf = plot_section(axes[i, 0], final, lat, depth, titles[0])
    pm_day0 = plot_diff(axes[i, 1], vs_day0[method], lat, depth, range_day0, titles[1])
    pm_ref = plot_diff(axes[i, 2], vs_reference[method], lat_1, depth_1, range_reference, titles[2])
    axes[i, 0].set_ylabel(f'{METHODS[method]}\ndepth [m]')
    print(f'{method} : day {r["days"][-1]:g}, RMSE to reference {r["rmse"][-1]:.3f}, transport {r["transport"][-1]:.1f} Sv')

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.set_ylim(stretch(DEPTH_MAX), 0)
        ax.set_yticks(stretch(DEPTH_TICKS))
        ax.set_yticklabels([str(d) for d in DEPTH_TICKS] if j == 0 else [])
        if i < len(axes) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('latitude [°N]')
for j, (mappable, label) in enumerate([(cf, r'$\sigma_0$ [kg m$^{-3}$ - 1000]'),
                                       (pm_day0, r'$\Delta\sigma_0$ [kg m$^{-3}$]'),
                                       (pm_ref, r'$\Delta\sigma_0$ [kg m$^{-3}$]')]):
    cbar = fig.colorbar(mappable, cax=fig.add_subplot(left[-1, j]), orientation='horizontal', label=label)
    if j == 0:
        cbar.ax.invert_xaxis()

series_axes = [fig.add_subplot(right[k]) for k in range(3)]
for (k, spinup, title, ylabel), ax in zip(
        [('transport', spinup_transport, 'a) Zonal transport through the channel', 'transport [Sv]'),
         ('rmse', spinup_rmse, 'b) RMSE to the reference density', r'RMSE $\sigma_0$ [kg m$^{-3}$]')], series_axes):
    for method, color in zip(METHODS, COLORS):
        if method in runs:
            ax.plot(runs[method]['days'], runs[method][k], color=color, linewidth=1.2, label=METHODS[method])
    ax.axhline(spinup, color='gray', linestyle='--', linewidth=1, label='1/4° spin-up')
    ax.set_title(title, loc='left')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time [days]')
    ax.set_xlim(0, max(r['days'][-1] for r in runs.values()))
series_axes[1].axhspan(data_rmse.min(), data_rmse.max(), color='gray', alpha=0.25, linewidth=0)
series_axes[1].set_ylim(bottom=0)
series_axes[0].legend(frameon=False, loc='lower right')
series_axes[2].axis('off')

os.makedirs(FIGURES, exist_ok=True)
fig.savefig(f'{FIGURES}/fig5_aposteriori.pdf', bbox_inches='tight')
plt.show()
