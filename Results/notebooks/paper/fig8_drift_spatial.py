"""Figure 8: map of the temperature drift over the first days of the a posteriori runs of fig5 and fig7 (experiment
EXPERIMENT, 1/4° NEMO run started from the median member of each generation of the sweep PREFIX).

For each method of METHODS, side by side: at every water column, the thickness-weighted RMSE over depth of T to the
initial state (day 0, the converted generated state),
    sqrt(sum_z e3t (T(x, y, z) - T_init(x, y, z))^2 / sum_z e3t),
averaged over the first FIRST_DAYS daily outputs, divided by its area-weighted mean over the domain (given in the title
of each panel). One logarithmic colour scale for all the panels, centred on 1 (the domain mean). A method whose run is
not collected yet is skipped.
"""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cm
from matplotlib.colors import LogNorm
from nemo_cookbook import NEMODataTree
from settings import COLLECT, EXP, REFERENCES_REL, aposteriori_dirs, experiment_paths
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

EXPERIMENT = 'daily_2m'
DENSITY = DENSITY_RUN
METHODS = {'data': 'Data',
           'none': 'No constraint',
           DENSITY: rf'Increasing density, $\eta = {gens_args[DENSITY]["eta"]:g}$',
           'zero_mean_border0': 'T,S mean constraint'}
REFERENCE = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/restart25_arch'
DOMAIN = f'{REFERENCE}/domain_cfg_out.nc'
FIRST_DAYS = 10  # the column RMSE of the maps is averaged over the first FIRST_DAYS daily outputs
RATIO_MAX = 10  # the colour scale spans [1 / RATIO_MAX, RATIO_MAX] times the domain mean


def output_tree(results):
    """NEMODataTree of the daily T outputs of a run, e3t time-varying (written by XIOS), times in seconds."""
    paths = {'parent': {'domain': DOMAIN, 'gridT': f'{results}/{EXP}_1d_grid_T_3D.nc'}}
    return NEMODataTree.from_paths(paths, iperio=True, decode_times=False)


def restart_tree(path):
    """NEMODataTree of the now T and S of a NEMO restart, with e3t = e3t_0."""
    r = xr.open_dataset(path)
    grid_t = r[['tn', 'sn']].rename(tn='toce', sn='soce', nav_lev='deptht')
    return NEMODataTree.from_datasets({'parent': {'domain': xr.open_dataset(DOMAIN), 'gridT': grid_t}},
                                      iperio=True, linssh=True)


def setup_map(ax, left_labels):
    """Coastlines, borders, extent and gridlines of the paper maps, as fig2."""
    ax.coastlines(alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = gl.right_labels = gl.bottom_labels = False
    gl.left_labels = left_labels
    gl.ylabel_style = {'size': 6}


@memo
def column_rmse(folder, n_days):
    """Thickness-weighted RMSE over depth (n_days, j, i) of T to the restart the EXPERIMENT run in `folder` starts from,
    for each of its first `n_days` daily outputs, NaN on land columns, and the cell area (j, i)."""
    paths = experiment_paths(EXPERIMENT)
    initial = restart_tree(f'{folder}/{paths["restart_1_4deg"]}')
    outputs = output_tree(f'{folder}/{paths["results"]}')
    tmask = initial['gridT/tmask'].values
    start = initial['gridT/toce'].values[0]
    out = []
    # One daily output at a time: a whole run does not fit in memory next to the session data.
    for t in range(n_days):
        e3t = np.nan_to_num(outputs['gridT/e3t'].isel(time_counter=t).values) * tmask
        sq = np.nan_to_num((outputs['gridT/toce'].isel(time_counter=t).values - start) ** 2)
        with np.errstate(invalid='ignore'):
            out.append(np.sqrt((sq * e3t).sum(0) / e3t.sum(0)).astype(np.float32))
    return np.stack(out), (outputs['gridT/e1t'] * outputs['gridT/e2t']).values


def domain_mean(rmse, area):
    """Area-weighted mean (n_days,) over the ocean columns of the column RMSE (n_days, j, i)."""
    w = np.where(np.isfinite(rmse), area, 0.)
    return (np.nan_to_num(rmse) * w).sum((-2, -1)) / w.sum((-2, -1))


def main(methods, out_name):
    """Map, side by side for each method of `methods` (method -> panel title) whose EXPERIMENT run is collected, the
    column RMSE of T to day 0 averaged over the first FIRST_DAYS daily outputs over its domain mean, and save the figure
    as `{FIGURES}/{out_name}.pdf`."""
    folders = aposteriori_dirs(constraints=methods)
    maps, means = {}, {}
    for method in methods:
        folder = folders.get(method)
        if folder is None or not os.path.exists(f'{folder}/{experiment_paths(EXPERIMENT)["results"]}/{COLLECT}'):
            print(f'WARNING : no collected {EXPERIMENT} run for {method}, skipped')
            continue
        rmse, area = column_rmse(folder, FIRST_DAYS)
        maps[method] = rmse.mean(0)
        means[method] = float(domain_mean(maps[method], area))
        print(f'{method} : {folder}, domain mean {means[method]:.3f} °C')

    grid = restart_tree(f'{folders[next(iter(maps))]}/{experiment_paths(EXPERIMENT)["restart_1_4deg"]}')['gridT']
    lon, lat = grid['glamt'].values - 60, grid['gphit'].values
    ratios = {m: maps[m] / means[m] for m in maps}

    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    fig, axes = plt.subplots(1, len(maps), figsize=(2.2 * len(maps), 4.2), subplot_kw=dict(projection=proj),
                             gridspec_kw=dict(wspace=0.15))
    for i, (ax, (method, r)) in enumerate(zip(np.atleast_1d(axes), ratios.items())):
        pm = ax.pcolormesh(lon, lat, r, cmap=cm.balance, norm=LogNorm(1 / RATIO_MAX, RATIO_MAX), shading='nearest',
                           rasterized=True, transform=ccrs.PlateCarree())
        setup_map(ax, i == 0)
        ax.set_title(f'{methods[method]}\ndomain mean {means[method]:.2g} °C', loc='left')
    ticks = [1 / RATIO_MAX, 1, RATIO_MAX]
    cb = fig.colorbar(pm, ax=axes, orientation='vertical', fraction=0.03, pad=0.02, ticks=ticks, extend='both',
                      label=f'RMSE over depth to day 0 / domain mean\n$T$, mean of days 1 to {FIRST_DAYS}')
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([f'{t:g}' for t in ticks])

    os.makedirs(FIGURES, exist_ok=True)
    save(fig, f'{FIGURES}/{out_name}.pdf', sources=[DOMAIN, *(folders[m] for m in maps)], bbox_inches='tight',
         dpi=200)
    plt.show()


if __name__ == '__main__':
    main(METHODS, 'fig8_drift_spatial')
