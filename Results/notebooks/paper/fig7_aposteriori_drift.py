"""Figure 7: drift of the temperature and salinity on each level in the a posteriori runs of fig5 (experiment EXPERIMENT,
1/4° NEMO run started from the median member of each generation of the sweep PREFIX).

For each method of METHODS, the volume-weighted RMSE on each level of T and S to the initial state (day 0, the converted
generated state), averaged over the last LAST_DAYS daily outputs of the run (60 days long), then smoothed along depth with a Gaussian of SMOOTHING levels. Shown as one line per method, one panel per field,
saved as two files in the `fig7` subfolder of FIGURES. A method whose run is not collected yet is skipped.
"""
from common import *

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

from nemo_cookbook import NEMODataTree
from scipy.ndimage import gaussian_filter1d
from settings import COLLECT, EXP, REFERENCES_REL, aposteriori_dirs, experiment_paths
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

EXPERIMENT = 'daily_2m'
DENSITY = DENSITY_RUN
METHODS = {'data': 'Data',
           'none': 'No constraint',
           DENSITY: 'Increasing density',
           'zero_mean_border0': 'T,S mean constraint'}
COLORS = ['k', 'C0', 'C1', 'C2']
FIELDS = {'toce': r'RMSE to day 0, $T$ [°C]', 'soce': r'RMSE to day 0, $S$ [psu]'}
REFERENCE = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/restart25_arch'
DOMAIN = f'{REFERENCE}/domain_cfg_out.nc'
LAST_DAYS = 10  # the RMSE is averaged over the last LAST_DAYS daily outputs
SMOOTHING = 1.5  # standard deviation of the vertical Gaussian, in levels
LEGEND_FIELD = 'toce'  # the only panel with a legend
TOP_MARGIN = 0.04  # space above the surface, in fraction of the stretched depth axis
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000]
stretch = lambda d: np.asarray(d) ** DEPTH_POWER


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


def level_rmse(outputs, initial, last_days):
    """RMSE (time, k) on each level of each field of FIELDS between the last `last_days` daily outputs and the initial
    state, weighted by the cell volume of each output over the ocean cells, NaN on land levels."""
    tmask = initial['gridT/tmask'].values
    start = {f: initial[f'gridT/{f}'].values[0] for f in FIELDS}
    volume = outputs.cell_volume('gridT')
    out = {f: [] for f in FIELDS}
    # One daily output at a time: a whole run (T, S and e3t, ~5 GB) does not fit in memory next to the session data.
    n = volume.sizes['time_counter']
    for t in range(n - last_days, n):
        vol = np.nan_to_num(volume.isel(time_counter=t).values) * tmask
        weight = vol.sum((-2, -1))
        weight = np.where(weight > 0, weight, np.nan)
        for f in FIELDS:
            sq = np.nan_to_num((outputs[f'gridT/{f}'].isel(time_counter=t).values - start[f]) ** 2)
            out[f].append(np.sqrt((sq * vol).sum((-2, -1)) / weight))
    return {f: np.stack(v) for f, v in out.items()}


@memo
def drift(folder, last_days):
    """Depth (k,) and drift (k,) of each field of FIELDS of the EXPERIMENT run in `folder`: per level, the RMSE of each
    of the last `last_days` daily outputs to the restart it starts from, averaged over them, smoothed along depth with a Gaussian of
    SMOOTHING levels over the ocean levels."""
    paths = experiment_paths(EXPERIMENT)
    restart = restart_tree(f'{folder}/{paths["restart_1_4deg"]}')
    errors = level_rmse(output_tree(f'{folder}/{paths["results"]}'), restart, last_days)
    out = {'depth': restart['gridT']['deptht'].values}
    for f in FIELDS:
        d = errors[f].mean(0)
        ocean = np.isfinite(d)
        out[f] = np.full_like(d, np.nan)
        out[f][ocean] = gaussian_filter1d(d[ocean], SMOOTHING, mode='nearest')
    return out


def plot_field(field, drifts, depth, methods=METHODS, colors=COLORS):
    """Lines of the RMSE drift of `field` against depth for each method of `drifts`, one figure, returned.

    `methods` maps each method to its legend label and `colors` gives the colour of the method at the same position in
    `methods`; `drifts` maps a subset of `methods` to the output of `drift`, `depth` is the depth (k,) of the levels.
    """
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    span = stretch(DEPTH_MAX)
    for method, d in drifts.items():
        k = list(methods).index(method)
        ax.plot(d[field], stretch(depth), color=colors[k], linewidth=1.2, label=methods[method])
    ax.set_ylim(span, -TOP_MARGIN * span)
    ax.set_yticks(stretch(DEPTH_TICKS))
    ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
    ax.set_ylabel('depth [m]')
    ax.set_xlabel(FIELDS[field])
    ax.set_xlim(left=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2)
    if field == LEGEND_FIELD:
        ax.legend(frameon=False, loc='lower right', handletextpad=0.4)
    return fig


def main(methods, colors, out_name):
    """Compute the drift of each method of `methods` (method -> legend label) whose EXPERIMENT run is collected, plot
    one figure per field of FIELDS with `colors` (one per method, same order) and save it as
    `{FIGURES}/fig7/{out_name}_{field}.pdf`."""
    folders = aposteriori_dirs(constraints=methods)
    drifts = {}
    for method in methods:
        folder = folders.get(method)
        if folder is None or not os.path.exists(f'{folder}/{experiment_paths(EXPERIMENT)["results"]}/{COLLECT}'):
            print(f'WARNING : no collected {EXPERIMENT} run for {method}, skipped')
            continue
        print(f'{method} : {folder}')
        drifts[method] = drift(folder, LAST_DAYS)
        for f in FIELDS:
            d = drifts[method][f]
            print(f'  {f} : max RMSE {np.nanmax(d):.4f} at level {int(np.nanargmax(d))}')

    depth = next(iter(drifts.values()))['depth']
    os.makedirs(f'{FIGURES}/fig7', exist_ok=True)
    for field in FIELDS:
        fig = plot_field(field, drifts, depth, methods, colors)
        save(fig, f'{FIGURES}/fig7/{out_name}_{field}.pdf', sources=[DOMAIN, *(folders[m] for m in drifts)],
             bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main(METHODS, COLORS, 'fig7_drift')
