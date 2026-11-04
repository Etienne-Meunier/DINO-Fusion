"""Figure 7: drift of the temperature and salinity on each level in the a posteriori runs of fig5 (experiment EXPERIMENT,
1/4° NEMO run started from the median member of each generation of the sweep PREFIX).

For each method of METHODS (those of fig5, with its COLORS), the RMSE on each level of T and S to the initial state
(day 0, the restart the run starts from), weighted by the cell volume e1t e2t e3t of each output (time-varying e3t),
averaged over the last LAST_DAYS daily outputs of the run, then smoothed along depth with a Gaussian of SMOOTHING
levels. One line per method, one panel per field, saved as two files in the `fig7` subfolder of FIGURES. A method whose
run is not collected is skipped.
"""
from common import *

from scipy.ndimage import gaussian_filter1d
from fig5_aposteriori import COLORS, EXPERIMENT, METHODS

FIELDS = {'toce': r'RMSE to day 0, $T$ [°C]', 'soce': r'RMSE to day 0, $S$ [psu]'}
LAST_DAYS = 10
SMOOTHING = 1.5  # standard deviation of the vertical Gaussian, in levels
LEGEND_FIELD = 'toce'
TOP_MARGIN = 0.04  # space above the surface, in fraction of the stretched depth axis
DEPTH_MAX, DEPTH_TICKS = 3500, [0, 100, 250, 500, 1000, 2000, 3000]


def drift(folder, last_days):
    """Drift (z_c,) of each field of FIELDS of the EXPERIMENT run in `folder`: per level, the volume-weighted RMSE of
    each of the last `last_days` daily outputs to the restart it starts from, averaged over them, smoothed along depth
    with a Gaussian of SMOOTHING levels over the ocean levels, NaN on land levels."""
    paths = experiment_paths(EXPERIMENT)
    exp = aposteriori_experiment(f'{folder}/{paths["results"]}', files=('T_3D',))
    T = exp.data['T_3D'].isel(t=slice(-last_days, None))
    initial = restart_state(exp, f'{folder}/{paths["restart_1_4deg"]}')
    volume = cell_volume(exp, T.e3t)
    out = {}
    for f in FIELDS:
        d = np.sqrt(((T[f] - initial[f]) ** 2).weighted(volume).mean(('y_c', 'x_c'))).mean('t').values
        ocean = np.isfinite(d)
        out[f] = np.full_like(d, np.nan)
        out[f][ocean] = gaussian_filter1d(d[ocean], SMOOTHING, mode='nearest')
    return out


def compute(methods=METHODS):
    """Drift of each method of `methods` whose EXPERIMENT run is collected: {'folders', 'drifts': {method: {field:
    (z_c,)}}, 'depth': gdept_1d of the 1/4° grid}."""
    folders = collected_runs(methods, EXPERIMENT)
    drifts = {}
    for method, folder in folders.items():
        print(f'{method} : {folder}')
        drifts[method] = drift(folder, LAST_DAYS)
        for f in FIELDS:
            print(f'  {f} : max RMSE {np.nanmax(drifts[method][f]):.4f} at level {int(np.nanargmax(drifts[method][f]))}')
    return {'folders': folders, 'drifts': drifts, 'depth': aposteriori_experiment().mask.gdept_1d.values}


def plot(computed, methods=METHODS, colors=COLORS):
    """One figure per field of FIELDS of `computed` (`compute`), one line per method with `colors` (one per method of
    `methods` (method -> legend label), same order): {field: figure}."""
    drifts, depth = computed['drifts'], stretch(computed['depth'])
    span = stretch(DEPTH_MAX)
    figs = {}
    for field, xlabel in FIELDS.items():
        fig, ax = plt.subplots(figsize=(3.2, 3.6))
        for method, d in drifts.items():
            ax.plot(d[field], depth, color=colors[list(methods).index(method)], linewidth=1.2, label=methods[method])
        depth_axis(ax, DEPTH_MAX, DEPTH_TICKS)
        ax.set_ylim(span, -TOP_MARGIN * span)
        ax.set_ylabel('depth [m]')
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2)
        if field == LEGEND_FIELD:
            ax.legend(frameon=False, loc='lower right', handletextpad=0.4)
        figs[field] = fig
    return figs


def main(methods=METHODS, colors=COLORS, out_name='fig7_drift'):
    """Compute and plot the drift of `methods` with `colors`, save the figures as
    `{FIGURES}/fig7/{out_name}_{field}.pdf`."""
    computed = compute(methods)
    for field, fig in plot(computed, methods, colors).items():
        save(fig, f'{FIGURES}/fig7/{out_name}_{field}.pdf', sources=list(computed['folders'].values()),
             bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
