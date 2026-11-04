"""Figure 8, density sweep: the temperature drift over the first days of the a posteriori runs started from the sweep of
the increasing-density constraint strength eta, and the density error of their initial states (experiment EXPERIMENT,
1/4° NEMO run started from the median member of each generation, minimal restart).

Runs: the sweep SWEEP_PREFIX, with eta read from `selection.json`, and the unconstrained run with BorderZero projection
(`none_border0` of the sweep ZERO_PREFIX) as eta = 0. A run that is not collected is skipped.

`fig8_sweep_density_huber.pdf`: maps as `fig8_drift_spatial.py`, one panel per run by increasing eta, on rows of
PANELS_PER_ROW.

`fig8_sweep_density_huber_vs_eta.pdf`: against eta (symlog axis), the domain mean of the maps and, on a second y axis,
the density error E (`common.density_error`) of the selected median member (top), and the error of the initial section
(bottom): the `rmse` of `selection.json`, RMSE of the zonal sigma_0 section of the selected median member to the
batch-mean section of a data batch, and `rmse_mean`, its mean over the member batch (both computed by the a posteriori
selection step).

`fig8_correlation.pdf`: domain mean of the maps against the density error E of the selected median member (log axis),
one point per run up to eta = CORRELATION_ETA_MAX, labelled with its eta.
"""
from common import *

from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
from fig8_drift_spatial import FIRST_DAYS, column_rmse, plot_maps
from settings import SELECTION, read_json

EXPERIMENT = 'daily_2w_minimal'
SWEEP_PREFIX = 'sweep_density_huber_aposteriori'
ZERO_PREFIX = 'optimal'
ZERO_CONSTRAINT = 'none_border0'
PANELS_PER_ROW = 5
CORRELATION_ETA_MAX = 30
LINTHRESH = 1  # linear part of the symlog eta axis, which has to hold eta = 0


def load_runs():
    """Runs collected for EXPERIMENT, by increasing eta: a list of dicts with the aposteriori `folder`, `eta`, the
    FIRST_DAYS mean column RMSE `map` (j, i), its domain `mean`, the `rmse` and `rmse_mean` of the selection, and the
    density error `E` of the selected member."""
    folders = (collected_runs(None, EXPERIMENT, prefix=SWEEP_PREFIX)
               | collected_runs([ZERO_CONSTRAINT], EXPERIMENT, prefix=ZERO_PREFIX))
    runs = []
    for constraint, folder in folders.items():
        selection = read_json(f'{folder}/{SELECTION}')
        rmse_map, mean = column_rmse(folder, EXPERIMENT, FIRST_DAYS)
        generated = np.load(f'{os.environ["OCEANDATA"]}/{selection["generation_rel"]}/generated.npy', mmap_mode='r')
        member = torch.from_numpy(np.array(generated[selection['member']:selection['member'] + 1]))
        runs.append(dict(folder=folder, eta=selection.get('eta', 0.), map=rmse_map, mean=mean, rmse=selection['rmse'],
                         rmse_mean=selection['rmse_mean'], E=float(density_error(member)[0])))
        print(f'{constraint} : {folder}, domain mean {mean:.3f} °C')
    return sorted(runs, key=lambda run: run['eta'])


def plot_vs_eta(runs):
    """Figure of the domain mean drift and of the density error of the initial state of `runs` against eta."""
    etas = [run['eta'] for run in runs]
    fig, (ax_drift, ax_density) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout='constrained')
    ax_drift.plot(etas, [run['mean'] for run in runs], marker='o', ms=4, color='C0')
    ax_drift.set_title(f'Domain mean of the RMSE over depth of $T$ to day 0, mean of days 1 to {FIRST_DAYS}, '
                       'and density error of the median member', loc='left')
    ax_drift.set_ylabel('RMSE of $T$ (°C)', color='C0')
    ax_drift.tick_params(axis='y', colors='C0')
    ax_error = ax_drift.twinx()
    ax_error.plot(etas, [run['E'] for run in runs], marker='s', ms=4, color='C1')
    ax_error.set_ylabel('density error $E$ (%)', color='C1')
    ax_error.tick_params(axis='y', colors='C1')
    ax_error.set_ylim(bottom=0)
    ax_density.plot(etas, [run['rmse'] for run in runs], marker='o', ms=4, color='C0', label='Selected median member')
    ax_density.plot(etas, [run['rmse_mean'] for run in runs], marker='o', ms=3, color='C0', alpha=0.5, ls='--',
                    label='Mean over the members')
    ax_density.set_title('RMSE of the initial zonal $\\sigma_0$ section to the data batch mean', loc='left')
    ax_density.set_ylabel('kg/m$^3$')
    ax_density.legend(frameon=False)
    ax_density.set_xlabel(r'$\eta$')
    ax_density.set_xscale('symlog', linthresh=LINTHRESH)
    ax_density.set_xticks(etas)
    ax_density.set_xticklabels([f'{eta:g}' for eta in etas])
    ax_density.xaxis.set_minor_locator(NullLocator())
    ax_density.tick_params(axis='x', labelrotation=90)
    for ax in (ax_drift, ax_density):
        ax.grid(alpha=0.3)
    return fig


def plot_correlation(runs):
    """Figure of the domain mean drift against the density error of the median member of `runs`, labelled with eta."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout='constrained')
    ax.scatter([run['E'] for run in runs], [run['mean'] for run in runs], s=18, color='C0')
    for run in runs:
        ax.annotate(rf'$\eta = {run["eta"]:g}$' if run['eta'] == 0 else f'{run["eta"]:g}', (run['E'], run['mean']),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1.5, 2, 3, 4, 5])
    ax.set_xlabel('Density error $E$ of the initial state (%)')
    ax.set_ylabel(f'Domain mean of the RMSE over depth of $T$ to day 0,\nmean of days 1 to {FIRST_DAYS} (°C)')
    ax.grid(alpha=0.3)
    return fig


def compute():
    """Drift and density error of every run of the sweep (`load_runs`), printed: {'runs'}."""
    runs = load_runs()
    print('eta, domain mean drift (°C), selection rmse, rmse_mean, density error E (%)')
    for run in runs:
        print(f'{run["eta"]:g}, {run["mean"]:.4f}, {run["rmse"]:.4f}, {run["rmse_mean"]:.4f}, {run["E"]:.3f}')
    return {'runs': runs}


def plot(computed):
    """The three figures of `computed` (`compute`): {file name in FIGURES, without extension: figure}."""
    runs = computed['runs']
    maps = plot_maps([run['map'] for run in runs], [run['mean'] for run in runs],
                     [rf'$\eta = {run["eta"]:g}$' for run in runs], PANELS_PER_ROW)
    return {'fig8_sweep_density_huber': maps, 'fig8_sweep_density_huber_vs_eta': plot_vs_eta(runs),
            'fig8_correlation': plot_correlation([run for run in runs if run['eta'] <= CORRELATION_ETA_MAX])}


def main():
    """Compute, plot and save the figures in FIGURES."""
    computed = compute()
    figs, runs = plot(computed), computed['runs']
    folders = [run['folder'] for run in runs]
    for name in ('fig8_sweep_density_huber', 'fig8_sweep_density_huber_vs_eta'):
        save(figs[name], f'{FIGURES}/{name}.pdf', sources=folders, bbox_inches='tight', dpi=200)
    save(figs['fig8_correlation'], f'{FIGURES}/fig8_correlation.pdf',
         sources=[run['folder'] for run in runs if run['eta'] <= CORRELATION_ETA_MAX], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
