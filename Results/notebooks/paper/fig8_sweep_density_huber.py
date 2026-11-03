"""Figure 8, density sweep: the temperature drift over the first days of the a posteriori runs started from the sweep of
the increasing-density constraint strength eta, and the density error of their initial states (experiment EXPERIMENT,
1/4° NEMO run started from the median member of each generation, minimal restart).

Runs: the sweep SWEEP_PREFIX, with eta read from `selection.json`, and the unconstrained run with BorderZero projection
(`none_border0` of the sweep ZERO_PREFIX) as eta = 0. A run that is not collected yet is skipped.

`fig8_sweep_density_huber.pdf`: maps as `fig8_drift_spatial.py`, one panel per run by increasing eta, on rows of
PANELS_PER_ROW. At every water column, the thickness-weighted RMSE over depth of T to the initial state, averaged over
the first FIRST_DAYS daily outputs, divided by its area-weighted mean over the domain (given in the title of each
panel). One logarithmic colour scale for all the panels, centred on 1 (the domain mean).

`fig8_sweep_density_huber_vs_eta.pdf`: against eta (symlog axis), the domain mean of the maps and, on a second y axis,
the density error E (percent of unstable volume, `metrics.compute_density_error`) of the selected median member (top),
and the error of the initial section (bottom): the `rmse` of `selection.json`, RMSE of the zonal sigma_0 section of the
selected median member to the batch-mean section of a data batch, and `rmse_mean`, its mean over the member batch.

`fig8_correlation.pdf`: domain mean of the maps against the density error E of the selected median member (log axis),
one point per run up to eta = CORRELATION_ETA_MAX, labelled with its eta.
"""
from common import *

import cartopy.crs as ccrs
import cmocean.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
from fig8_drift_spatial import DOMAIN, FIRST_DAYS, RATIO_MAX, domain_mean, output_tree, restart_tree, setup_map
from metrics import compute_density_error
from settings import COLLECT, SELECTION, aposteriori_dirs, experiment_paths, read_json

EXPERIMENT = 'daily_2w_minimal'
SWEEP_PREFIX = 'sweep_density_huber_aposteriori'
ZERO_PREFIX = 'optimal'
ZERO_CONSTRAINT = 'none_border0'
PANELS_PER_ROW = 5
CORRELATION_ETA_MAX = 30
LINTHRESH = 1  # linear part of the symlog eta axis, which has to hold eta = 0


@memo
def column_rmse(folder, n_days):
    """Thickness-weighted RMSE over depth (n_days, j, i) of T to the restart the EXPERIMENT run in `folder` starts from,
    for each of its first `n_days` daily outputs, NaN on land columns, and the cell area (j, i).

    Same computation as `fig8_drift_spatial.column_rmse`, which is not reused by setting its module EXPERIMENT: its memo
    key does not hold the experiment, so in a session it would return the daily_2m result of a folder holding both runs.
    """
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


def member_density_error(selection):
    """Percent of unstable ocean volume of the member of `selection` (content of `selection.json`), from its generation."""
    generated = np.load(f'{os.environ["OCEANDATA"]}/{selection["generation_rel"]}/generated.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(generated[selection['member']:selection['member'] + 1]))
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    return float(np.asarray(compute_density_error(fields, load_mask()), dtype=float)[0])


def load_runs():
    """Runs collected for EXPERIMENT, by increasing eta: a list of dicts with the aposteriori `folder`, `eta`, the
    FIRST_DAYS mean column RMSE `map` (j, i), its domain `mean`, the `rmse` and `rmse_mean` of the selection, and the
    density error `E` of the selected member."""
    folders = aposteriori_dirs(prefix=SWEEP_PREFIX) | aposteriori_dirs(prefix=ZERO_PREFIX, constraints=[ZERO_CONSTRAINT])
    runs = []
    for constraint, folder in folders.items():
        if not os.path.exists(f'{folder}/{experiment_paths(EXPERIMENT)["results"]}/{COLLECT}'):
            print(f'WARNING : no collected {EXPERIMENT} run for {constraint}, skipped')
            continue
        selection = read_json(f'{folder}/{SELECTION}')
        rmse, area = column_rmse(folder, FIRST_DAYS)
        rmse_map = rmse.mean(0)
        runs.append(dict(folder=folder, eta=selection.get('eta', 0.), map=rmse_map, mean=float(domain_mean(rmse_map, area)),
                         rmse=selection['rmse'], rmse_mean=selection['rmse_mean'], E=member_density_error(selection)))
        print(f'{constraint} : {folder}, domain mean {runs[-1]["mean"]:.3f} °C')
    return sorted(runs, key=lambda run: run['eta'])


def plot_maps(runs):
    """Save the map of the column RMSE over its domain mean of each of `runs` as `fig8_sweep_density_huber.pdf`."""
    grid = restart_tree(f'{runs[0]["folder"]}/{experiment_paths(EXPERIMENT)["restart_1_4deg"]}')['gridT']
    lon, lat = grid['glamt'].values - 60, grid['gphit'].values

    n_rows = int(np.ceil(len(runs) / PANELS_PER_ROW))
    proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
    fig, axes = plt.subplots(n_rows, PANELS_PER_ROW, figsize=(2.2 * PANELS_PER_ROW, 4.2 * n_rows),
                             subplot_kw=dict(projection=proj), gridspec_kw=dict(wspace=0.15), squeeze=False)
    for i, (ax, run) in enumerate(zip(axes.flat, runs)):
        pm = ax.pcolormesh(lon, lat, run['map'] / run['mean'], cmap=cm.balance, norm=LogNorm(1 / RATIO_MAX, RATIO_MAX),
                           shading='nearest', rasterized=True, transform=ccrs.PlateCarree())
        setup_map(ax, i % PANELS_PER_ROW == 0)
        ax.set_title(rf'$\eta = {run["eta"]:g}$' f'\ndomain mean {run["mean"]:.2g} °C', loc='left')
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    ticks = [1 / RATIO_MAX, 1, RATIO_MAX]
    cb = fig.colorbar(pm, ax=axes, orientation='vertical', fraction=0.03, pad=0.02, ticks=ticks, extend='both',
                      label=f'RMSE over depth to day 0 / domain mean\n$T$, mean of days 1 to {FIRST_DAYS}')
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([f'{t:g}' for t in ticks])

    save(fig, f'{FIGURES}/fig8_sweep_density_huber.pdf', sources=[DOMAIN, *(run['folder'] for run in runs)],
         bbox_inches='tight', dpi=200)


def plot_vs_eta(runs):
    """Save the domain mean drift and the density error of the initial state of `runs` against eta as
    `fig8_sweep_density_huber_vs_eta.pdf`."""
    etas = [run['eta'] for run in runs]
    fig, (ax_drift, ax_density) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout='constrained')
    ax_drift.plot(etas, [run['mean'] for run in runs], marker='o', ms=4, color='C0')
    ax_drift.set_title(f'Domain mean of the RMSE over depth of $T$ to day 0, mean of days 1 to {FIRST_DAYS}, and density error of the median member', loc='left')
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
    save(fig, f'{FIGURES}/fig8_sweep_density_huber_vs_eta.pdf', sources=[DOMAIN, *(run['folder'] for run in runs)],
         bbox_inches='tight', dpi=200)


def plot_correlation(runs):
    """Save the domain mean drift against the density error of the median member of `runs` up to CORRELATION_ETA_MAX,
    labelled with eta, as `fig8_correlation.pdf`."""
    runs = [run for run in runs if run['eta'] <= CORRELATION_ETA_MAX]
    errors, drifts = [run['E'] for run in runs], [run['mean'] for run in runs]
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout='constrained')
    ax.scatter(errors, drifts, s=18, color='C0')
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
    save(fig, f'{FIGURES}/fig8_correlation.pdf', sources=[DOMAIN, *(run['folder'] for run in runs)], bbox_inches='tight')


if __name__ == '__main__':
    runs = load_runs()
    plot_maps(runs)
    plot_vs_eta(runs)
    plot_correlation(runs)
    print('eta, domain mean drift (°C), selection rmse, rmse_mean, density error E (%)')
    for run in runs:
        print(f'{run["eta"]:g}, {run["mean"]:.4f}, {run["rmse"]:.4f}, {run["rmse_mean"]:.4f}, {run["E"]:.3f}')
    plt.show()
