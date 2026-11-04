"""Figure 4: strength of the increasing density constraint (negative_grad_density_huber, C7) as a function of eta.

Top: on twin y axes, density error (`common.density_error`, percent of the ocean volume with a sigma_0 inversion, mean
+/- std over the batch) and RMSE of the batch mean to the data (RMS over ocean cells of the difference between the
batch-average and the data-average sigma_0), of the b8 eta sweep CONFIGS["fig4_sweep"]. eta 0 is the paired
unconstrained run (boundary projection only, same seed and batch). The data references are dotted: its density error,
and for the RMSE, computed on b8 batches against all the data states (`batch`), the mean RMSE of N_SUBSETS random 8-member subsets
of the data.
Bottom: batch-average zonal sigma_0 section, with its profile over the channel latitudes as inset (as fig1), of the
unconstrained run and of the ETAS_FRIEZE runs.
"""
from common import *

SWEEP = CONFIGS['fig4_sweep']
NONE_B8 = os.path.dirname(by_timestamp(glob(f"{CONFIGS['fig4_none']}/generated.npy"))[-1])
ETAS_FRIEZE = CONFIGS['fig4_frieze']
N_SUBSETS = 200
COLOR_ERROR, COLOR_RMSE = 'C0', 'C1'


def load_sweep(sweep=SWEEP):
    """Generation of every run of `sweep`, keyed by its eta; for a repeated eta the latest timestamp is kept."""
    runs = {}
    for path in by_timestamp(glob(f'{sweep}/*/generated.npy')):
        with open(f'{os.path.dirname(path)}/args.json') as f:
            runs[json.load(f)['eta']] = torch.from_numpy(np.load(path))
    return dict(sorted(runs.items()))


def ocean_sigma0(x):
    """sigma_0 (float32) of a normalized batch x on the ocean cells of GRID, (member, n_ocean)."""
    return sigma0(to_dataset(x)).values[:, GRID.mask.tmask.values == 1].astype(np.float32)


def compute():
    """Density error and RMSE to the data of every eta (0: the paired unconstrained run), printed, and the batch-mean
    sections of the frieze: {'errors': {eta: (member,)}, 'rmses': {eta: float}, 'error_data', 'rmse_data', 'frieze':
    [(section (z, y), title)], 'depth', 'lat'}."""
    sweep = load_sweep()
    none_b8 = torch.from_numpy(np.load(f'{NONE_B8}/generated.npy'))
    runs = {0: none_b8} | sweep
    sigma_data = ocean_sigma0(batch)
    rmse_to_data = lambda s: np.sqrt(np.mean((s.mean(0) - sigma_data.mean(0)) ** 2))
    rng = np.random.default_rng(0)
    rmse_data = np.mean([rmse_to_data(sigma_data[rng.choice(len(sigma_data), len(none_b8), replace=False)])
                         for _ in range(N_SUBSETS)])
    errors = {eta: density_error(x) for eta, x in runs.items()}
    rmses = {eta: rmse_to_data(ocean_sigma0(x)) for eta, x in runs.items()}
    error_data = density_error(batch).mean()
    for eta in runs:
        print(f'eta {eta:g} : density error {errors[eta].mean():.2f} +/- {errors[eta].std(ddof=1):.2f}, '
              f'RMSE {rmses[eta]:.3f}')
    print(f'data : density error {error_data:.3f}, RMSE of b{len(none_b8)} subsets {rmse_data:.3f}')
    frieze = [(none_b8, 'No constraint')] + [(sweep[eta], rf'$\eta = {eta:g}$') for eta in ETAS_FRIEZE]
    return {'errors': errors, 'rmses': rmses, 'error_data': error_data, 'rmse_data': rmse_data,
            'frieze': [(np.nanmean(zonal_sections(x), axis=0), title) for x, title in frieze],
            'depth': GRID.mask.gdept_1d.values[:-1], 'lat': GRID.mask.gphit.mean('x_c').values}


def plot(computed):
    """Curves of the density error and RMSE against eta and section frieze of `computed` (`compute`)."""
    errors, rmses, error_data, rmse_data = (computed[k] for k in ('errors', 'rmses', 'error_data', 'rmse_data'))
    frieze, depth, lat = computed['frieze'], computed['depth'], computed['lat']
    fig = plt.figure(figsize=(9, 4.6))
    gs = fig.add_gridspec(2, len(frieze), height_ratios=[1, 1.25], hspace=0.45, wspace=0.08)
    etas = np.asarray(list(errors))
    ticks = etas[np.isin(etas, [0, 10, 30, 100, 300])]

    ax = fig.add_subplot(gs[0, :])
    mean = np.asarray([e.mean() for e in errors.values()])
    std = np.asarray([e.std(ddof=1) for e in errors.values()])
    ax.fill_between(etas, mean - std, mean + std, color=COLOR_ERROR, alpha=0.25, linewidth=0)
    ax.plot(etas, mean, 'o-', color=COLOR_ERROR, markersize=3)
    ax.axhline(error_data, color=COLOR_ERROR, linestyle=':', linewidth=1)
    ax.set_ylabel('density error [%]', color=COLOR_ERROR)
    ax.tick_params(axis='y', colors=COLOR_ERROR)
    ax.set_ylim(bottom=0)
    ax_rmse = ax.twinx()
    ax_rmse.plot(etas, list(rmses.values()), 's-', color=COLOR_RMSE, markersize=3)
    ax_rmse.axhline(rmse_data, color=COLOR_RMSE, linestyle=':', linewidth=1)
    ax_rmse.set_ylabel(r'RMSE to data $\sigma_0$ [kg m$^{-3}$]', color=COLOR_RMSE)
    ax_rmse.tick_params(axis='y', colors=COLOR_RMSE)
    ax_rmse.set_ylim(bottom=0)
    ax.set_xscale('symlog', linthresh=10, linscale=0.5)
    ax.set_xlim(-1.5, 1.2 * etas.max())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{e:g}' for e in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r'$\eta$')
    ax.set_title('a) Density error and distance to the data', loc='left')
    ax.legend(handles=[plt.Line2D([], [], color='gray', linestyle=':', linewidth=1, label=NAMES['data'])],
              loc='upper center', frameon=False)

    frieze_axes = [fig.add_subplot(gs[1, j]) for j in range(len(frieze))]
    for j, (ax, (section, title)) in enumerate(zip(frieze_axes, frieze)):
        cf = plot_section(ax, section, lat, depth, f'{panel_letter(j + 1)}) {title}')
        ax.set_xlabel('latitude [°N]')
        if j == 0:
            ax.set_ylabel('depth [m]')
        else:
            ax.set_yticklabels([])
    fig.colorbar(cf, ax=frieze_axes, label=r'$\sigma_0$ [kg m$^{-3}$ - 1000]', fraction=0.03,
                 pad=0.02).ax.invert_yaxis()
    return fig


def main():
    """Compute, plot and save the figure as fig4_constraint_strength.pdf in FIGURES."""
    save(plot(compute()), f'{FIGURES}/fig4_constraint_strength.pdf', sources=[SWEEP, NONE_B8], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
