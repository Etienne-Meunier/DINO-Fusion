"""Figure 4: strength of the increasing density constraint (negative_grad_density_huber, C7) as a function of eta.

Top: on twin y axes, density error (percent of unstable volume, mean +/- std over the batch) and RMSE of the batch mean
to the data (RMS over ocean cells of the difference between the batch-average and the data-average sigma_0), of the b8
eta sweep of `generate/4_generate_huber_sweep.py`. eta 0 is the paired unconstrained run (boundary projection only, same
seed and batch). The data references are dotted: its density error, and for the RMSE, computed on b8 batches against the
full data batch, the mean RMSE of N_SUBSETS random 8-member subsets of the data.
Bottom: batch-average zonal sigma_0 section, with the profile over the BAND latitudes as inset (as fig1), of the
unconstrained run and of the ETAS_FRIEZE runs.
"""
from common import *

import cmocean.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metrics import compute_density_error
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

SWEEP = f'{INFERENCE}/sweep_density_huber/negative_grad_density_huber_border0'
NONE_B8 = f'{INFERENCE}/sweep_baselines/none_border0_b8/151026-145216'
ETAS_FRIEZE = [10, 100, 300]
N_SUBSETS = 200
BAND = (-65, -45)
LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]
COLOR_ERROR, COLOR_RMSE = 'C0', 'C1'

file_mask_LR = load_mask()
stretch = lambda d: np.asarray(d) ** DEPTH_POWER
depth = file_mask_LR.gdept_1d.values[:-1]
latitude = file_mask_LR.gphit.isel(nav_lon=10).values
ocean = file_mask_LR.tmask.values.astype(bool)


def load_sweep(sweep=SWEEP):
    """Generation of every run of `sweep`, keyed by its eta; for a repeated eta the latest timestamp is kept."""
    runs = {}
    for d in sorted(glob(f'{sweep}/*/generated.npy'),
                    key=lambda v: datetime.strptime(os.path.basename(os.path.dirname(v)), '%d%m%y-%H%M%S')):
        with open(f'{os.path.dirname(d)}/args.json') as f:
            runs[json.load(f)['eta']] = torch.from_numpy(np.load(d))
    return dict(sorted(runs.items()))


def density_errors(x):
    """Percent of unstable ocean volume of each member of a normalized batch x."""
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    return np.asarray(compute_density_error(fields, file_mask_LR), dtype=float)


def sigma0(x):
    """sigma_0 - 1000 of a normalized batch x on the ocean cells, (batch, n_ocean)."""
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    return (compute_density(fields, file_mask_LR).values[:, ocean] - 1000).astype(np.float32)


def rmse_to_data(s):
    """RMS over ocean cells of the batch mean of `s` (batch, n_ocean) minus the data mean."""
    return np.sqrt(np.mean((s.mean(0) - sigma_data.mean(0)) ** 2))


def band_profile(section):
    """Mean over the BAND latitudes of a (depth, lat) section."""
    return np.nanmean(section[:, (latitude >= BAND[0]) & (latitude < BAND[1])], axis=1)


def plot_section(ax, section, title):
    """Contour a (depth, lat) section on the stretched depth axis, with its BAND profile as a black inset."""
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
    inset_ax.plot(band_profile(section), y, color='black', linewidth=2)
    inset_ax.set_facecolor('gray')
    inset_ax.patch.set_alpha(0.4)
    inset_ax.set_xlim(PROFILE_RANGE)
    inset_ax.set_ylim(ax.get_ylim())
    inset_ax.set_yticks(ax.get_yticks())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    return cf


sweep = load_sweep()
# Here why don't we use the none_b16 from optimal ? 
none_b8 = torch.from_numpy(np.load(f'{NONE_B8}/generated.npy'))
runs = {0: none_b8} | sweep
sigma_data = sigma0(batch)
rng = np.random.default_rng(0)
rmse_data = np.mean([rmse_to_data(sigma_data[rng.choice(len(sigma_data), len(none_b8), replace=False)])
                     for _ in range(N_SUBSETS)])
errors = {eta: density_errors(x) for eta, x in runs.items()}
rmses = {eta: rmse_to_data(sigma0(x)) for eta, x in runs.items()}
error_data = density_errors(batch).mean()
for eta in runs:
    print(f'eta {eta:g} : density error {errors[eta].mean():.2f} +/- {errors[eta].std(ddof=1):.2f}, RMSE {rmses[eta]:.3f}')
print(f'data : density error {error_data:.2f}, RMSE of b{len(none_b8)} subsets {rmse_data:.3f}')

frieze = [(none_b8, 'No constraint')] + [(sweep[eta], rf'$\eta = {eta:g}$') for eta in ETAS_FRIEZE]
fig = plt.figure(figsize=(9, 4.6))
gs = fig.add_gridspec(2, len(frieze), height_ratios=[1, 1.25], hspace=0.45, wspace=0.08)
etas = np.asarray(list(runs))
ticks = etas[np.isin(etas, [0, 10, 30, 100, 300])]

ax = fig.add_subplot(gs[0, :])
mean, std = np.asarray([e.mean() for e in errors.values()]), np.asarray([e.std(ddof=1) for e in errors.values()])
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

frieze_axes = []
for j, (x, title) in enumerate(frieze):
    ax = fig.add_subplot(gs[1, j])
    frieze_axes.append(ax)
    cf = plot_section(ax, np.nanmean(zonal_sections(x, file_mask_LR), axis=0), f'{chr(ord("b") + j)}) {title}')
    ax.set_xlabel('latitude [°N]')
    if j == 0:
        ax.set_ylabel('depth [m]')
    else:
        ax.set_yticklabels([])
cbar = fig.colorbar(cf, ax=frieze_axes, label=r'$\sigma_0$ [kg m$^{-3}$ - 1000]', fraction=0.03, pad=0.02)
cbar.ax.invert_yaxis()

os.makedirs(FIGURES, exist_ok=True)
fig.savefig(f'{FIGURES}/fig4_constraint_strength.pdf', bbox_inches='tight')
plt.show()
