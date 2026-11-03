"""
Strength of the C7 constraint with and without the surface data constraint, as `paper/fig4_constraint_strenght.py`.

Top: density error E (mean +/- std over the batch) and RMSE of the batch mean sigma_0 to the data, against the C7 eta,
for C7 alone (repo b8 sweep, and runs/c7_c7200) and for C7 + the level-0 mean T and S held at the data mean
(`SurfaceMean`, sqrt, eta 1e-4, eps 9 ; runs/surf_c7*_e1e-4). eta 0 is the paired unconstrained run.
Middle: level-0 sigma_0 bias to the data against eta, both curves.
Bottom: batch-average zonal sigma_0 sections with the BAND profile as inset, of the unconstrained run and of C7 + surface
at ETAS_FRIEZE. Saves figures/fig_strength_surface.pdf / .png and prints the table.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

import cmocean.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metrics import compute_density_error
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

SWEEP = f'{INFERENCE}/sweep_density_huber/negative_grad_density_huber_border0'
NONE_B8 = f'{INFERENCE}/sweep_baselines/none_border0_b8/151026-145216'
ETAS = [10, 30, 50, 60, 70, 80, 90, 100, 200, 300]
ETAS_FRIEZE = [10, 100, 300]
N_SUBSETS = 200
BAND = (-65, -45)
LEVELS = np.arange(22., 28.51, 0.5)
PROFILE_RANGE = (25., 28.5)
DEPTH_POWER, DEPTH_MAX = 0.625, 3500
DEPTH_TICKS = [0, 100, 250, 500, 1000, 1500, 2000, 3000]
COLOR_C7, COLOR_SURF = 'C0', 'C1'

mask = load_mask()
stretch = lambda d : np.asarray(d) ** DEPTH_POWER
depth = mask.gdept_1d.values[:-1]
latitude = mask.gphit.isel(nav_lon=10).values
ocean = mask.tmask.values.astype(bool)
v0 = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[0]


def surf_run(eta) :
    """Run directory of C7 eta + surface constraint."""
    return HERE / 'runs' / ('c7surf_eta100_sqrt1e-4' if eta == 100 else f'surf_c7{eta:03d}_e1e-4')


def c7_runs() :
    """C7-alone batches keyed by eta : repo sweep (latest timestamp per eta) and runs/c7_c7200."""
    out = {}
    for d in sorted(glob(f'{SWEEP}/*/generated.npy'),
                    key=lambda v : datetime.strptime(os.path.basename(os.path.dirname(v)), '%d%m%y-%H%M%S')) :
        out[json.load(open(f'{os.path.dirname(d)}/args.json'))['eta']] = np.load(d)
    if (HERE / 'runs/c7_c7200/generated.npy').exists() :
        out[200.] = np.load(HERE / 'runs/c7_c7200/generated.npy')
    return dict(sorted(out.items()))


def fields(x) :
    return {k : v.detach().cpu().numpy() for k, v in observer.unnormalized(torch.as_tensor(x)).items()}


def stats(x) :
    """Per-member E (b,), sigma_0 on the ocean cells (b, n_ocean), level-0 volume-weighted sigma_0 (b,)."""
    f = fields(x)
    sig = compute_density(f, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values - 1000
    lev0 = np.nansum(np.where(ocean[0], sig[:, 0], 0) * v0, axis=(1, 2)) / v0.sum()
    return np.asarray(compute_density_error(f, mask), dtype=float), sig[:, ocean].astype(np.float32), lev0


def rmse_to_data(s) :
    return np.sqrt(np.mean((s.mean(0) - sigma_data.mean(0))**2))


def band_profile(section) :
    return np.nanmean(section[:, (latitude >= BAND[0]) & (latitude < BAND[1])], axis=1)


def plot_section(ax, section, title) :
    """Contour a (depth, lat) section on the stretched depth axis, BAND profile as inset (dashed : data)."""
    y = stretch(depth)
    cf = ax.contourf(latitude, y, section, levels=LEVELS, cmap=cm.deep_r, extend='both')
    ax.contour(latitude, y, section, levels=LEVELS, colors='k', linewidths=0.6)
    for lat in BAND :
        ax.axvline(x=lat, color='gray', linestyle='dashed', linewidth=1)
    ax.set_title(title, loc='left')
    ax.set_ylim(stretch(DEPTH_MAX), 0)
    ax.set_yticks(stretch(DEPTH_TICKS))
    ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
    inset_ax = inset_axes(ax, width='15%', height='100%', loc='right', borderpad=0)
    inset_ax.plot(band_profile(section), y, color='black', linewidth=2)
    inset_ax.plot(band_profile(section_data), y, color='black', linewidth=1, linestyle='--')
    inset_ax.set_facecolor('gray')
    inset_ax.patch.set_alpha(0.4)
    inset_ax.set_xlim(PROFILE_RANGE)
    inset_ax.set_ylim(ax.get_ylim())
    inset_ax.set_yticks(ax.get_yticks())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    return cf


none = np.load(f'{NONE_B8}/generated.npy')
E_data, sigma_data, lev0_data = stats(batch)
section_data = np.nanmean(zonal_sections(batch, mask), axis=0)
rng = np.random.default_rng(0)
rmse_data = np.mean([rmse_to_data(sigma_data[rng.choice(len(sigma_data), len(none), replace=False)]) for _ in range(N_SUBSETS)])
curves = {'C7' : {0. : none} | c7_runs(),
          'C7 + surface' : {0. : none} | {float(e) : np.load(surf_run(e) / 'generated.npy') for e in ETAS
                                          if (surf_run(e) / 'generated.npy').exists()}}
res = {}
for k, runs in curves.items() :
    res[k] = {}
    for eta, x in runs.items() :
        E, sig, lev0 = stats(x)
        res[k][eta] = dict(E=E, rmse=rmse_to_data(sig), bias0=lev0.mean() - lev0_data.mean())
        print(f'{k:13s} eta {eta:5g} : E {E.mean():.2f} +/- {E.std(ddof=1):.2f}, RMSE {res[k][eta]["rmse"]:.3f}, '
              f'bias lev0 {res[k][eta]["bias0"]:+.2f}')
print(f'data : E {E_data.mean():.2f}, RMSE of b{len(none)} subsets {rmse_data:.3f}')

frieze = [(none, 'No constraint')] + [(curves['C7 + surface'][float(e)], rf'C7 $\eta = {e:g}$ + surface') for e in ETAS_FRIEZE
                                      if float(e) in curves['C7 + surface']]
fig = plt.figure(figsize=(9, 6.4))
gs = fig.add_gridspec(3, len(frieze), height_ratios=[1, 0.6, 1.25], hspace=0.55, wspace=0.08)
ticks = [0, 10, 30, 100, 300]


def eta_axis(ax) :
    ax.set_xscale('symlog', linthresh=10, linscale=0.5)
    ax.set_xlim(-1.5, 360)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{e:g}' for e in ticks])
    ax.minorticks_off()


ax = fig.add_subplot(gs[0, :])
ax_rmse = ax.twinx()
for k, c, mk in [('C7', COLOR_C7, 'o'), ('C7 + surface', COLOR_SURF, 'o')] :
    etas = np.asarray(list(res[k]))
    mean = np.asarray([r['E'].mean() for r in res[k].values()])
    std = np.asarray([r['E'].std(ddof=1) for r in res[k].values()])
    ax.fill_between(etas, mean - std, mean + std, color=c, alpha=0.2, linewidth=0)
    ax.plot(etas, mean, f'{mk}-', color=c, markersize=3, label=f'{k}: density error')
    ax_rmse.plot(etas, [r['rmse'] for r in res[k].values()], 's--', color=c, markersize=3, linewidth=1, label=f'{k}: RMSE')
ax.axhline(E_data.mean(), color='gray', linestyle=':', linewidth=1)
ax_rmse.axhline(rmse_data, color='gray', linestyle='-.', linewidth=0.8)
ax.set_ylabel('density error [%] (solid)')
ax_rmse.set_ylabel(r'RMSE [kg m$^{-3}$] (dashed)')
ax.set_ylim(0, 6)
ax_rmse.set_ylim(bottom=0)
eta_axis(ax)
ax.set_xlabel(r'C7 $\eta$')
ax.set_title('a) Density error and distance to the data', loc='left')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_rmse.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False, ncol=2)

ax = fig.add_subplot(gs[1, :])
for k, c in [('C7', COLOR_C7), ('C7 + surface', COLOR_SURF)] :
    ax.plot(list(res[k]), [r['bias0'] for r in res[k].values()], 'o-', color=c, markersize=3, label=k)
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.set_ylabel(r'bias level 0 [kg m$^{-3}$]')
eta_axis(ax)
ax.set_title(r'b) Surface $\sigma_0$ bias to the data', loc='left')
ax.legend(loc='lower left', frameon=False)

frieze_axes = []
for j, (x, title) in enumerate(frieze) :
    ax = fig.add_subplot(gs[2, j])
    frieze_axes.append(ax)
    cf = plot_section(ax, np.nanmean(zonal_sections(torch.as_tensor(x), mask), axis=0), f'{chr(ord("c") + j)}) {title}')
    ax.set_xlabel('latitude [°N]')
    if j == 0 :
        ax.set_ylabel('depth [m]')
    else :
        ax.set_yticklabels([])
cbar = fig.colorbar(cf, ax=frieze_axes, label=r'$\sigma_0$ [kg m$^{-3}$ - 1000]', fraction=0.03, pad=0.02)
cbar.ax.invert_yaxis()

(HERE / 'figures').mkdir(exist_ok=True)
for ext in ['pdf', 'png'] :
    fig.savefig(HERE / 'figures' / f'fig_strength_surface.{ext}', bbox_inches='tight', dpi=200)
