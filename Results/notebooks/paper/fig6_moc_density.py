"""Figure 6, density-space counterpart: meridional overturning stream function of the a posteriori runs in the 1/4° NEMO
model in sigma_2 classes, shown at its pseudo-depth, one panel per configuration of `NAMES` whose EXPERIMENT run is
collected, averaged over the last YEARS years of the run. Computed with the dinostics tools (Results/a_posteriori/dinostics).

    psi(rho, y) = sum_{rho' <= rho} sum_x T(rho', y, x)  [Sv]

T(rho', y, x) is the transport <v e3v> e1v of `voce_e3v` (exact under zstar vvl), remapped conservatively by xgcm onto the
density classes of `GridManipulation.transform_to_density`, rho = S-EOS density referenced at z = 2000 m (rho_ref =
RHO_REF) of the daily toce, soce interpolated on the V points. Its class edges are fixed, from rho_ref - 4 to rho_ref +
1.67 kg m-3: water outside them is dropped from the transform. `Diagnostics.get_MOC_pseudodepth` maps each class to the
depth z* where the volume of the zonal line below z* (from the bathymetry of e3v_0) equals the volume denser than the
class:

    sum_x area (H - z*)^+ = sum_x sum_{rho' >= rho} V(rho', y, x)

The MOC and its pseudo-depth are computed for every daily output of the last YEARS years and averaged over time in
density space. The y axis of the main figures is the pseudo-depth, stretched as fig6_amoc.py; the sigma figures show
psi against the class index, labelled in sigma_2 = rho - 1000. Domain, mesh mask and namelist (S-EOS coefficients) are
those of the 1/4° reference restart, as the a posteriori runs do not collect theirs. A run shorter than YEARS is averaged
over its full length.
"""
from common import *
import xnemogcm as xn

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')
sys.path.append(f'{PRP}/Results/a_posteriori/dinostics/')

import cmocean.cm as cm
from experiment import NemoExperiment
from settings import COLLECT, EXP, REFERENCES_REL, aposteriori_dirs, experiment_paths
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

YEARS = 1 / 12            # length of the average, taken at the end of each run (1 / 12: last month)
EXPERIMENT = 'daily_2m'   # EXPERIMENTS key of a_posteriori/settings.py
DAYS_PER_YEAR = 360       # 360_day calendar of DINO
NCOLS = 4
DENSITY = DENSITY_RUN
ROW_METHODS = ['data', 'none', DENSITY, 'zero_mean_border0']  # aposteriori/moc_density.pdf, aposteriori/moc_sigma.pdf
PSI_PERCENTILE = 99       # half range of the colour scale: this percentile of |psi| over all the runs
PSI_STEP = 4.             # contour interval [Sv], zero contour left out (noisy in weak-flow regions)
Z_REF, RHO_REF = 2000., 1035.
REFERENCE = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/'
REFERENCE_RUN = 'restart25_arch'  # domain_cfg_out.nc, mesh_mask.nc and namelist_cfg of the 1/4° reference
DEPTH_POWER, DEPTH_MAX = 0.625, 4000
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000, 4000]
SIGMA_TICKS = [31, 32, 33, 34, 35, 35.5, 36, 36.25, 36.5]
stretch = lambda d: np.asarray(d) ** DEPTH_POWER


def experiment(results):
    """dinostics NemoExperiment of the reference grid holding the T_3D and V_3D daily outputs of `results`.

    get_MOC_pseudodepth calls transform_to_density with `levels=200`, which it does not take, and without T, S, so
    get_rho would read `inst_T_3D`: its transform_to_density is wrapped to drop `levels` and pass the run's toce, soce.
    """
    exp = NemoExperiment(REFERENCE, REFERENCE_RUN, restarts=[''], files=[])
    for name in ('T_3D', 'V_3D'):
        # The `EXP` prefix keeps out the ._* AppleDouble files of the external volume, which `*_grid_*` matches.
        exp.data[name] = xn.open_nemo(domcfg=exp.domain, files=sorted(glob(f'{results}/{EXP}_*_grid_{name}.nc')))
    transform, grid_T = exp.grid_manipulation.transform_to_density, exp.data['T_3D']
    exp.grid_manipulation.transform_to_density = (
        lambda var, levels=None, T=grid_T.toce, S=grid_T.soce, rho_ref=RHO_REF, **kwargs:
        transform(var=var, T=T, S=S, rho_ref=rho_ref, **kwargs))
    return exp


@memo
def mean_moc_density(results, years=YEARS):
    """Density-space stream function [Sv] (y_f, rho) of the daily outputs of `results` and pseudo-depth [m] (y_f, rho)
    of its classes, both averaged over the last `years` years.

    Returns (psi, pseudo-depth, length of the average in years, zonal mean latitude of the V points).
    """
    exp = experiment(results)
    V = exp.data['V_3D']
    end, start = V['t_bounds'].values[-1, 1], V['t_bounds'].values[0, 0]
    total = (end - start).total_seconds()
    window = min(years * DAYS_PER_YEAR * 86400, total)
    if window < years * DAYS_PER_YEAR * 86400:
        print(f'WARNING : {results} covers {window / 86400:g} days only, averaged over its full length')
    last = int(sum((end - t).total_seconds() < window for t in V['t'].values))
    # 'vocetr_eff' is the name get_MOC_pseudodepth takes as a transport already, voce_e3v holding e3v.
    transport = (V['voce_e3v'] * exp.domain.e1v).rename('vocetr_eff')
    e3v = exp.domain.e3v_0.expand_dims(t=V['t'])
    psi, depth = 0, 0
    # One day per call: over several days the (t, y, rho, z) mask of get_MOC_pseudodepth takes tens of GB.
    for t in range(V.sizes['t'] - last, V.sizes['t']):
        moc, z = exp.diagnostics.get_MOC_pseudodepth(transport, e3v, isel={'t': t}, z=Z_REF)
        psi, depth = psi + moc.compute() / last, depth + z.compute() / last
    order = lambda a: a.drop_vars('t', errors='ignore').transpose('y_f', 'rho')
    return order(psi), order(depth), window / (DAYS_PER_YEAR * 86400), exp.domain.gphiv.mean('x_c').values


def scale(mocs):
    """Half range of the colour scale and contour levels of the stream functions `mocs`."""
    half_range = float(np.nanpercentile(np.abs(np.stack([m.values for m in mocs])), PSI_PERCENTILE))
    half_range = PSI_STEP * np.ceil(half_range / PSI_STEP)
    levels = np.arange(-half_range, half_range + PSI_STEP / 2, PSI_STEP)
    return half_range, levels[np.abs(levels) > PSI_STEP / 2]


def panel(ax, method, half_range, levels, space):
    """Draw psi of `method` on `ax`, at its stretched pseudo-depth (`space` 'depth') or against its class index ('sigma')."""
    psi = mocs[method].values
    if space == 'depth':
        y, z = np.broadcast_to(lat[:, None], psi.shape), stretch(depths[method].values)
    else:
        y, z = lat, np.arange(psi.shape[1])
        psi = psi.T
    pm = ax.pcolormesh(y, z, psi, cmap=cm.balance, vmin=-half_range, vmax=half_range, shading='nearest',
                       rasterized=True)
    ax.contour(y, z, psi, levels=levels, colors='k', linewidths=0.3)
    return pm


def vertical_axis(ax, space, label):
    """Inverted vertical axis of `space` ('depth' or 'sigma'), labelled when `label`."""
    if space == 'depth':
        ax.set_ylim(stretch(DEPTH_MAX), 0)
        ax.set_yticks(stretch(DEPTH_TICKS))
        ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
    else:
        sigma = rho - 1000
        ax.set_ylim(len(sigma) - 0.5, -0.5)
        ax.set_yticks(np.interp(SIGMA_TICKS, sigma, np.arange(len(sigma))))
        ax.set_yticklabels([f'{s:g}' for s in SIGMA_TICKS])
    if label:
        ax.set_ylabel('pseudo-depth [m]' if space == 'depth' else r'$\sigma_2$ [kg m$^{-3}$]')


folders = aposteriori_dirs(constraints=NAMES)
mocs, depths, lengths = {}, {}, {}
for method in NAMES:
    folder = folders.get(method)
    results = folder and f'{folder}/{experiment_paths(EXPERIMENT)["results"]}'
    if folder is None or not os.path.exists(f'{results}/{COLLECT}'):
        print(f'WARNING : no collected {EXPERIMENT} run for {method}, skipped')
        continue
    try:
        mocs[method], depths[method], lengths[method], lat = mean_moc_density(results)
    except (OSError, ValueError, KeyError) as error:
        print(f'WARNING : {method} : {results} unreadable ({error!r}), skipped')
        continue
    psi = mocs[method]
    print(f'{method} : {results}, mean of {lengths[method]:g} years, psi in [{float(psi.min()):.1f}, '
          f'{float(psi.max()):.1f}] Sv, pseudo-depth in [{float(depths[method].min()):.0f}, '
          f'{float(depths[method].max()):.0f}] m')
rho = next(iter(mocs.values()))['rho'].values

length = min(lengths.values())
period = f'{length:g} year{"s" if length > 1 else ""}' if length >= 1 else f'{length * DAYS_PER_YEAR:g} days'
label = rf'$\psi_\sigma$ [Sv], mean of the last {period}'
sources = [f'{REFERENCE}{REFERENCE_RUN}']
row = [m for m in ROW_METHODS if m in mocs]
os.makedirs(FIGURES, exist_ok=True)
for space, grid_file, row_file in [('depth', 'fig6_moc_density.pdf', 'moc_density.pdf'),
                                   ('sigma', 'fig6_moc_sigma.pdf', 'moc_sigma.pdf')]:
    half_range, levels = scale(mocs.values())
    nrows = -(-len(mocs) // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(11, 1.9 * nrows + 0.5), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.3))
    for ax, method in zip(axes.flat, mocs):
        pm = panel(ax, method, half_range, levels, space)
        ax.set_title(NAMES[method], loc='left', fontsize=7)
    for i, ax in enumerate(axes.flat):
        if i >= len(mocs):
            ax.set_visible(False)
            continue
        vertical_axis(ax, space, i % NCOLS == 0)
        # Last panel of each column: it may sit above an empty slot of the last row.
        if i + NCOLS >= len(mocs):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('latitude [°N]')
    fig.colorbar(pm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.12, aspect=50, label=label)
    save(fig, f'{FIGURES}/{grid_file}', sources=[*sources, *(folders[m] for m in mocs)], bbox_inches='tight')

    half_range, levels = scale(mocs[m] for m in row)
    fig, axs = plt.subplots(1, len(row), figsize=(2.75 * len(row), 1.9), sharey=True, squeeze=False,
                            gridspec_kw={'wspace': 0.05})
    for j, method in enumerate(row):
        ax = axs[0, j]
        pm = panel(ax, method, half_range, levels, space)
        ax.set_title(f'{chr(ord("a") + j)}) {NAMES[method]}', loc='left')
        vertical_axis(ax, space, j == 0)
        ax.set_xlabel('latitude [°N]')
    fig.colorbar(pm, ax=axs, fraction=0.03, pad=0.02, label=label)
    save(fig, f'{FIGURES}/aposteriori/{row_file}', sources=[*sources, *(folders[m] for m in row)], bbox_inches='tight')
plt.show()
