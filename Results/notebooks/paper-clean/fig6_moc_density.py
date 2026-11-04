"""Figure 6, density-space counterpart: meridional overturning stream function of the a posteriori runs in the 1/4° NEMO
model in sigma_2 classes, shown at its pseudo-depth, one panel per configuration of `NAMES` whose EXPERIMENT run is
collected, averaged over the last YEARS years of the run. Computed with `Diagnostics.get_MOC_pseudodepth`.

    psi(rho, y) = sum_{rho' <= rho} sum_x T(rho', y, x)  [Sv]

T(rho', y, x) is the transport <v e3v> e1v of `voce_e3v`, remapped conservatively by xgcm onto the density classes of
`GridManipulation.transform_to_density`, rho = S-EOS density referenced at z = Z_REF (rho_ref = RHO_REF) of the daily
toce, soce interpolated on the V points. Its class edges are fixed, from rho_ref - 4 to rho_ref + 1.67 kg m-3: water
outside them is dropped from the transform. get_MOC_pseudodepth maps each class to the depth z* where the volume of the
zonal line below z* (from the bathymetry of e3v_0) equals the volume denser than the class:

    sum_x area (H - z*)^+ = sum_x sum_{rho' >= rho} V(rho', y, x)

The MOC and its pseudo-depth are computed for every daily output of the last YEARS years and averaged over time in
density space. The y axis of the depth figures is the pseudo-depth, stretched as fig6_amoc.py; the sigma figures show
psi against the class index, labelled in sigma_2 = rho - 1000. The averaging (YEARS of the last outputs), colour scale
and panel layout are those of `fig6_amoc.py`.
"""
from common import *

from fig6_amoc import (DEPTH_MAX, DEPTH_TICKS, EXPERIMENT, NCOLS, ROW_METHODS, YEARS, last_outputs, panel_grid, period,
                       scale)

Z_REF, RHO_REF = 2000., 1035.
SIGMA_TICKS = [31, 32, 33, 34, 35, 35.5, 36, 36.25, 36.5]
FILES = {'depth': ('fig6_moc_density', 'aposteriori/moc_density'), 'sigma': ('fig6_moc_sigma', 'aposteriori/moc_sigma')}


def experiment(results):
    """`aposteriori_experiment` of the T_3D and V_3D outputs of `results`, set up for get_MOC_pseudodepth.

    get_MOC_pseudodepth calls transform_to_density with `levels=200`, which it does not take, and without T, S, so
    get_rho would read `inst_T_3D`: its transform_to_density is wrapped to drop `levels` and pass the run's toce, soce.
    """
    exp = aposteriori_experiment(results, files=('T_3D', 'V_3D'))
    transform, T = exp.grid_manipulation.transform_to_density, exp.data['T_3D']
    exp.grid_manipulation.transform_to_density = (
        lambda var, levels=None, **kwargs: transform(var=var, T=T.toce, S=T.soce, rho_ref=RHO_REF, **kwargs))
    return exp


def mean_moc_density(results, years=YEARS):
    """Density-space stream function [Sv] (y_f, rho) of the daily outputs of `results` and pseudo-depth [m] (y_f, rho)
    of its classes, both averaged over the last `years` years, and the length of the average in years."""
    exp = experiment(results)
    V = exp.data['V_3D']
    last, length = last_outputs(V, years)
    # 'vocetr_eff' is the name get_MOC_pseudodepth takes as a transport already, voce_e3v holding e3v.
    transport = (V.voce_e3v * exp.domain.e1v).rename('vocetr_eff')
    e3v = exp.domain.e3v_0.expand_dims(t=V.t)
    psi, depth = 0, 0
    # One day per call: over several days the (t, y, rho, z) mask of get_MOC_pseudodepth takes tens of GB.
    for t in range(V.sizes['t'] - last, V.sizes['t']):
        moc, z = exp.diagnostics.get_MOC_pseudodepth(transport, e3v, isel={'t': t}, z=Z_REF)
        psi, depth = psi + moc.compute() / last, depth + z.compute() / last
    order = lambda a: a.drop_vars('t', errors='ignore').transpose('y_f', 'rho')
    return order(psi), order(depth), length


def panel(ax, psi, depth, lat, half_range, levels, space):
    """Draw `psi` (y, rho) on `ax`, at its stretched pseudo-depth `depth` (`space` 'depth') or against its class index
    ('sigma')."""
    if space == 'depth':
        y, z, values = np.broadcast_to(lat[:, None], psi.shape), stretch(depth.values), psi.values
    else:
        y, z, values = lat, np.arange(psi.shape[1]), psi.values.T
    pm = ax.pcolormesh(y, z, values, cmap='cmo.balance', vmin=-half_range, vmax=half_range, shading='nearest',
                       rasterized=True)
    ax.contour(y, z, values, levels=levels, colors='k', linewidths=0.3)
    return pm


def vertical_axis(ax, space, sigma, label):
    """Inverted vertical axis of `space` ('depth' or 'sigma', classes `sigma`), labelled when `label`."""
    if space == 'depth':
        depth_axis(ax, DEPTH_MAX, DEPTH_TICKS)
    else:
        ax.set_ylim(len(sigma) - 0.5, -0.5)
        ax.set_yticks(np.interp(SIGMA_TICKS, sigma, np.arange(len(sigma))))
        ax.set_yticklabels([f'{s:g}' for s in SIGMA_TICKS])
    if label:
        ax.set_ylabel('pseudo-depth [m]' if space == 'depth' else r'$\sigma_2$ [kg m$^{-3}$]')


def compute():
    """Density-space stream function and pseudo-depth of the collected EXPERIMENT runs of NAMES, averaged over their
    last YEARS years: {'folders', 'mocs', 'depths': {method: (y_f, rho)}, 'lengths': {method: years}, 'sigma': sigma_2
    of the classes, 'lat'}."""
    lat = aposteriori_experiment().domain.gphiv.mean('x_c').values
    folders = collected_runs(NAMES, EXPERIMENT)
    mocs, depths, lengths = {}, {}, {}
    for method, folder in folders.items():
        results = f'{folder}/{experiment_paths(EXPERIMENT)["results"]}'
        mocs[method], depths[method], lengths[method] = mean_moc_density(results)
        print(f'{method} : {results}, mean of {lengths[method]:g} years, psi in [{float(mocs[method].min()):.1f}, '
              f'{float(mocs[method].max()):.1f}] Sv, pseudo-depth in [{float(depths[method].min()):.0f}, '
              f'{float(depths[method].max()):.0f}] m')
    sigma = next(iter(mocs.values()))['rho'].values - 1000
    return {'folders': folders, 'mocs': mocs, 'depths': depths, 'lengths': lengths, 'sigma': sigma, 'lat': lat}


def plot(computed):
    """Depth and sigma figures of all the runs and of ROW_METHODS of `computed` (`compute`): {file name in FIGURES,
    without extension, of FILES: figure}."""
    mocs, depths, lengths, sigma, lat = (computed[k] for k in ('mocs', 'depths', 'lengths', 'sigma', 'lat'))
    label = rf'$\psi_\sigma$ [Sv], mean of the last {period(lengths.values())}'
    row = [m for m in ROW_METHODS if m in mocs]

    figs = {}
    for space, (grid_file, row_file) in FILES.items():
        half_range, levels = scale(mocs.values())
        fig, axes = panel_grid(len(mocs))
        for i, (ax, method) in enumerate(zip(axes.flat, mocs)):
            pm = panel(ax, mocs[method], depths[method], lat, half_range, levels, space)
            ax.set_title(NAMES[method], loc='left', fontsize=7)
            vertical_axis(ax, space, sigma, i % NCOLS == 0)
        fig.colorbar(pm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.12, aspect=50, label=label)
        figs[grid_file] = fig

        half_range, levels = scale(mocs[m] for m in row)
        fig, axs = plt.subplots(1, len(row), figsize=(2.75 * len(row), 1.9), sharey=True, squeeze=False,
                                gridspec_kw={'wspace': 0.05})
        for j, method in enumerate(row):
            pm = panel(axs[0, j], mocs[method], depths[method], lat, half_range, levels, space)
            axs[0, j].set_title(f'{panel_letter(j)}) {NAMES[method]}', loc='left')
            vertical_axis(axs[0, j], space, sigma, j == 0)
            axs[0, j].set_xlabel('latitude [°N]')
        fig.colorbar(pm, ax=axs, fraction=0.03, pad=0.02, label=label)
        figs[row_file] = fig
    return figs


def main():
    """Compute, plot and save the figures in FIGURES."""
    computed = compute()
    figs, folders = plot(computed), computed['folders']
    row = [folders[m] for m in ROW_METHODS if m in folders]
    for grid_file, row_file in FILES.values():
        save(figs[grid_file], f'{FIGURES}/{grid_file}.pdf', sources=list(folders.values()), bbox_inches='tight')
        save(figs[row_file], f'{FIGURES}/{row_file}.pdf', sources=row, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
