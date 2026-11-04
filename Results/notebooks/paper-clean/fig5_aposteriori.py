"""Figure 5: a posteriori behaviour of the generated states in the 1/4° NEMO model, recomputed from the NEMO outputs of
the runs of `notebooks/a_posteriori/` (experiment EXPERIMENT, run started from the median member of each generation),
on the dinostics `NemoExperiment` of the 1/4° reference grid.

Left, one row per method of METHODS:
    zonal mean sigma_0 (TEOS-10, `common.sigma0`, volume-weighted with the time-varying e3t of the outputs) after
    2 months (last daily output), with its channel profile as inset (as fig1),
    its difference to the initial state (day 0, the restart the run starts from, e3t_0),
    its difference to the reference density, the mean zonal sigma_0 section of the data states (`batch`, 1° grid, the 1/4° section
    interpolated onto its latitudes, both grids share their levels).
Right, over time: a) transport through the re-entrant channel, `Diagnostics.get_ACC` (uoce e3u_0 e2u through the first
U column), b) RMSE of the zonal sigma_0 section to the reference density (grey band: RMSE of each data state to it,
min-max), c) empty.
Also saved in the `aposteriori` subfolder of FIGURES: density_2m, the zonal sigma_0 after 2 months of each method of
ROW_METHODS side by side, as fig1. Dashed: the 1/4° spin-up restart. A method whose run is not collected is skipped.
"""
from common import *

EXPERIMENT = 'daily_2m'
METHODS = {'data': 'Data',
           'none': 'No constraint',
           DENSITY_RUN: 'Increasing density',
           'zero_mean_border0': 'T,S mean constraint'}
COLORS = ['k', 'C0', 'C1', 'C2']
ROW_METHODS = ['none', DENSITY_RUN, 'zero_mean_border0']
DIFF_PERCENTILE = 99  # half range of each difference colour scale: this percentile of |difference| over all the runs
DEPTH_MAX, DEPTH_TICKS = 3500, [0, 100, 250, 500, 1000, 2000, 3000]
SIGMA_LABEL = r'$\sigma_0$ [kg m$^{-3}$ - 1000]'


def restart(path):
    """Zonal sigma_0 section (z_c, y_c) and channel transport [Sv] of the 1/4° restart `path`."""
    exp = aposteriori_experiment()
    state = restart_state(exp, path)
    exp.data['U_3D'] = state
    section = zonal_mean(sigma0(state, exp), cell_volume(exp))
    return section.transpose('z_c', 'y_c').values, float(exp.diagnostics.get_ACC().values[0])


def run(folder):
    """Days from the start, zonal sigma_0 sections (frames, z_c, y_c) and channel transports [Sv] of the EXPERIMENT run
    in `folder`. Frame 0 is the restart the run starts from (instantaneous), frame i >= 1 the i-th daily mean."""
    paths = experiment_paths(EXPERIMENT)
    exp = aposteriori_experiment(f'{folder}/{paths["results"]}', files=('T_3D', 'U_3D'))
    T = exp.data['T_3D']
    days = [(t - T.t_bounds.values[0, 0]).total_seconds() / 86400 for t in T.t.values]
    sections = zonal_mean(sigma0(T, exp), cell_volume(exp, T.e3t)).transpose('t', 'z_c', 'y_c').values
    section0, transport0 = restart(f'{folder}/{paths["restart_1_4deg"]}')
    return (np.r_[0., days], np.concatenate([section0[None], sections]),
            np.r_[transport0, exp.diagnostics.get_ACC().values])


def to_lat(section, lat, lat_to):
    """Section (z, y) on `lat` linearly interpolated onto `lat_to`, NaN where a neighbour is NaN."""
    def interp(a):
        finite = np.isfinite(a)
        v = np.interp(lat_to, lat, np.where(finite, a, 0.))
        w = np.interp(lat_to, lat, finite.astype(float))
        return np.where(w > 1 - 1e-6, v, np.nan)
    return np.stack([interp(row) for row in section])


def plot_diff(ax, diff, lat, depth, half_range, title):
    """Pseudocolour of a (depth, lat) difference on the stretched depth axis, symmetric scale of half range `half_range`."""
    pm = ax.pcolormesh(lat, stretch(depth), diff, cmap='cmo.balance', vmin=-half_range, vmax=half_range,
                       shading='nearest', rasterized=True)
    ax.set_title(title, loc='left')
    return pm


def compute():
    """Zonal sigma_0 sections, their RMSE to the reference density and channel transports of the collected runs of
    METHODS and of the 1/4° spin-up, and the differences of the last section to day 0 and to the reference: {'folders',
    'spinup_path', 'runs': {method: {'days', 'sections', 'transport', 'rmse'}}, 'vs_day0', 'vs_reference': {method: (z,
    y)}, 'spinup_transport', 'spinup_rmse', 'data_rmse': (member,), 'lat', 'depth' (1/4°), 'lat_1', 'depth_1'
    (1°)}."""
    lat_1 = GRID.mask.gphit.mean('x_c').values
    data_sections = zonal_sections(batch)
    reference = np.nanmean(data_sections, axis=0)
    data_rmse = rmse(data_sections, reference)

    grid = aposteriori_experiment()
    lat, depth = grid.mask.gphit.mean('x_c').values, grid.mask.gdept_1d.values
    nz = reference.shape[0]
    rmse_to_reference = lambda section: float(rmse(to_lat(section[:nz], lat, lat_1)[None], reference)[0])
    spinup_path = sorted(glob(f'{REFERENCES}{GRID_1_4DEG}/{EXP}_*_restart.nc'))[-1]
    spinup_section, spinup_transport = restart(spinup_path)
    spinup_rmse = rmse_to_reference(spinup_section)
    print(f'spin-up {spinup_path} : transport {spinup_transport:.1f} Sv, RMSE {spinup_rmse:.3f}')

    folders = collected_runs(METHODS, EXPERIMENT)
    runs = {}
    for method, folder in folders.items():
        days, sections, transports = run(folder)
        runs[method] = dict(days=days, sections=sections, transport=transports,
                            rmse=np.array([rmse_to_reference(s) for s in sections]))
        print(f'{method} : {folder}, day {days[-1]:g}, RMSE to reference {runs[method]["rmse"][-1]:.3f}, '
              f'transport {transports[-1]:.1f} Sv')
    return {'folders': folders, 'spinup_path': spinup_path, 'runs': runs,
            'vs_day0': {m: r['sections'][-1] - r['sections'][0] for m, r in runs.items()},
            'vs_reference': {m: to_lat(r['sections'][-1][:nz], lat, lat_1) - reference for m, r in runs.items()},
            'spinup_transport': spinup_transport, 'spinup_rmse': spinup_rmse, 'data_rmse': data_rmse,
            'lat': lat, 'depth': depth, 'lat_1': lat_1, 'depth_1': depth[:nz]}


def plot(computed):
    """fig5 and aposteriori/density_2m of `computed` (`compute`): {file name in FIGURES, without extension: figure}."""
    runs, vs_day0, vs_reference = computed['runs'], computed['vs_day0'], computed['vs_reference']
    lat, depth, lat_1, depth_1 = computed['lat'], computed['depth'], computed['lat_1'], computed['depth_1']
    spinup_transport, spinup_rmse, data_rmse = (computed[k] for k in ('spinup_transport', 'spinup_rmse', 'data_rmse'))
    fig = plt.figure(figsize=(11, 1.75 * len(runs) + 0.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.15], wspace=0.2)
    left = outer[0].subgridspec(len(runs) + 2, 3, height_ratios=[1] * len(runs) + [0.12, 0.06], wspace=0.08,
                                hspace=0.25)
    right = outer[1].subgridspec(3, 1, hspace=0.7)
    axes = np.array([[fig.add_subplot(left[i, j]) for j in range(3)] for i in range(len(runs))])
    columns = [rf'$\sigma_0$, day {next(iter(runs.values()))["days"][-1]:g}', 'minus day 0', 'minus reference density']
    half_range = lambda diffs: float(np.nanpercentile(np.abs(np.stack(list(diffs.values()))), DIFF_PERCENTILE))
    range_day0, range_reference = half_range(vs_day0), half_range(vs_reference)
    print(f'difference half ranges : minus day 0 {range_day0:.3f}, minus reference {range_reference:.3f} kg m-3')

    for i, (method, r) in enumerate(runs.items()):
        titles = columns if i == 0 else [''] * 3
        cf = plot_section(axes[i, 0], r['sections'][-1], lat, depth, titles[0], depth_max=DEPTH_MAX,
                          ticks=DEPTH_TICKS, lines=0.4, profile_width=1.5)
        pm_day0 = plot_diff(axes[i, 1], vs_day0[method], lat, depth, range_day0, titles[1])
        pm_ref = plot_diff(axes[i, 2], vs_reference[method], lat_1, depth_1, range_reference, titles[2])
        axes[i, 0].set_ylabel(f'{METHODS[method]}\ndepth [m]')
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            depth_axis(ax, DEPTH_MAX, DEPTH_TICKS, labels=j == 0)
            if i < len(axes) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('latitude [°N]')
    for j, (mappable, label) in enumerate([(cf, SIGMA_LABEL), (pm_day0, r'$\Delta\sigma_0$ [kg m$^{-3}$]'),
                                           (pm_ref, r'$\Delta\sigma_0$ [kg m$^{-3}$]')]):
        cbar = fig.colorbar(mappable, cax=fig.add_subplot(left[-1, j]), orientation='horizontal', label=label)
        if j == 0:
            cbar.ax.invert_xaxis()

    series_axes = [fig.add_subplot(right[k]) for k in range(3)]
    for (k, spinup, title, ylabel), ax in zip(
            [('transport', spinup_transport, 'a) Zonal transport through the channel', 'transport [Sv]'),
             ('rmse', spinup_rmse, 'b) RMSE to the reference density', r'RMSE $\sigma_0$ [kg m$^{-3}$]')],
            series_axes):
        for method, color in zip(METHODS, COLORS):
            if method in runs:
                ax.plot(runs[method]['days'], runs[method][k], color=color, linewidth=1.2, label=METHODS[method])
        ax.axhline(spinup, color='gray', linestyle='--', linewidth=1, label='1/4° spin-up')
        ax.set_title(title, loc='left')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('time [days]')
        ax.set_xlim(0, max(r['days'][-1] for r in runs.values()))
    series_axes[1].axhspan(data_rmse.min(), data_rmse.max(), color='gray', alpha=0.25, linewidth=0)
    series_axes[1].set_ylim(bottom=0)
    series_axes[0].legend(frameon=False, loc='lower right')
    series_axes[2].axis('off')
    figs = {'fig5_aposteriori': fig}

    row = [m for m in ROW_METHODS if m in runs]
    fig, axs = plt.subplots(1, len(row), figsize=(3.4 * len(row), 1.9), sharey=True, squeeze=False,
                            gridspec_kw={'wspace': 0.05})
    for j, method in enumerate(row):
        cf = plot_section(axs[0, j], runs[method]['sections'][-1], lat, depth, f'{panel_letter(j)}) {METHODS[method]}',
                          depth_max=DEPTH_MAX, ticks=DEPTH_TICKS, lines=0.4, profile_width=1.5)
        axs[0, j].set_xlabel('latitude [°N]')
    axs[0, 0].set_ylabel('depth [m]')
    fig.colorbar(cf, ax=axs, label=SIGMA_LABEL, fraction=0.03, pad=0.02).ax.invert_yaxis()
    figs['aposteriori/density_2m'] = fig
    return figs


def main():
    """Compute, plot and save the figures in FIGURES."""
    computed = compute()
    figs, folders = plot(computed), computed['folders']
    save(figs['fig5_aposteriori'], f'{FIGURES}/fig5_aposteriori.pdf',
         sources=[computed['spinup_path'], *folders.values()], bbox_inches='tight')
    save(figs['aposteriori/density_2m'], f'{FIGURES}/aposteriori/density_2m.pdf',
         sources=[folders[m] for m in ROW_METHODS if m in folders], bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
