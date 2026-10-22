"""Figure 6: depth-space meridional overturning stream function of the a posteriori runs in the 1/4° NEMO model, one
panel per configuration of `NAMES` whose EXPERIMENT run is collected, averaged over the last YEARS years of the run.

    psi(z, y) = sum_{k' <= k} sum_x <v e3v> e1v  [Sv]

<v e3v> is `voce_e3v`, the time mean of the thickness-weighted meridional velocity written by XIOS, so the transport is
exact under zstar vvl without any e3v. Grids and masks come from the nemo_cookbook NEMODataTree of the run, built with
linssh=True only because the output holds no time-varying e3v (it is not used: integrating over i weights by e1v). The basin is integrated whole (DINO has a single basin) and cumulated from the
surface down. The y axis is the nominal level depth `depthv`: DINO is in s-coordinates, so below ~500 m the actual
depth of a level is shallower where the bottom is shallower. A run shorter than YEARS is averaged over its full length.
"""
from common import *
from nemo_cookbook import NEMODataTree

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

import cmocean.cm as cm
from settings import COLLECT, EXP, REFERENCES_REL, aposteriori_dirs, experiment_paths
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7})

YEARS = 1 / 12            # length of the average, taken at the end of each run (1 / 12: last month)
EXPERIMENT = 'daily_2m'   # EXPERIMENTS key of a_posteriori/settings.py
DAYS_PER_YEAR = 360       # 360_day calendar of DINO
NCOLS = 4
PSI_PERCENTILE = 99       # half range of the colour scale: this percentile of |psi| over all the runs
PSI_STEP = 4.             # contour interval [Sv], zero contour left out (noisy in weak-flow regions)
DOMAIN = f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/restart25_arch/domain_cfg_out.nc'
DEPTH_POWER, DEPTH_MAX = 0.625, 4000
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000, 4000]
stretch = lambda d: np.asarray(d) ** DEPTH_POWER


def mean_moc(results, years=YEARS):
    """Stream function [Sv] (k, j) of the V-grid output of `results`, averaged over the last `years` years.

    Returns (psi, length of the average in years, zonal mean latitude of the V points), psi NaN on levels with no wet
    V point along the zonal line.
    """
    path, = glob(f'{results}/{EXP}_*_grid_V_3D.nc')
    nemo = NEMODataTree.from_paths({'parent': {'domain': DOMAIN, 'gridV': path}}, iperio=True, linssh=True,
                                   decode_times=False)
    grid = nemo['gridV']
    end, start = grid['time_counter_bounds'].values[-1, 1], grid['time_counter_bounds'].values[0, 0]
    window = min(years * DAYS_PER_YEAR * 86400, end - start)
    if window < years * DAYS_PER_YEAR * 86400:
        print(f'WARNING : {path} covers {window / 86400:g} days only, averaged over its full length')
    ve3 = nemo['gridV/voce_e3v'].isel(time_counter=grid['time_counter'].values > end - window).mean('time_counter')
    psi = 1e-6 * ve3.integral(dims=['i']).data.cumsum('k')
    return (psi.where(grid['vmask'].any('i')).compute(), window / (DAYS_PER_YEAR * 86400),
            grid['gphiv'].mean('i').values)


folders = aposteriori_dirs(constraints=NAMES)
mocs, lengths = {}, {}
for method in NAMES:
    folder = folders.get(method)
    results = folder and f'{folder}/{experiment_paths(EXPERIMENT)["results"]}'
    if folder is None or not os.path.exists(f'{results}/{COLLECT}'):
        print(f'WARNING : no collected {EXPERIMENT} run for {method}, skipped')
        continue
    mocs[method], lengths[method], lat = mean_moc(results)
    psi = mocs[method]
    print(f'{method} : {results}, mean of {lengths[method]:g} years, psi in [{float(psi.min()):.1f}, {float(psi.max()):.1f}] Sv')

half_range = float(np.nanpercentile(np.abs(np.stack([m.values for m in mocs.values()])), PSI_PERCENTILE))
half_range = PSI_STEP * np.ceil(half_range / PSI_STEP)
levels = np.arange(-half_range, half_range + PSI_STEP / 2, PSI_STEP)
levels = levels[np.abs(levels) > PSI_STEP / 2]

nrows = -(-len(mocs) // NCOLS)
fig, axes = plt.subplots(nrows, NCOLS, figsize=(11, 1.9 * nrows + 0.5), sharex=True, sharey=True, squeeze=False,
                         gridspec_kw=dict(wspace=0.08, hspace=0.3))
for ax, (method, psi) in zip(axes.flat, mocs.items()):
    depth = stretch(psi.depthv.values)
    pm = ax.pcolormesh(lat, depth, psi.values, cmap=cm.balance, vmin=-half_range, vmax=half_range,
                       shading='nearest', rasterized=True)
    ax.contour(lat, depth, psi.values, levels=levels, colors='k', linewidths=0.3)
    ax.set_title(NAMES[method], loc='left', fontsize=7)
for i, ax in enumerate(axes.flat):
    if i >= len(mocs):
        ax.set_visible(False)
        continue
    ax.set_ylim(stretch(DEPTH_MAX), 0)
    ax.set_yticks(stretch(DEPTH_TICKS))
    ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
    if i % NCOLS == 0:
        ax.set_ylabel('depth [m]')
    # Last panel of each column: it may sit above an empty slot of the last row.
    if i + NCOLS >= len(mocs):
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel('latitude [°N]')
length = min(lengths.values())
period = f'{length:g} year{"s" if length > 1 else ""}' if length >= 1 else f'{length * DAYS_PER_YEAR:g} days'
fig.colorbar(pm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.12, aspect=50,
             label=rf'$\psi_z$ [Sv], mean of the last {period}')

os.makedirs(FIGURES, exist_ok=True)
fig.savefig(f'{FIGURES}/fig6_amoc.pdf', bbox_inches='tight')
plt.show()
