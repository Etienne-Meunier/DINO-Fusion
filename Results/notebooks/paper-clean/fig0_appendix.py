"""Figure 0, appendix: each layer of the unconstrained states has its own shift, as panel B of the "No constraint" tab
of `machine/density_viewer.html`, for FIELD.

Left: basin mean of FIELD (area-weighted mean of each level, `xgcm.Grid.average` of GRID over X and Y) at the levels
nearest to SHIFT_DEPTHS, one dot per state, N_STATES random data states and the unconstrained ones. Middle: correlation
over the unconstrained members of the basin means of every pair of levels. Right: its mean against the number of levels
apart, with the +-1/sqrt(n - 1) noise level of n members in grey."""
from common import *

from mpl_toolkits.axes_grid1 import make_axes_locatable

SOURCE = 'none'
SHIFT_DEPTHS = [5., 50., 100., 200., 500., 1000., 2000., 3300.]  # [m], the nearest levels are used
FIELD = 'toce'
COLORS = {'data': '#555', SOURCE: '#d95f02'}
UNITS = {'toce': '°C', 'soce': r'g kg$^{-1}$'}
LABELS = {'toce': 'T', 'soce': 'S'}
MAX_LAG = 12
N_STATES = 16


def compute():
    """Level means of FIELD of N_STATES random data states and of the SOURCE states, and their correlation between
    levels: {'depth', 'shift_levels', 'means': {source: (member, z)}, 'C': (z, z), 'lags', 'lag_mean', 'n', 'rng': the
    generator, continued by `plot` for the jitter}."""
    depth = GRID.mask.gdept_1d.values[:-1]
    shift_levels = sorted({int(np.abs(depth - d).argmin()) for d in SHIFT_DEPTHS})
    rng = np.random.default_rng(0)
    states = {'data': batch[rng.choice(len(batch), N_STATES, replace=False)], SOURCE: sources[SOURCE]}
    means = {s: GRID.grid.average(to_dataset(x)[FIELD], ['X', 'Y']).values[:, :-1] for s, x in states.items()}
    C = np.corrcoef(means[SOURCE].T)
    lags = np.arange(1, MAX_LAG + 1)
    lag_mean = [np.nanmean(np.diagonal(C, offset=l)) for l in lags]
    return {'depth': depth, 'shift_levels': shift_levels, 'means': means, 'C': C, 'lags': lags, 'lag_mean': lag_mean,
            'n': means[SOURCE].shape[0], 'rng': rng}


def plot(computed):
    """The three panels of `computed` (`compute`)."""
    depth, shift_levels, means, C = computed['depth'], computed['shift_levels'], computed['means'], computed['C']
    lags, lag_mean, n, rng = computed['lags'], computed['lag_mean'], computed['n'], computed['rng']
    plt.rcParams.update({'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [1.5, 1.2, 1], 'wspace': 0.6})

    ax = axs[0]
    for dy, source in [(-0.2, 'data'), (0.2, SOURCE)]:
        m = means[source][:, shift_levels]
        y = np.arange(len(shift_levels))[None] + dy + rng.uniform(-0.12, 0.12, m.shape)
        ax.scatter(m.ravel(), y.ravel(), s=5, color=COLORS[source], alpha=0.75, linewidths=0, label=NAMES[source])
    ax.set_yticks(range(len(shift_levels)))
    ax.set_yticklabels([f'{depth[k]:.0f} m' for k in shift_levels])
    ax.set_ylim(len(shift_levels) - 0.5, -0.5)
    ax.set_xlim(left=0)
    ax.set_xlabel(f'{LABELS[FIELD]} [{UNITS[FIELD]}]')
    ax.grid(True, axis='x', color='#e8e8e8', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, handletextpad=0.1, markerscale=1.8)
    ax.set_title(f'a) Basin mean of {LABELS[FIELD]}', loc='left')

    ax = axs[1]
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest', aspect='auto')
    ax.set_xticks(shift_levels[::2])
    ax.set_xticklabels([f'{depth[k]:.0f}' for k in shift_levels[::2]])
    ax.set_yticks(shift_levels)
    ax.set_yticklabels([f'{depth[k]:.0f}' for k in shift_levels])
    ax.set_xlabel('depth [m]')
    ax.set_ylabel('depth [m]', labelpad=1)
    cbar = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05),
                        ticks=np.linspace(-1, 1, 5))
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=2)
    ax.set_title('b) Correlation r', loc='left')

    ax = axs[2]
    noise = 1 / np.sqrt(n - 1)
    ax.axhspan(-noise, noise, color='black', alpha=0.08, linewidth=0, label=r'$\pm 1/\sqrt{n-1}$')
    ax.axhline(0, color='#999', linewidth=0.6)
    ax.plot(lags, lag_mean, '-o', color=COLORS[SOURCE], linewidth=1.2, markersize=2.5)
    ax.set_xlim(0.5, MAX_LAG + 0.5)
    ax.set_xticks(range(2, MAX_LAG + 1, 2))
    ax.set_ylim(-0.3, 1)
    ax.set_xlabel('levels apart')
    ax.set_ylabel('mean r', labelpad=1)
    ax.grid(True, color='#eee', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('c) Mean r by separation', loc='left')
    for ax in axs[[0, 2]]:
        ax.spines[['top', 'right']].set_visible(False)
    return fig


def main():
    """Compute, plot and save the figure as fig0_appendix.pdf in FIGURES."""
    save(plot(compute()), f'{FIGURES}/fig0_appendix.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
