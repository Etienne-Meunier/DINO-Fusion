"""Figure 1: zonal sigma_0 sections (TEOS-10, `common.sigma0`), with the batch-average profile over the
re-entrant channel latitudes (CHANNEL) as a black inset.

Main figure, split in two: batch-average section of each source in PAIRS, the two sources of a pair side by side
(fig1a, fig1b), saved in the `fig1` subfolder of FIGURES. Appendix: every source in CONFIGS['fig1_appendix'], one per
row, median member (upper median of the RMSE of its section to REFERENCE_DENSITY, the batch-mean section of the
200-state data batch, the criterion of the a posteriori selection) on the left and batch-average section on the
right."""
from common import *

PAIRS = {'fig1a': CONFIGS['fig1_main'][:2], 'fig1b': CONFIGS['fig1_main'][2:4]}
APPENDIX = CONFIGS['fig1_appendix']
OUT = f'{FIGURES}/fig1'
SIGMA_LABEL = r'$\sigma_0$ [kg m$^{-3}$ - 1000]'


def median_member(errors):
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


def compute():
    """Zonal sigma_0 sections of every source of fig1 and its median member to REFERENCE_DENSITY, printed: {'depth', 'lat', 'sections':
    {source: (member, z, y)}, 'medians': {source: member index}}."""
    depth = GRID.mask.gdept_1d.values[:-1]
    lat = GRID.mask.gphit.mean('x_c').values
    sections = {s: zonal_sections(sources[s]) for s in dict.fromkeys(CONFIGS['fig1_main'] + APPENDIX)}
    medians = {}
    for source, s in sections.items():
        errors = rmse(s, REFERENCE_DENSITY)
        medians[source] = median_member(errors)
        print(f'{source} : median member {medians[source]} / {len(errors)}, RMSE {errors[medians[source]]:.3f} '
              f'(mean {errors.mean():.3f})')
    return {'depth': depth, 'lat': lat, 'sections': sections, 'medians': medians}


def plot(computed):
    """fig1a, fig1b and fig1_appendix of `computed` (`compute`): {file name in the `fig1` subfolder, without
    extension: figure}."""
    depth, lat, sections, medians = computed['depth'], computed['lat'], computed['sections'], computed['medians']
    figs = {}
    for name, row in PAIRS.items():
        fig, axs = plt.subplots(1, len(row), figsize=(3.4 * len(row), 1.9), sharey=True, squeeze=False,
                                gridspec_kw={'wspace': 0.05})
        for j, s in enumerate(row):
            cf = plot_section(axs[0, j], np.nanmean(sections[s], axis=0), lat, depth, f'{panel_letter(j)}) {NAMES[s]}')
            axs[0, j].set_xlabel('latitude [°N]')
        axs[0, 0].set_ylabel('depth [m]')
        fig.colorbar(cf, ax=axs, label=SIGMA_LABEL, fraction=0.03, pad=0.02).ax.invert_yaxis()
        figs[name] = fig

    fig, axs = plt.subplots(len(APPENDIX), 2, figsize=(6.8, 1.55 * len(APPENDIX)), sharex=True, sharey=True,
                            squeeze=False, gridspec_kw={'hspace': 0.35, 'wspace': 0.05})
    for i, s in enumerate(APPENDIX):
        mean = np.nanmean(sections[s], axis=0)
        profile = channel_profile(mean, lat)
        for j, (section, title) in enumerate([(sections[s][medians[s]], 'median'), (mean, 'batch mean')]):
            cf = plot_section(axs[i, j], section, lat, depth, f'{panel_letter(2 * i + j)}) {NAMES[s]}, {title}',
                              profile=profile)
        axs[i, 0].set_ylabel('depth [m]')
    for ax in axs[-1]:
        ax.set_xlabel('latitude [°N]')
    fig.colorbar(cf, ax=axs, label=SIGMA_LABEL, fraction=0.015, pad=0.01).ax.invert_yaxis()
    figs['fig1_appendix'] = fig
    return figs


def main():
    """Compute, plot and save the figures in OUT."""
    for name, fig in plot(compute()).items():
        save(fig, f'{OUT}/{name}.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
