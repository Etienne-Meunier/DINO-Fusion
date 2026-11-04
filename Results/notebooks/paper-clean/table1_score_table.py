"""Table 1: mean over the batch of the box metrics and of the density error, one row per source, ordered and named as in
NAMES.

Box metrics: volume mean (e1t e2t e3t_0 of GRID) of T and S over the Bottom Water box (below DEPTH_BOX or poleward of
30°) and the Deep Water box (between 500 m and DEPTH_BOX, within 30° of the equator), the depth of a cell being its
gdept_0. Density error: `common.density_error`. The LaTeX table rounds mean and std to one decimal and gives the box
metrics of the generated sources as their bias to the data mean."""
from common import *

DEPTH_BOX = 1500
COLUMNS = [('BWbox', 'toce'), ('BWbox', 'soce'), ('DWbox', 'toce'), ('DWbox', 'soce'), ('density_errors', 'density')]
LABELS = {('BWbox', 'toce'): r'BW T [$^\circ$C]', ('BWbox', 'soce'): r'BW S [g kg$^{-1}$]',
          ('DWbox', 'toce'): r'DW T [$^\circ$C]', ('DWbox', 'soce'): r'DW S [g kg$^{-1}$]',
          ('density_errors', 'density'): r'Density error [\%]'}


def scores(x):
    """Per-member value (member,) of every column of COLUMNS for a normalized batch x."""
    ds = to_dataset(x)
    lat, depth = np.abs(GRID.mask.gphit), GRID.mask.gdept_0
    boxes = {'BWbox': ~((depth < DEPTH_BOX) & (lat < 30)), 'DWbox': (depth > 500) & (depth < DEPTH_BOX) & (lat < 30)}
    volume = cell_volume(GRID)
    out = {(b, f): ds[f].weighted(volume.where(box, 0.)).mean(('z_c', 'y_c', 'x_c')).values
           for b, box in boxes.items() for f in ('toce', 'soce')}
    return out | {('density_errors', 'density'): density_error(x)}


def format_mean_std(mean, std, precision=2):
    """Format `mean ± std` with `precision` decimals, a std that rounds to zero written as `< 10^-precision`."""
    floor = 10 ** -precision
    std_str = f'< {floor:.{precision}f}' if std < floor / 2 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'


def compute():
    """Mean and std over the batch of the scores of every source of CONFIGS['table']: {'order': sources, 'means',
    'stds': {source: {column: float}}}."""
    order = [s for s in CONFIGS['table'] if s in sources]
    values = {s: scores(sources[s]) for s in order}
    return {'order': order, 'means': {s: {c: v.mean() for c, v in values[s].items()} for s in order},
            'stds': {s: {c: v.std(ddof=1) for c, v in values[s].items()} for s in order}}


def table(computed):
    """Print the table of `computed` (`compute`) and its LaTeX version, return the LaTeX."""
    order, means, stds = computed['order'], computed['means'], computed['stds']
    table = pd.DataFrame({f'{m} {f}': [format_mean_std(means[s][(m, f)], stds[s][(m, f)],
                                                       1 if m == 'density_errors' else 2) for s in order]
                          for m, f in COLUMNS}, index=order)
    print(table.to_string())

    def latex_cell(s, c):
        """`mean ± std` of source s, the box metrics of a generated source as their bias to the data mean."""
        value = means[s][c] - (means['data'][c] if c[0] != 'density_errors' and s != 'data' else 0)
        return rf'{round(value, 1) + 0.0:.1f} $\pm$ {stds[s][c]:.1f}'  # + 0.0 turns -0.0 into 0.0

    latex = pd.DataFrame({LABELS[c]: [latex_cell(s, c) for s in order] for c in COLUMNS},
                         index=[NAMES[s] for s in order]).to_latex(column_format='l' + 'c' * len(COLUMNS))
    print(latex)
    return latex


def main():
    """Compute the table and save its LaTeX version as table1_score_table.tex in FIGURES."""
    save_tex(table(compute()), f'{FIGURES}/table1_score_table.tex')


if __name__ == '__main__':
    main()
