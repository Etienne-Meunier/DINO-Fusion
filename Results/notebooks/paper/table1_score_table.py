"""Table 1: mean ± std over the batch of the box metrics and of the density error, one row per source, ordered and named as in NAMES."""
from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric

file_mask_LR = load_mask()

box_metrics = {'BWbox': partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox': partial(temperature_DWbox_metric, depth_box=1500)}
columns = [('BWbox', 'toce'), ('BWbox', 'soce'),
           ('DWbox', 'toce'), ('DWbox', 'soce'),
           ('density_errors', 'density')]
LABELS = {('BWbox', 'toce'): r'BW T [$^\circ$C]', ('BWbox', 'soce'): r'BW S [g kg$^{-1}$]',
          ('DWbox', 'toce'): r'DW T [$^\circ$C]', ('DWbox', 'soce'): r'DW S [g kg$^{-1}$]',
          ('density_errors', 'density'): r'Density error [\%]'}


stats = []
for source, x in sources.items():
    fields = {k: v.detach().cpu() for k, v in observer.unnormalized(x).items()}
    for field in ['toce', 'soce']:
        for bi, b in enumerate(fields[field]):
            sample = file_mask_LR.e3t_0.copy()
            sample[:] = np.asarray(b)
            base = {'source': source, 'index': bi, 'field': field}
            for key, f in box_metrics.items():
                stats.append(base | {'metric': key, 'value': f(sample, file_mask_LR).item()})
    errors = compute_density_error(fields, file_mask_LR)
    for bi, v in enumerate(errors):
        stats.append({'source': source, 'index': bi, 'field': 'density',
                      'metric': 'density_errors', 'value': float(v)})
stats = pd.DataFrame(stats)


def format_mean_std(mean, std, precision=2):
    """Format `mean ± std` with `precision` decimals, a std that rounds to zero written as `< 10^-precision`."""
    floor = 10 ** -precision
    std_str = f'< {floor:.{precision}f}' if std < floor / 2 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'


grouped = stats.groupby(['source', 'metric', 'field'])['value']
means, stds = grouped.mean(), grouped.std()
table = pd.DataFrame({
    f'{metric} {field}': [format_mean_std(means[(s, metric, field)], stds[(s, metric, field)], 1 if metric == 'density_errors' else 2) for s in sources]
    for metric, field in columns}, index=list(sources))
ORDER = ['data', 'none', 'none_border0', 'negative_grad_density_huber_border0', 'mean_density_border0', 'zero_mean_border0']
table = table.loc[[s for s in ORDER if s in table.index]]
print(table.to_string())

latex = table.rename(index=NAMES, columns={f'{m} {f}': LABELS[(m, f)] for m, f in columns})
latex = latex.map(lambda s: s.replace('±', r'$\pm$'))
latex = latex.to_latex(column_format='l' + 'c' * len(columns))
print(latex)
os.makedirs(FIGURES, exist_ok=True)
with open(f'{FIGURES}/table1_score_table.tex', 'w') as f:
    f.write(latex)
