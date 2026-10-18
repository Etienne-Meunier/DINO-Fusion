"""Table 1: mean ± std over the batch of the box metrics and of the density error, one row per source (full run name)."""
from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric

file_mask_LR = load_mask()

box_metrics = {'BWbox': partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox': partial(temperature_DWbox_metric, depth_box=1500)}
columns = [('BWbox', 'toce'), ('BWbox', 'soce'),
           ('DWbox', 'toce'), ('DWbox', 'soce'),
           ('density_errors', 'density')]


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


def format_mean_std(mean, std, precision=1):
    """Format `mean ± std`, with the std in scientific notation when below 0.1."""
    std_str = f'{std:.1e}'.replace('e-0', 'e-') if std < 0.1 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'


grouped = stats.groupby(['source', 'metric', 'field'])['value']
means, stds = grouped.mean(), grouped.std()
table = pd.DataFrame({
    f'{metric} {field}': [format_mean_std(means[(s, metric, field)], stds[(s, metric, field)]) for s in sources]
    for metric, field in columns}, index=list(sources))
print(table.to_string())

latex = table.rename(index=lambda s: s.replace('_', r'\_'), columns=lambda c: c.replace('_', r'\_'))
latex = latex.map(lambda s: s.replace('±', r'$\pm$'))
latex = latex.to_latex(column_format='l' + 'c' * len(columns))
print(latex)
os.makedirs(FIGURES, exist_ok=True)
with open(f'{FIGURES}/table1_score_table.tex', 'w') as f:
    f.write(latex)
