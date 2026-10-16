from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric

file_mask_LR = load_mask()

box_metrics = {'BWbox': partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox': partial(temperature_DWbox_metric, depth_box=1500)}


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

panels = [('BWbox', 'toce'), ('BWbox', 'soce'),
          ('DWbox', 'toce'), ('DWbox', 'soce'),
          ('density_errors', 'density')]

fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), squeeze=False)
for ax, (metric, field) in zip(axs[0], panels):
    sub = stats.query('metric == @metric & field == @field')
    sns.histplot(data=sub, x='value', hue='source', ax=ax, element='step')
    ax.set_title(f'metric : {metric} - field : {field}', fontsize=9)
fig.tight_layout()



def format_mean_std(mean, std, precision=1):
    """Format `mean ± std`, with the std in scientific notation when below 0.1."""
    std_str = f'{std:.1e}'.replace('e-0', 'e-') if std < 0.1 else f'{std:.{precision}f}'
    return f'{mean:.{precision}f} ± {std_str}'

grouped = stats.groupby(['source', 'metric', 'field'])['value']
table = pd.DataFrame({
    f'{metric} {field}': [format_mean_std(grouped.mean()[(s, metric, field)], grouped.std()[(s, metric, field)])
                          for s in sources]
    for metric, field in panels}, index=list(sources))
print(table.to_string())


table['border0'] = ['border0' in n for n in table.index]
table['constraint'] = [n.replace('_border0', '') for n in table.index]
ti = table.set_index(['constraint', 'border0'])

cols = [f'{metric} {field}' for metric, field in panels]
wide = ti[cols].unstack('border0').reindex(columns=pd.MultiIndex.from_product([cols, [False, True]]))
wide.columns = pd.MultiIndex.from_tuples([(c.replace('_', r'\_'), 'border0' if b else 'free')
                                          for c, b in wide.columns])
wide.index = wide.index.str.replace('_', r'\_').rename(None)
wide = wide.map(lambda s: s.replace('±', r'$\pm$') if isinstance(s, str) else s)
print(wide.to_latex(multicolumn=True, multicolumn_format='c', na_rep='--', column_format='l' + 'cc' * len(cols)))