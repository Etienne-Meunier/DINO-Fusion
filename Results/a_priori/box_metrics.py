#%%
# Paper Table 1 : Bottom-Water / Deep-Water box averages of T and S and density errors (mean ± std).
import numpy as np
import seaborn as sns
import pandas as pd
from functools import partial
from load_data import load_generations, load_data_batch, load_mask, get_args, LABELS
from metrics import *

s = get_args('Bottom-Water / Deep-Water boxes and density errors')
bg = load_generations(s.get('path_box_metrics', s['path']))
batch = load_data_batch(batch_size=8)
file_mask_LR = load_mask()

#%%
# Bottom-Water and Deep-Water boxes

stats = []

metrics = {'BWbox' : partial(temperature_BWbox_metric, depth_box=1500),
           'DWbox' : partial(temperature_DWbox_metric, depth_box=1500)}

for gen in list(bg.keys()) + ['data'] :
    batch_raw =  batch if gen == 'data' else bg[gen]
    for field in ['toce', 'soce'] :
        for bi, b in  enumerate(batch_raw[field]) :
            data = file_mask_LR.e3t_0.copy()
            data[:] = b
            base = {'source' : gen, 'index' : bi, 'field' : field}
            for key, f in metrics.items() :
                stats.append(base | {'metric' : key , 'value' : f(data, file_mask_LR).item()})
stats = pd.DataFrame(stats)

#%%
# Density errors

densities_errors_stats = {'data' : compute_density_error(batch, file_mask_LR)}
for key in bg.keys() :
    densities_errors_stats[key] = compute_density_error(bg[key], file_mask_LR)

for key, v in densities_errors_stats.items() :
    df = pd.DataFrame({'index' : np.arange(v.shape[0]), 'value' : v})
    df['source'] = key
    df['metric'] ='density_errors'
    df['field'] = 'density'
    stats = pd.concat([stats, df])

sns.histplot(data=stats.query('(metric == "density_errors") & (field=="density") & (source!="no_constraint")'), x='value', hue='source')

#%%
# Latex table

def create_pivot_with_stats(stats_df):
    mean_pivot = pd.pivot_table(stats_df,
                              index='source',
                              columns=['metric', 'field'],
                              values='value',
                              aggfunc='mean')

    std_pivot = pd.pivot_table(stats_df,
                             index='source',
                             columns=['metric', 'field'],
                             values='value',
                             aggfunc='std')

    return mean_pivot, std_pivot

def format_value_with_std(mean, std, precision=1):
    """Format value with compact scientific notation for small standard deviations"""
    mean_str = f"{mean:.{precision}f}"

    if std < 0.1:
        std_str = f"{std:.1e}"
        std_str = std_str.replace('e-0', 'e-')
    else:
        std_str = f"{std:.{precision}f}"

    return f"{mean_str} $\\pm$ {std_str}"

def create_latex_table(mean_df, std_df):
    """Latex table with one row per source : mean ± std of the box averages and density errors."""
    row_order = ['data'] + list(bg.keys())
    row_names = LABELS | {'data' : 'Data'}

    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l|cc|cc|c}",
        "\\hline",
        "& \\multicolumn{2}{c|}{Bottom-Water} & \\multicolumn{2}{c|}{Deep-Water} & Density \\\\",
        "Source & $\\mathcal{S}$ & $\\mathcal{T}$ & $\\mathcal{S}$ & $\\mathcal{T}$ & Errors \\\\",
        "\\hline"
    ]

    # Add data rows in specified order
    for idx in row_order:
        row_values = []
        for col in [('BWbox', 'soce'), ('BWbox', 'toce'),
                    ('DWbox', 'soce'), ('DWbox', 'toce'),
                    ('density_errors', 'density')]:
            mean_val = mean_df.loc[idx, col]
            std_val = std_df.loc[idx, col]
            row_values.append(format_value_with_std(mean_val, std_val))

        row = f"{row_names[idx]} & " + " & ".join(row_values) + " \\\\"
        latex.append(row)

    latex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Statistical analysis of water masses and density errors. Values are presented as mean $\\pm$ standard deviation.}",
        "\\label{tab:water_masses}",
        "\\end{table}"
    ])

    return "\n".join(latex)

mean_pivot, std_pivot = create_pivot_with_stats(stats)
print(mean_pivot)
latex_table = create_latex_table(mean_pivot, std_pivot)
print(latex_table)
