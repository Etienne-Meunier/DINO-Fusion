import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import pandas as pd
sys.path.append('../../../Diffusion_Model/')
sys.path.append('..')
from metrics import *
from functools import partial
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'



from utils import get_dataloader

# 1. Data load
home = '/Users/emeunier/Documents/'#/Volumes/LoCe/oceandata/models/dino-fusion/'
model_path = f'{home}/tav0h83b/'

#Small generations
# path = {
#     'constraint' : f'{model_path}/inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/20250130-165431_clean',
#     'no_constraint' : f'{model_path}/inference/infesteps_1000/constraints_no_constraints/20250131-110120_clean'
#     }

# # Large generations
path = {
    'constraint' : f'{model_path}/inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/20250203-175158_clean',
    'no_constraint' : f'{model_path}/inference/infesteps_1000/constraints_no_constraints/20250203-141645_clean'
    }

bg = {'constraint' : [], 'no_constraint' : []}

for key, p in path.items() :
      bg[key] = {'ssh' : np.load(p + '/ssh.npy'),
                 'soce' : np.load(p + '/soce.npy'),
                 'toce' : np.load(p + '/toce.npy')}

data_file = '/Users/emeunier/Documents/dino_1_4_degree_coarse_240125.tar'
train_dataloader = get_dataloader(data_file, batch_size=200, transform=False, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)

for k in ['toce', 'soce', 'ssh'] :
    batch[k] = batch.pop(f'{k}.npy')

file_mask_LR=xr.open_dataset("../data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
file_mask_LR=file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})


# fig, axs = plt.subplots(1,3, figsize=(17,5))
# im = axs[0].imshow(file_mask_LR.e1t.values)
# axs[0].set_title('e1t')
# fig.colorbar(im, ax=axs[0])

# im2 = axs[1].imshow(file_mask_LR.e2t.values)
# axs[1].set_title('e2t')
# fig.colorbar(im2, ax=axs[1])

# file_mask_LR.e3t_0.mean(dim=['nav_lat', 'nav_lon']).plot(ax=axs[2])
# axs[2].set_title('e3t_0')


# 2. Get metrics
# T500, TBW and TDW during ML Training of the 8 lineages

stats = []


metrics = {'BWbox' : partial(temperature_BWbox_metric, depth_box=1500),
           'DWbox' : partial(temperature_DWbox_metric, depth_box=1500)}

for gen in ['no_constraint', 'constraint', 'data'] :
    batch_raw =  batch if gen == 'data' else bg[gen]
    for field in ['toce', 'soce'] :
        for bi, b in  enumerate(batch_raw[field]) :
            data = file_mask_LR.e3t_0.copy()
            data[:] = b
            base = {'source' : gen, 'index' : bi, 'field' : field}
            for key, f in metrics.items() :
                stats.append(base | {'metric' : key , 'value' : f(data, file_mask_LR).item()})
stats = pd.DataFrame(stats)
stats

# # Data distribution
# sns.relplot(data=stats.query('source == "data"'), x='BWbox', y='DWbox',
#             hue='index', col='field',facet_kws={'sharey': False, 'sharex': False})
# sns.relplot(data=stats, x='BWbox', y='DWbox',
#             hue='source', col='field',facet_kws={'sharey': False, 'sharex': False})
# stats.groupby('source').count()

# sns.relplot(data=stats.query('source != "no_constraint"'), x='BWbox', y='DWbox',
#             hue='source', col='field',facet_kws={'sharey': False, 'sharex': False})



def compute_density_error(batch, file_mask_LR) :
    """
        Take a batch with toce (b, z, h, w) and soce along with the maks
        and return the pourcent of volume unstable per element in the batch
    """
    bsize = batch['soce'].shape[0]

    data = file_mask_LR.e3t_0.copy().expand_dims({'batch' : bsize})
    soce = data.copy()
    soce[:] = batch['soce']
    toce = data.copy()
    toce[:] = batch['toce']

    tmask = data.copy()
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)

    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0).expand_dims({'batch' : bsize})

    density = get_density_at_surface(toce, soce, tmask)
    errors_density = (((density.diff('depth') < 0.0) * volume * tmask).sum(['nav_lat', 'nav_lon', 'depth']) / (volume * tmask).sum(['nav_lat', 'nav_lon', 'depth']))*100
    return errors_density.values




densities_errors = compute_density_error(batch, file_mask_LR)


densities_errors_constraint = compute_density_error(bg['constraint'], file_mask_LR)
densities_errors_no_constraint = compute_density_error(bg['no_constraint'], file_mask_LR)
densities_errors_stats = {'data' : densities_errors, 'constraint' : densities_errors_constraint, 'no_constraint' : densities_errors_no_constraint}
for key, v in densities_errors_stats.items() :
    df = pd.DataFrame({'index' : np.arange(densities_errors.shape[0]), 'value' : v})
    df['source'] = key
    df['metric'] ='density_errors'
    df['field'] = 'density'
    stats = pd.concat([stats, df])
stats.pivot_table(index='source', columns=['metric', 'field'], values='value', aggfunc='std')
stats.pivot_table(index='source', columns=['metric', 'field'], values='value')
sns.histplot(data=stats.query('(metric == "density_errors") & (field=="density") & (source!="no_constraint")'), x='value', hue='source')#, col='field')

stats.query('(metric == "density_errors") & (field=="density")')

stats.query('(metric == "density_errors") & (field=="density")').groupby('source')['value'].mean()

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
    # Define row order
    row_order = ['data', 'no_constraint', 'constraint']

    # Define row names mapping
    row_names = {
        'data': 'Data',
        'no_constraint': 'No Constraint',
        'constraint': 'Constraint'
    }

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

# Usage:
mean_pivot, std_pivot = create_pivot_with_stats(stats)
latex_table = create_latex_table(mean_pivot, std_pivot)
print(latex_table)
