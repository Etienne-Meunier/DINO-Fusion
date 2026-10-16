# In Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/sweep_strength
# Load all the argjson and put the configuration in a pandas table with the parameters + path to read the simu
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import json
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr

from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pipelines.losses import get_loss
from metrics import compute_density_error

SWEEP = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference/sweep_strength'


def load_runs(sweep=SWEEP):
    """Return a DataFrame with one row per run of `sweep`: the args.json entries plus `path`, the run folder
    (holding generated.npy, lbd.npy and clean/)."""
    rows = []
    for args_path in sorted(glob(f'{sweep}/*/*/args.json')):
        with open(args_path) as f:
            rows.append(json.load(f) | {'path': os.path.dirname(args_path)})
    return pd.DataFrame(rows)


runs = load_runs()
print(runs[['loss', 'eps', 'eta', 'lbd0', 'path']].to_string())

# Iterate through all runs, load the clean data and compute the density error and the constraint applied on the generated state 
# (So if it's zero mean this one else mean_density) add that to the table 

file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
file_mask_LR = file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
observer = Observer('data', TrainingConfig(), device='cpu')
losses = {name: get_loss(name, observer) for name in runs['loss'].unique()}

density_error, constraint = [], []
for _, run in runs.iterrows():
    clean = {f: np.load(f'{run.path}/clean/{f}.npy') for f in ['toce', 'soce']}
    density_error.append(compute_density_error(clean, file_mask_LR).mean())
    with torch.no_grad():
        generated = torch.from_numpy(np.load(f'{run.path}/generated.npy'))
        constraint.append(losses[run.loss](generated).median().item())
runs['density_error'] = density_error
runs['constraint'] = constraint
print(runs[['loss', 'eps', 'eta', 'lbd0', 'density_error', 'constraint']].to_string())

dr = runs[['loss', 'eps', 'eta', 'lbd0', 'density_error', 'constraint']]

dr
dr.query('loss == "mean_density"')

import seaborn as sns

drm = dr.melt(id_vars=['loss', 'eps', 'eta', 'lbd0'], value_vars=['density_error', 'constraint'])

pr = lambda l : sns.relplot(data=drm.query(f'loss == "{l}"'),
                            kind='line', x='eta', y='value', hue='eps',
                            style='lbd0', col='variable', facet_kws={"sharey": False})

pr('mean_density')

pr('zero_mean')

sns.scatterplot(data=dr.query('loss == "mean_density"'),
                x='density_error', y='constraint', hue='eta', style='lbd0',palette='viridis')


sns.scatterplot(data=dr.query('loss == "zero_mean"'),
                x='density_error', y='constraint', hue='eta', style='lbd0',palette='viridis')

# Starting from the dataframe above we want to make 2 lines (one for each loss) of 3 
# plots : eta vs c(x) eta vs dE and dE vs c(x)
# Althouhg we would like to show the 2 varying config for an eta (lbd, eps) on the left plots
# And the 3 varying config on the right plot
# In a beautiful way

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ETA, C, DE = r'$\eta$', 'c(x)', 'Density error'
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for row, loss in zip(axes, ['zero_mean', 'mean_density']):
    sub = (dr.query('loss == @loss and not (loss == "mean_density" and eta == 1)').sort_values('eta')
           .rename(columns={'eta': ETA, 'constraint': C, 'density_error': DE}))
    line_kw = {'hue': 'eps', 'style': 'lbd0', 'markers': True, 'palette': 'Set1', 'markersize': 8}
    sns.lineplot(data=sub, x=ETA, y=C, ax=row[0], **line_kw)
    sns.lineplot(data=sub, x=ETA, y=DE, ax=row[1], legend=False, **line_kw)
    sns.scatterplot(data=sub, x=DE, y=C, ax=row[2],
                    hue=ETA, hue_norm=LogNorm(dr.eta.min(), dr.eta.max()), palette='viridis',
                    style='lbd0', size='eps', sizes=(50, 140), legend='full')
    sns.move_legend(row[2], 'upper left', bbox_to_anchor=(1, 1), frameon=False)
    row[0].set_ylabel(f'{loss.replace("_", " ").capitalize()}\n\n{C}')

for ax in axes.flat:
    ax.set(xscale='log', yscale='log')
    ax.grid(alpha=.3)
for ax, title in zip(axes[0], [f'{ETA} vs {C}', f'{ETA} vs {DE}', f'{DE} vs {C}']):
    ax.set_title(title)
sns.despine(fig)

FIG = f'{PRP}/Results/notebooks/figures/understand/1_constraint_strenght'
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(f'{FIG}.pdf', bbox_inches='tight')
