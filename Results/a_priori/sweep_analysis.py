#%%
# Constraint enforcement vs variance of the hyper-parameter sweep (`sweep_run.py`).
#   enforcement : density errors (% of volume with a density inversion, Table 1),
#                 RMSE of the mean density profile and of its vertical gradient against the training mean
#   variance    : mean point by point variance of T and S over the batch (Table 2)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from load_data import paper_path, LABELS
from sweep_run import sweep_path, runs
from campaign_database import describe, run_metrics, references, figures_path

#%%
# Metrics of the sweep runs and of the references

rows = []
for name in runs().keys() :
    if not Path(f'{sweep_path}/{name}/clean/toce.npy').exists() :
        continue
    rows.append({'run' : name} | describe(name) | run_metrics(name))

df = pd.DataFrame(references() + rows)
Path(figures_path).mkdir(exist_ok=True, parents=True)
df.to_csv(f'{figures_path}/sweep_metrics.csv', index=False)
pd.set_option('display.width', 250)
print(df.reindex(columns=['run', 'diverged', 'density_errors', 'profile_rmse', 'gradient_rmse', 'var_T', 'var_S', 'final_beta', 'final_rho', 'final_norm_lambda']).round(4).to_string(index=False))

#%%
# Trade-off : enforcement vs variance, one column per constraint family

families = [f for f in df['family'].unique() if f.startswith(('gradient', 'projection'))]
if not families :
    sys.exit('No constrained run finished yet')
y_metrics = {'density_errors' : 'density errors [% volume]', 'profile_rmse' : 'mean density profile RMSE [kg m$^{-3}$]'}
ref_style = {'data' : ('*', 'k'), 'paper No constraint' : ('X', 'grey'), 'paper Constraint TS' : ('s', 'tab:blue'),
             'paper Constraint C1' : ('^', 'tab:orange'), 'paper Constraint C2' : ('v', 'tab:green')}
controls = df[df['run'].isin(['none', 'border_zero_only'])]

fig, axs = plt.subplots(len(y_metrics), len(families), figsize=(4.5 * len(families), 4 * len(y_metrics)), squeeze=False, sharex=True)
for j, family in enumerate(families) :
    sub = df[df['family'] == family]
    gradient = family.startswith('gradient')
    marker_key = 'beta' if gradient else 'eta_rho'
    markers = dict(zip(sorted(sub[marker_key].unique()), ['o', 'D', 'P', 'h']))
    for i, (metric, label) in enumerate(y_metrics.items()) :
        ax = axs[i, j]
        for value, m in markers.items() :
            s = sub[sub[marker_key] == value]
            sc = ax.scatter(s['var_T'], s[metric], c=s['eta'].clip(lower=1e-3) if not gradient else s['eta'], cmap='viridis', norm=LogNorm(sub['eta'][sub['eta'] > 0].min() if gradient else 1e-3, sub['eta'].max()),
                            marker=m, s=60, edgecolor='k', label=f'{marker_key}={value:g}')
            if not gradient :
                zero = s[s['eta'] == 0]
                ax.scatter(zero['var_T'], zero[metric], facecolor='white', edgecolor='k', marker=m, s=60)
        for _, r in df[df['family'] == 'reference'].iterrows() :
            mk, col = ref_style.get(r['run'], ('o', 'k'))
            ax.scatter(r['var_T'], r[metric], marker=mk, color=col, s=90, label=r['run'])
        for _, r in controls.iterrows() :
            ax.scatter(r['var_T'], r[metric], marker='x', color='red' if r['run'] == 'none' else 'purple', s=70, label=r['run'])
        ax.set_xscale('log')
        ax.set_title(LABELS.get(family, family) if i == 0 else '')
        if j == 0 : ax.set_ylabel(label)
        if i == len(y_metrics) - 1 : ax.set_xlabel('mean point by point variance of T [°C$^2$]')
        ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axs[:, j], orientation='horizontal', fraction=0.04, pad=0.12)
    cb.set_label('eta (white : eta = 0)' if not gradient else 'eta')
    axs[0, j].legend(fontsize=7, loc='upper left')
fig.savefig(f'{figures_path}/sweep_tradeoff.png', dpi=200, bbox_inches='tight')
print(f'Saved in {figures_path}')
