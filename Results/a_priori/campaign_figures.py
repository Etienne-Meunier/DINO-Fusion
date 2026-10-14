#%%
# Figures of the beta campaign (`campaign_run.py`) from the database (`campaign_database.py`).
#   1. Tweedie vs x_t : paired runs (same loss, schedule and hyper-parameters) with the loss evaluated on x0_hat or on x_t
#   2. Variability / constraint trade-off along the constant beta scan
#   3. Adaptive (eq, ineq) and scheduled (linear_up, linear_down) beta against the Pareto front of the constant beta
# A run is better when it has fewer density errors and keeps more variance (point by point variance of T).
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from campaign_database import build, figures_path
from campaign_run import rms_grad

LOSSES = {'zero_mean' : 'zero mean T/S', 'mean_grad_density' : 'mean density gradient', 'negative_grad_density' : 'negative density gradient',
          'dist_isoton' : 'distance to isotonic'}
SCHEDULES = {'constant' : ('o', 'k'), 'linear_up' : ('^', 'tab:red'), 'linear_down' : ('v', 'tab:blue'), 'eq' : ('D', 'tab:green'), 'ineq' : ('s', 'tab:purple')}
REFS = {'data' : ('*', 'k'), 'paper No constraint' : ('X', 'grey'), 'paper Constraint TS' : ('P', 'tab:blue'),
        'paper Constraint C1' : ('<', 'tab:orange'), 'paper Constraint C2' : ('>', 'tab:green')}

df = build() if '--rebuild' in sys.argv else pd.read_csv(f'{figures_path}/database.csv')
df['diverged'] = df['diverged'].fillna(False).astype(bool)
df['tweedie'] = df['tweedie'].fillna(False).astype(bool)
data = df[df['run'] == 'data'].iloc[0]
df['var_T_ratio'] = df['var_T'] / data['var_T']
df['var_S_ratio'] = df['var_S'] / data['var_S']
gen = df[(df['method'] == 'gradient') & df['loss'].isin(LOSSES.keys()) & ~df['diverged']].copy()
# Hyper-parameter key shared by the x_t and Tweedie versions of a run
gen['key'] = gen['run'].str.replace('gradient_tweedie_', 'gradient_').str.replace(r'(_eq|_beta=0)?_eta=', '_eq_eta=', regex=True)
gen.loc[gen['beta_type'] != 'eq', 'key'] = gen['run'].str.replace('gradient_tweedie_', 'gradient_')
gen = gen.drop_duplicates(['key', 'tweedie'])
gen['step'] = gen['mean_beta'] * gen['loss'].map(rms_grad)
none = df[df['run'] == 'none'].iloc[0]
bzo = df[df['run'] == 'border_zero_only'].iloc[0]


def pareto(d, x='density_errors', y='var_T') :
    """
        Rows of d not dominated by another row (lower x and higher y).
    """
    d = d.sort_values([x, y], ascending=[True, False])
    front, best = [], -np.inf
    for i, r in d.iterrows() :
        if r[y] > best :
            front.append(i)
            best = r[y]
    return d.loc[front]


def references(ax, x='density_errors', y='var_T_ratio') :
    """
        Data, paper generations and controls on a trade-off plane.
    """
    for _, r in df[df['family'] == 'reference'].iterrows() :
        mk, col = REFS.get(r['run'], ('o', 'k'))
        ax.scatter(r[x], r[y], marker=mk, color=col, s=110, zorder=5, edgecolor='w', label=r['run'])
    ax.scatter(none[x], none[y], marker='x', color='red', s=80, zorder=5, label='none')
    ax.scatter(bzo[x], bzo[y], marker='x', color='purple', s=80, zorder=5, label='border_zero_only')

Path(figures_path).mkdir(exist_ok=True, parents=True)

#%%
# 1. Tweedie vs x_t

paired = gen[~gen['tweedie']].merge(gen[gen['tweedie']], on='key', suffixes=('_xt', '_tw'))
if len(paired) == 0 :
    sys.exit('No paired x_t / Tweedie runs yet')
n_div = df[(df['method'] == 'gradient') & df['loss'].isin(LOSSES.keys())].groupby('tweedie')['diverged'].agg(['sum', 'count'])
print('Diverged runs (tweedie : diverged / total)\n', n_div)
metrics = {'density_errors' : 'density errors [% volume]', 'profile_rmse' : 'profile RMSE [kg m$^{-3}$]', 'var_T_ratio' : 'var T / var T data'}
fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.6))
for ax, (m, label) in zip(axs, metrics.items()) :
    for loss, (mk, col) in zip(LOSSES, [('o', 'tab:blue'), ('s', 'tab:orange'), ('^', 'tab:green'), ('D', 'tab:red')]) :
        p = paired[paired['loss_xt'] == loss]
        ax.scatter(p[f'{m}_xt'], p[f'{m}_tw'], marker=mk, color=col, edgecolor='k', s=40, label=LOSSES[loss])
    lim = [min(paired[f'{m}_xt'].min(), paired[f'{m}_tw'].min()) * 0.8, max(paired[f'{m}_xt'].max(), paired[f'{m}_tw'].max()) * 1.2]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'{label}, loss on x_t'); ax.set_ylabel(f'{label}, loss on Tweedie x0')
    ax.grid(alpha=0.3)
    better = (paired[f'{m}_tw'] < paired[f'{m}_xt']).mean()
    ax.set_title(f'Tweedie lower in {100 * better:.0f}% of {len(paired)} pairs')
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{figures_path}/campaign_1_tweedie_vs_xt.png', dpi=200)
summary = paired.groupby('loss_xt').apply(lambda p : pd.Series({
    'pairs' : len(p),
    'tweedie fewer errors [%]' : 100 * (p['density_errors_tw'] < p['density_errors_xt']).mean(),
    'median errors ratio tw/xt' : (p['density_errors_tw'] / p['density_errors_xt']).median(),
    'median var_T ratio tw/xt' : (p['var_T_tw'] / p['var_T_xt']).median()}))
print(summary.round(2))
summary.to_csv(f'{figures_path}/campaign_1_tweedie_vs_xt.csv')

#%%
# 2. Variability / constraint trade-off along the constant beta scan (Tweedie)

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, var in zip(axs, ['var_T_ratio', 'var_S_ratio']) :
    for loss, col in zip(LOSSES, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']) :
        c = gen[gen['tweedie'] & (gen['loss'] == loss) & (gen['beta_type'] == 'constant')].sort_values('beta')
        if len(c) == 0 :
            continue
        ax.plot(c['density_errors'], c[var], '-', color=col, lw=1.5, alpha=0.7, label=LOSSES[loss])
        sc = ax.scatter(c['density_errors'], c[var], c=c['step'], cmap='viridis', norm=LogNorm(1e-4, 1e-1), s=60, edgecolor=col, linewidth=1.5, zorder=4)
    references(ax, y=var)
    ax.axhline(1, color='k', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.04, 40)
    ax.set_xlabel('density errors [% volume]')
    ax.set_ylabel(f'{var.split("_")[1]} variance / data variance')
    ax.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=axs, fraction=0.03)
cb.set_label('beta * rms|grad L| (step per sampling step)')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=7, fontsize=8, bbox_to_anchor=(0.45, -0.08))
axs[0].set_title('Constant beta scan, loss on Tweedie x0 : T')
axs[1].set_title('S')
fig.savefig(f'{figures_path}/campaign_2_tradeoff_constant_beta.png', dpi=200, bbox_inches='tight')

#%%
# 2b. Constraint and variance as a function of the constant beta (x_t dashed, Tweedie solid)

fig, axs = plt.subplots(2, len(LOSSES), figsize=(4.5 * len(LOSSES), 7), sharex='col', squeeze=False)
for j, loss in enumerate(LOSSES) :
    for tweedie, ls in [(True, '-'), (False, '--')] :
        c = gen[(gen['tweedie'] == tweedie) & (gen['loss'] == loss) & (gen['beta_type'] == 'constant')].sort_values('beta')
        x = c['beta'] * rms_grad[loss]
        axs[0, j].plot(x, c['density_errors'], ls, marker='o', color='tab:red', label=f'loss on {"Tweedie x0" if tweedie else "x_t"}')
        axs[1, j].plot(x, c['var_T_ratio'], ls, marker='o', color='tab:blue', label=f'T, loss on {"Tweedie x0" if tweedie else "x_t"}')
        axs[1, j].plot(x, c['var_S_ratio'], ls, marker='s', color='tab:cyan', label=f'S, loss on {"Tweedie x0" if tweedie else "x_t"}')
    for i, (m, ref) in enumerate([('density_errors', None), ('var_T_ratio', None)]) :
        ax = axs[i, j]
        ax.axhline(none[m], color='red', lw=0.8, ls=':', label='none')
        ax.axhline(bzo[m], color='purple', lw=0.8, ls=':', label='border_zero_only')
        ax.axhline(data[m] if m != 'var_T_ratio' else 1, color='k', lw=0.8, ls=':', label='data')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.3)
    axs[0, j].set_title(LOSSES[loss])
    axs[1, j].set_xlabel('beta * rms|grad L|')
axs[0, 0].set_ylabel('density errors [% volume]')
axs[1, 0].set_ylabel('variance / data variance')
axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f'{figures_path}/campaign_2b_metrics_vs_beta.png', dpi=200)

#%%
# 3. Adaptive / scheduled beta against the Pareto front of the constant beta

fig, axs = plt.subplots(2, len(LOSSES), figsize=(4.8 * len(LOSSES), 9), squeeze=False)
rows = []
for i, tweedie in enumerate([True, False]) :
    for j, loss in enumerate(LOSSES) :
        ax = axs[i, j]
        g = gen[(gen['tweedie'] == tweedie) & (gen['loss'] == loss)]
        const = g[g['beta_type'] == 'constant']
        if len(const) :
            front = pareto(const)
            ax.plot(front['density_errors'], front['var_T_ratio'], 'k-', lw=1, alpha=0.6)
        for schedule, (mk, col) in SCHEDULES.items() :
            s = g[g['beta_type'] == schedule]
            ax.scatter(s['density_errors'], s['var_T_ratio'], marker=mk, color=col, edgecolor='k', s=50, label=schedule, zorder=4)
            for _, r in s.iterrows() :
                dominated = ((const['density_errors'] <= r['density_errors']) & (const['var_T'] >= r['var_T'])).any() if len(const) else np.nan
                dominates = ((const['density_errors'] >= r['density_errors']) & (const['var_T'] <= r['var_T'])).any() if len(const) else np.nan
                rows.append({'loss' : loss, 'tweedie' : tweedie, 'beta_type' : schedule, 'run' : r['run'], 'density_errors' : r['density_errors'],
                             'var_T_ratio' : r['var_T_ratio'], 'dominated_by_constant' : dominated, 'dominates_a_constant' : dominates})
        references(ax)
        ax.axhline(1, color='k', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.04, 40)
        ax.set_title(f'{LOSSES[loss]}, loss on {"Tweedie x0" if tweedie else "x_t"}', fontsize=10)
        ax.set_xlabel('density errors [% volume]')
        if j == 0 : ax.set_ylabel('T variance / data variance')
        ax.grid(alpha=0.3)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f'{figures_path}/campaign_3_adaptive_vs_constant.png', dpi=200)
dom = pd.DataFrame(rows)
dom.to_csv(f'{figures_path}/campaign_3_adaptive_vs_constant.csv', index=False)
if len(dom) :
    print(dom[dom['beta_type'] != 'constant'].groupby(['beta_type', 'tweedie'])[['dominated_by_constant', 'dominates_a_constant']].mean().round(2))
print(f'Saved in {figures_path}')
