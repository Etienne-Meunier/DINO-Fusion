"""
Figures of `REPORT.md` from the tables already computed :
    outputs/sweep_table.csv            (`analyze_sweep.py`)
    outputs/local_metrics.csv          (`local_metrics.py`)
    outputs/compare_100steps_all.csv   (`compare_runs.py` on `prelim_sweep_clip/`)
    outputs/compare_1000steps_all.csv  (`compare_runs.py` on `test_new_sampler_clip/`)

Writes outputs/fig_calibration.png, fig_combo100.png, fig_relax1000.png, fig_eta1000.png, fig_posthoc.png.
"""
from __init__ import PRP; import sys

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = f'{PRP}/Results/paper-analysis/preliminaries/outputs'
LOSSES = ['mean_density', 'mean_grad_density', 'negative_grad_density', 'zero_mean', 'dist_isoton']
ZM_EPS = 2.62e-3


def references() :
    """Data and unconstrained (100 steps) medians of the local metrics, as two dicts."""
    local = pd.read_csv(f'{OUT}/local_metrics.csv')
    data = local[local['name'] == 'data'].median(numeric_only=True)
    unc = local[local['name'].str.startswith('none_s')].median(numeric_only=True)
    return data, unc


def fig_calibration() :
    """Final loss and distance vs eta, raw and sqrt modes, single loss, seed 0, no variant."""
    t = pd.read_csv(f'{OUT}/sweep_table.csv')
    t = t[(t['seed'] == 0) & t['loss'].isin(LOSSES) & t['eps_scale'].isna()]
    t = t[t[['t_start', 'lbd_max', 'final_opt', 'through_unet']].isna().all(axis=1)]
    eps = pd.read_csv(f'{OUT}/eps_data_quantiles.csv', index_col=0)['q0.5']
    unc = pd.read_csv(f'{OUT}/sweep_table.csv').query('name == "none_s0_n100_b8"').iloc[0]
    fig, axs = plt.subplots(3, len(LOSSES), figsize=(3.6*len(LOSSES), 9.0))
    for j, loss in enumerate(LOSSES) :
        for mode, c in [('raw', 'C0'), ('sqrt', 'C1')] :
            d = t[(t['loss'] == loss) & (t['mode'] == mode)].sort_values('eta')
            for ax, col in zip(axs[:, j], [f'loss_{loss}', 'satisfied', 'dist_unc']) :
                ax.plot(d['eta'], d[col].clip(upper=1e12), 'o-', c=c, ms=4, label=mode)
        axs[0, j].axhline(max(eps[loss], 1e-6), ls=':', c='g', label='eps (data q0.5)')
        axs[0, j].axhline(unc[f'loss_{loss}'], ls=':', c='k', label='unconstrained')
        axs[2, j].axhline(0.044, ls=':', c='g', label='two data states')
        axs[0, j].set_title(loss, fontsize=10)
        for ax in axs[:, j] :
            ax.set_xscale('log')
        axs[0, j].set_yscale('log')
        axs[2, j].set_yscale('log')
        axs[2, j].set_xlabel('eta')
    axs[0, 0].set_ylabel('final loss (median, inf shown at 1e12)')
    axs[1, 0].set_ylabel('satisfaction rate')
    axs[2, 0].set_ylabel('distance to unconstrained')
    axs[0, 0].legend(fontsize=7)
    axs[2, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_calibration.png', dpi=120)


def fig_combo100(data, unc) :
    """Inversions vs level-mean variance of the 100-step runs, one marker per seed."""
    c = pd.read_csv(f'{OUT}/compare_100steps_all.csv', index_col=0)
    groups = [('unconstrained', r'^none_s', 'k', 'o'),
              ('zero_mean raw 10', r'^zero_mean_raw_eta10.0_q0.5_s', 'C0', 'o'),
              ('mean_density sqrt 1', r'^mean_density_sqrt_eta1.0_q0.5_s', 'C7', 'o'),
              ('dist_isoton sqrt 1e-4', r'^dist_isoton_sqrt_eta0.0001_q0.5_s', 'C1', 'o'),
              ('iso+zm (1e-4, 1)', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-1.0_q0.5_s\d_n100_b8$', 'C2', 'o'),
              ('iso+zm (1e-4, 1) + joint', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-1.0_q0.5_s\d_n100_b8_joint', 'C2', '*'),
              ('iso+zm (1e-4, 1) + final_opt', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-1.0_q0.5_fo10', 'C2', 'P'),
              ('iso+zm (1e-4, 3)', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-3.0', 'C3', 'o'),
              ('iso+zm (1e-4, 10)', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-10.0', 'C3', 's'),
              ('iso+zm (3e-5, 1)', r'^dist_isoton\+zero_mean_sqrt_eta3e-05-1.0', 'C4', 'o'),
              ('iso+zm (1e-4, 1), zm eps x64/128/256', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-1.0_q0.5_x1.0-\d+.0_s\d_n100_b8$', 'C5', 'o'),
              ('  same + joint', r'^dist_isoton\+zero_mean_sqrt_eta0.0001-1.0_q0.5_x1.0-\d+.0_s\d_n100_b8_joint', 'C5', '*'),
              ('iso+zm+md (1e-4, 1, *)', r'^dist_isoton\+zero_mean\+mean_density_sqrt_eta0.0001-1.0', 'C6', 'o'),
              ('iso+md (1e-4, 1)', r'^dist_isoton\+mean_density_sqrt_eta0.0001-1.0', 'C8', 'o')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, col, label in zip(axs, ['inv', 'inv>0.01'], ['inversions', 'inversions > 0.01 kg/m3']) :
        for name, pattern, color, marker in groups :
            d = c[c.index.str.contains(pattern)]
            ax.scatter(d['lvl'], 100*d[col], c=color, marker=marker, s=60 if marker == '*' else 30, label=name)
        ax.scatter([data['level_mean']], [100*data['inversions' if col == 'inv' else 'inversions_0.01']], c='g', marker='X', s=80, label='data')
        ax.set_xscale('log')
        ax.set_xlabel('variance of level means')
        ax.set_ylabel(f'{label} (%)')
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.set_ylim(bottom=0)
    axs[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.suptitle('100 steps, batch 8 (one marker per seed)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_combo100.png', dpi=120)


def fig_relax1000(data) :
    """dist_isoton + zero_mean at 1000 steps, eta (5e-6, 0.1), against the zero_mean eps scale, before / after joint."""
    c = pd.read_csv(f'{OUT}/compare_1000steps_all.csv', index_col=0)
    runs = {1 : 'combo_iso5e-6_zm0.1', 16 : 'combo_iso5e-6_zm0.1_eps4.2e-2', 64 : 'combo_iso5e-6_zm0.1_eps1.66e-1',
            256 : 'combo_iso5e-6_zm0.1_eps6.66e-1', 512 : 'combo_iso5e-6_zm0.1_eps1.33', 1024 : 'combo_iso5e-6_zm0.1_eps2.66'}
    unc = c.loc['none']
    panels = [('lvl', 'variance of level means', True, data['level_mean']), ('inv', 'inversions', False, data['inversions']),
              ('inv>0.01', 'inversions > 0.01 kg/m3', True, None), ('mean_density', 'mean_density (median)', True, 6.32e-3),
              ('zero_mean', 'zero_mean (median)', True, ZM_EPS)]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.6))
    scales = np.array(list(runs))
    for ax, (col, label, logy, ref) in zip(axs, panels) :
        ax.plot(scales, [c.loc[r, col] for r in runs.values()], 'o-', label='guided')
        joint = [(s, c.loc[f'{r}_joint0.01', col]) for s, r in runs.items() if f'{r}_joint0.01' in c.index]
        ax.plot(*zip(*joint), '*--', ms=9, label='guided + joint (c = 0.01)')
        ax.axhline(unc[col], ls=':', c='k', label='unconstrained')
        if ref is not None :
            ax.axhline(ref, ls=':', c='g', label='data' if col != 'mean_density' else 'data q0.5')
        if col == 'zero_mean' :
            ax.plot(scales, ZM_EPS*scales, '--', c='0.6', label='eps')
        ax.set_xscale('log')
        if logy :
            ax.set_yscale('log')
        ax.set_xlabel('zero_mean eps / data q0.5')
        ax.set_title(label, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle('dist_isoton + zero_mean sqrt, eta (5e-6, 0.1), 1000 steps, seed 0')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_relax1000.png', dpi=120)


def fig_eta1000(data) :
    """dist_isoton + zero_mean at 1000 steps, eps q0.5, against the zero_mean eta."""
    c = pd.read_csv(f'{OUT}/compare_1000steps_all.csv', index_col=0)
    runs = {0.01 : 'combo_iso5e-6_zm0.01', 0.02 : 'combo_iso5e-6_zm0.02', 0.05 : 'combo_iso5e-6_zm0.05', 0.1 : 'combo_iso5e-6_zm0.1'}
    panels = [('lvl', 'variance of level means', True, data['level_mean']), ('inv', 'inversions', False, data['inversions']),
              ('mean_density', 'mean_density (median)', True, 6.32e-3)]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.4))
    for ax, (col, label, logy, ref) in zip(axs, panels) :
        ax.plot(list(runs), [c.loc[r, col] for r in runs.values()], 'o-')
        ax.axhline(c.loc['none', col], ls=':', c='k', label='unconstrained')
        ax.axhline(ref, ls=':', c='g', label='data')
        ax.set_xscale('log')
        if logy :
            ax.set_yscale('log')
        ax.set_xlabel('zero_mean eta')
        ax.set_title(label, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle('dist_isoton (sqrt, 5e-6) + zero_mean (sqrt), eps q0.5, 1000 steps', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_eta1000.png', dpi=120)


def fig_posthoc(data) :
    """Inversions of post-hoc projection vs guidance, 100 and 1000 steps."""
    local = pd.read_csv(f'{OUT}/local_metrics.csv').groupby('name')['inversions'].median()
    rows = {100 : {'mean_density' : ('posthoc_unc100_mean_density', ['mean_density_sqrt_eta1.0_q0.5_s0_n100_b8']),
                   'negative_grad_density' : ('posthoc_unc100_negative_grad_density', ['negative_grad_density_sqrt_eta1.0_q0.5_s0_n100_b8']),
                   'zero_mean' : ('posthoc_unc100_zero_mean', ['zero_mean_raw_eta10.0_q0.5_s0_n100_b8'])},
            1000 : {'mean_density' : ('posthoc_unc1000_mean_density', ['mean_density_sqrt_eta0.1_eps6.3e-3']),
                    'negative_grad_density' : ('posthoc_unc1000_negative_grad_density', ['negative_grad_density_sqrt_eta0.1_eps0', 'negative_grad_density_sqrt_eta0.3_eps0']),
                    'zero_mean' : ('posthoc_unc1000_zero_mean', ['zero_mean_eta0.4_eps2.6e-3', 'zero_mean_eta1_eps2.6e-3'])}}
    labels = {'mean_density_sqrt_eta1.0_q0.5_s0_n100_b8' : 'sqrt 1', 'negative_grad_density_sqrt_eta1.0_q0.5_s0_n100_b8' : 'sqrt 1',
              'zero_mean_raw_eta10.0_q0.5_s0_n100_b8' : 'raw 10', 'mean_density_sqrt_eta0.1_eps6.3e-3' : 'sqrt 0.1',
              'negative_grad_density_sqrt_eta0.1_eps0' : 'sqrt 0.1', 'negative_grad_density_sqrt_eta0.3_eps0' : 'sqrt 0.3',
              'zero_mean_eta0.4_eps2.6e-3' : 'raw 0.4', 'zero_mean_eta1_eps2.6e-3' : 'raw 1'}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (steps, r) in zip(axs, rows.items()) :
        unc = local['none_s0_n100_b8' if steps == 100 else 'none']
        for i, (loss, (post, guided)) in enumerate(r.items()) :
            ax.bar(i - 0.25, 100*local[post], 0.25, color='C0', label='post-hoc' if i == 0 else None)
            for k, g in enumerate(guided) :
                ax.bar(i + 0.25*k, 100*local[g], 0.25, color='C1', label='guided' if i == 0 and k == 0 else None)
                ax.text(i + 0.25*k, 100*local[g] + 0.3, labels[g], ha='center', fontsize=7)
        ax.axhline(100*unc, ls=':', c='k', label='unconstrained')
        ax.axhline(100*data['inversions'], ls=':', c='g', label='data')
        ax.set_xticks(range(len(r)), list(r), fontsize=8)
        ax.set_title(f'{steps} steps, seed 0', fontsize=10)
    axs[0].set_ylabel('inversions (%)')
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_posthoc.png', dpi=120)


if __name__ == '__main__' :
    data, unc = references()
    fig_calibration()
    fig_combo100(data, unc)
    fig_relax1000(data)
    fig_eta1000(data)
    fig_posthoc(data)
