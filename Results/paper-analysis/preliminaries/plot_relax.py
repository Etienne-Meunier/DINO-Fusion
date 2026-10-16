"""
Relaxation of one constraint (Q2) : metrics of the runs of `sweep.py` against eps_scale (eps = eps_scale * eps_c(q)),
mean and spread over seeds, with the unconstrained runs and the training batches as references.

Panels : final constrained loss, satisfaction, variance over the batch of the per-level spatial means (the subspace the
horizontal-mean constraints act on), variance of the spatial anomalies, fraction of local density inversions, distance
to the unconstrained state.
Needs `outputs/sweep_table.csv` (`analyze_sweep.py`) and `outputs/local_metrics.csv` (`local_metrics.py`).
"""
from __init__ import PRP; import sys

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PANELS = [('loss', 'final loss (median)', True), ('satisfied', 'satisfaction rate', False),
          ('level_mean', 'variance of level means', True), ('anomaly', 'variance of anomalies', True),
          ('inversions', 'local density inversions', False), ('dist_unc', 'distance to unconstrained', False)]


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--loss', type=str, default='zero_mean')
    parser.add_argument('--mode', type=str, default='raw')
    parser.add_argument('--eta', type=float, default=10.0)
    parser.add_argument('--fig', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    args = parser.parse_args()

    table = pd.read_csv(f'{args.fig}/sweep_table.csv')
    local = pd.read_csv(f'{args.fig}/local_metrics.csv')
    data = local[local['name'] == 'data']
    local = local[local['folder'] == 'prelim_sweep_clip'].groupby('name').median(numeric_only=True)
    table = table.join(local, on='name')
    eps_data = pd.read_csv(f'{args.fig}/eps_data_quantiles.csv', index_col=0).loc[args.loss, 'q0.5']

    d = table[(table['loss'] == args.loss) & (table['mode'] == args.mode) & (table['eta'] == args.eta)].copy()
    d = d[d[['t_start', 'lbd_max', 'final_opt', 'through_unet']].isna().all(axis=1)]
    d['eps_scale'] = pd.to_numeric(d['eps_scale'], errors='coerce').fillna(1.0)
    d['loss'] = d[f'loss_{args.loss}']
    unc = table[table['loss'] == 'none']
    unc = unc.assign(loss=unc[f'loss_{args.loss}'])

    fig, axs = plt.subplots(1, len(PANELS), figsize=(3.6*len(PANELS), 3.4))
    for ax, (col, label, logy) in zip(axs, PANELS) :
        g = d.groupby('eps_scale')[col]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt='o-', ms=4, capsize=2)
        if col in unc :
            ax.axhline(unc[col].mean(), ls=':', c='k', label='unconstrained')
        if col in data :
            ax.axhline(data[col].median(), ls=':', c='g', label='data')
        if col == 'loss' :
            ax.axhline(eps_data, ls=':', c='g', label='data q=0.5')
            ax.plot(g.mean().index, eps_data * g.mean().index, '--', c='0.6', label='eps')
        ax.set_xscale('log')
        if logy :
            ax.set_yscale('log')
        ax.set_xlabel('eps_scale')
        ax.set_title(label, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle(f'{args.loss} {args.mode} eta={args.eta} : relaxation of eps ({d["seed"].nunique()} seeds)')
    fig.tight_layout()
    fig.savefig(f'{args.fig}/relax_{args.loss}.png', dpi=130)
