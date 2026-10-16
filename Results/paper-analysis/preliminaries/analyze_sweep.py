"""
Aggregate the runs of `sweep.py` (one `metrics.json` per run folder) into a table and figures.

Outputs in `--fig` :
    sweep_table.csv : one row per run (arguments, median final loss of each constraint, satisfaction, distance to the
                      unconstrained state, diversity)
    sweep_frontier.png : final loss against distance to unconstrained / diversity for every run (`plot_frontier`)
    sweep_<loss>.png : for each constrained loss, target loss / satisfaction / distance / diversity against eta, one line
                       per (mode, t_start, q, lbd_max, final_opt), dotted lines : unconstrained run and data references
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipelines.losses import LOSSES

VARIANT = ['mode', 'eps_scale', 't_start', 'q', 'lbd_max', 'final_opt', 'through_unet']


def load_table(out) :
    """One row per run folder of `out`."""
    rows = []
    for p in sorted(glob.glob(f'{out}/*/metrics.json')) :
        m = json.load(open(p))
        r = {'name' : m['name'], **{k : m['args'].get(k) for k in ['loss', 'eta', 'seed', 'steps', 'batch'] + VARIANT},
             'eps' : m['eps'], 'finite' : m['finite'], 'satisfied' : m.get('satisfied'), 'time' : m['time']}
        if isinstance(r['eta'], list) :
            r['eta_list'], r['eta'], r['eps'] = str(r['eta']), np.nan, np.nan
        for n, v in m['losses'].items() :
            r[f'loss_{n}'] = np.nanmedian(v) if m['finite'] else np.nan
        r['dist_unc'] = np.mean(m['dist_unc']) if m['dist_unc'] is not None else np.nan
        for k, v in (m['diversity'] or {}).items() :
            r[f'div_{k}'] = v
        r['lbd_med'] = np.nanmedian(m['lbd']) if m['lbd'] is not None else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def data_quantile(eps_csv, loss, q=0.5) :
    """eps_c(q) of `eps_data.py`."""
    df = pd.read_csv(eps_csv, index_col=0)
    return df.loc[loss, f'q{q}']


def plot_loss(df, loss, ref, eps_csv, path=None) :
    """Figure of the module docstring for one constrained loss."""
    d = df[df['loss'] == loss].copy()
    if len(d) == 0 :
        return None
    unc = df[df['loss'] == 'none']
    d['variant'] = d[VARIANT].apply(lambda r : ' '.join('-' if pd.isna(v) else str(v) for v in r), axis=1)
    panels = [(f'loss_{loss}', 'final loss (median)', True), ('satisfied', 'satisfaction rate', False),
              ('dist_unc', 'RMS distance to unconstrained', True), ('div_all', 'diversity (batch std)', False)]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.5*len(panels), 3.8))
    for ax, (col, label, logy) in zip(axs, panels) :
        for v, g in d.groupby('variant') :
            g = g.groupby('eta')[col].mean().reset_index()
            ax.plot(g['eta'], g[col], 'o-', ms=3, label=v)
        if col.startswith('loss') :
            ax.axhline(unc[col].median(), ls=':', c='k', label='unconstrained')
            ax.axhline(max(data_quantile(eps_csv, loss), 1e-12), ls=':', c='g', label='data q=0.5')
        if col == 'div_all' :
            ax.axhline(unc[col].median(), ls=':', c='k')
            ax.axhline(np.median(ref['diversity']['all']), ls=':', c='g')
        if col == 'dist_unc' :
            ax.axhline(np.median(ref['distance']), ls=':', c='g')
        ax.set_xscale('log')
        if logy :
            ax.set_yscale('log')
        ax.set_xlabel('eta')
        ax.set_title(label, fontsize=9)
    axs[0].legend(fontsize=6, title=' '.join(VARIANT), title_fontsize=6)
    fig.suptitle(loss)
    fig.tight_layout()
    if path is not None :
        fig.savefig(path, dpi=130)
    return fig


def plot_frontier(df, ref, eps_csv, path=None) :
    """
        Trade-off of every constrained run of each single loss : final target loss against the distance to the
        unconstrained state (left) and against the diversity (right), one colour per mode, marker size growing with eta.
        Dotted lines : unconstrained run and data references. A loss of exactly 0 is drawn at 1e-12.
    """
    losses = [l for l in LOSSES if (df['loss'] == l).any()]
    unc = df[df['loss'] == 'none']
    fig, axs = plt.subplots(2, len(losses), figsize=(4*len(losses), 7), squeeze=False)
    for k, loss in enumerate(losses) :
        d = df[(df['loss'] == loss) & df['finite']]
        for (mode, variant), g in d.groupby([d['mode'], d[[v for v in VARIANT if v not in ['mode', 'q']]].apply(lambda r : any(not pd.isna(v) for v in r), axis=1)]) :
            label = f'{mode}' + (' variants' if variant else '')
            eta_min = df.loc[df['eta'] > 0, 'eta'].min()
            size = (10 + 8 * (np.log10(g['eta'].clip(lower=eta_min)) - np.log10(eta_min))).fillna(10)
            for ax, col in zip(axs[:, k], ['dist_unc', 'div_all']) :
                ax.scatter(g[col], g[f'loss_{loss}'].clip(lower=1e-12), s=size, alpha=0.7, marker='o' if not variant else 'x', label=label)
        for ax, col, r in zip(axs[:, k], ['dist_unc', 'div_all'], [np.median(ref['distance']), np.median(ref['diversity']['all'])]) :
            ax.axhline(unc[f'loss_{loss}'].median(), ls=':', c='k')
            ax.axhline(max(data_quantile(eps_csv, loss), 1e-12), ls=':', c='g')
            ax.axvline(r, ls=':', c='g')
            if col == 'div_all' :
                ax.axvline(unc['div_all'].median(), ls=':', c='k')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(col)
        axs[0, k].set_title(loss)
    axs[0, 0].set_ylabel('final loss (median)')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    if path is not None :
        fig.savefig(path, dpi=130)
    return fig


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=str, default=f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/')
    parser.add_argument('--fig', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    parser.add_argument('--eps_csv', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/eps_data_quantiles.csv')
    parser.add_argument('--ref', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/data_reference.json')
    args = parser.parse_args()

    df = load_table(args.out)
    df.to_csv(f'{args.fig}/sweep_table.csv', index=False)
    ref = json.load(open(args.ref))
    for loss in LOSSES :
        plot_loss(df, loss, ref, args.eps_csv, path=f'{args.fig}/sweep_{loss}.png')
    plot_frontier(df, ref, args.eps_csv, path=f'{args.fig}/sweep_frontier.png')
    print(f'{len(df)} runs')
