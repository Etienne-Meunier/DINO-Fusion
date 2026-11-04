"""Inline figure of Section 3: the one-sided Huber penalty h_tau of the increasing density constraint
(`NegativeGradDensityHuber` in `Diffusion_Model/pipelines/losses.py`), as a function of the vertical density step delta,
in units of tau. Standalone: needs no data and no dinostics. Saved at the final width (about 1.9 in) with 7-8 pt fonts
in the `figures` folder of the working directory (PAPER_FIGURES), this folder when run from it."""
import os

import numpy as np
import matplotlib.pyplot as plt

FIGURES = os.environ.get('PAPER_FIGURES', 'figures')


def h(delta, tau=1.):
    """One-sided Huber: 0 for delta >= 0, delta^2 / (2 tau) on (-tau, 0), -delta - tau / 2 below -tau."""
    return np.where(delta >= 0, 0., np.where(delta < -tau, -delta - tau / 2, delta**2 / (2 * tau)))


def compute():
    """delta on [-3, 1.5] (units of tau) and h_tau(delta) for tau = 1: {'delta', 'h'}."""
    delta = np.linspace(-3, 1.5, 500)
    return {'delta': delta, 'h': h(delta)}


def plot(computed):
    """Figure of h_tau of `computed` (`compute`) and its linear asymptote."""
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    d = computed['delta']
    fig, ax = plt.subplots(figsize=(1.9, 1.3))
    ax.axvspan(-1, 0, color='0.9', lw=0)
    ax.plot(d, computed['h'], color='k', lw=1.2)
    ax.plot(d[d < 0], -d[d < 0], color='0.5', lw=0.8, ls='--')
    ax.set_xticks([-1, 0])
    ax.set_xticklabels([r'$-\tau$', r'$0$'])
    ax.set_yticks([0])
    ax.set_xlabel(r'$\delta$', labelpad=1)
    ax.set_ylabel(r'$h_\tau(\delta)$', labelpad=1)
    ax.text(-2.9, 0.75, 'inversion', fontsize=7, va='bottom')
    ax.text(0.15, 0.75, 'stable', fontsize=7, va='bottom')
    ax.set_xlim(-3, 1.5)
    ax.set_ylim(-0.1, 3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


def main():
    """Compute, plot and save the figure as fig_huber.pdf in FIGURES."""
    fig = plot(compute())
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f'{FIGURES}/fig_huber.pdf')


if __name__ == '__main__':
    main()
