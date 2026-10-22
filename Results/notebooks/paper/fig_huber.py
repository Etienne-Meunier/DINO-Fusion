"""Inline figure of Section 3: the one-sided Huber penalty h_tau of the increasing density constraint
(`NegativeGradDensityHuber` in `Diffusion_Model/pipelines/losses.py`), as a function of the vertical density step delta,
in units of tau. Standalone: needs no data. Saved at the final width (about 1.9 in) with 7-8 pt fonts."""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


def h(delta, tau=1.):
    """One-sided Huber: 0 for delta >= 0, delta^2 / (2 tau) on (-tau, 0), -delta - tau / 2 below -tau."""
    return np.where(delta >= 0, 0., np.where(delta < -tau, -delta - tau / 2, delta**2 / (2 * tau)))


d = np.linspace(-3, 1.5, 500)
fig, ax = plt.subplots(figsize=(1.9, 1.3))
ax.axvspan(-1, 0, color='0.9', lw=0)
ax.plot(d, h(d), color='k', lw=1.2)
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
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig('figures/fig_huber.pdf')
