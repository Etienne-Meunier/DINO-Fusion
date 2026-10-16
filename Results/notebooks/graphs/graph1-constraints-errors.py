from common import *

eps = eps_data[False]

fig, axs = plt.subplots(len(sources), len(losses), sharex='col',
                        figsize=(5 * len(losses), 3 * len(sources)), squeeze=False)
for i, (source, x) in enumerate(sources.items()):
    for j, loss in enumerate(losses):
        v = np.asarray(losses[loss](x).detach().cpu()).ravel()
        ax = axs[i, j]
        sns.histplot(x=v, ax=ax)
        med = np.median(v)
        ax.axvline(med, color='k', ls='--', lw=1, label=f'median = {med:.3g}')
        if eps[loss] is not None:
            ax.axvline(eps[loss], color='r', lw=1.5, label=f'eps = {eps[loss]:.3g}')
        ax.legend(fontsize=8)
        ax.set_title(f'metric : {loss} - constraint : {source}', fontsize=9)
fig.tight_layout()
