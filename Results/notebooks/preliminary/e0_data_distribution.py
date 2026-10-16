#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common import *

losses = {name : get_loss(name, observer, volume_weighted=False) for name in losses}


fig, axs = plt.subplots(len(losses), 2, figsize=(20, 4 * len(losses)), squeeze=False)
for i, l in enumerate(losses.keys()):
    for j, (name, x) in enumerate([('data', batch), ('generated', gen)]):
        v = np.asarray(losses[l](x).detach().cpu()).ravel()
        ax = axs[i, j]
        sns.histplot(x=v, ax=ax)
        med = np.median(v)
        ax.axvline(med, color='k', ls='--', lw=1, label=f'median = {med:.3g}')
        ax.legend(fontsize=8)
        ax.set_title(f'{l} - {name}')
fig.tight_layout()

