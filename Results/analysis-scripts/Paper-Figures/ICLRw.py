from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys

sys.path.append('../../../Diffusion_Model/')
from utils import get_dataloader
mpl.rcParams['image.origin'] = 'lower'

# 1. Data load
home = '/Users/emeunier/Documents/'
model_path = f'{home}/tav0h83b/'

path = {
    'constraint' : f'{model_path}/inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/20250130-165431_clean',
    'no_constraint' : f'{model_path}/inference/infesteps_1000/constraints_no_constraints/20250131-110120_clean'
    }

bg = {'constraint' : [], 'no_constraint' : []}

for key, p in path.items() :
      bg[key] = {'ssh' : np.load(p + '/ssh.npy'),
                 'soce' : np.load(p + '/soce.npy'),
                 'toce' : np.load(p + '/toce.npy')}

data_file = '/Users/emeunier/Documents/dino_1_4_degree_coarse_240125.tar'
train_dataloader = get_dataloader(data_file, batch_size=8, transform=False, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)
for k in ['toce', 'soce', 'ssh'] :
    batch[k] = batch.pop(f'{k}.npy')

plt.imshow(batch[field][0,id], vmin=min, vmax=max)
plt.contour(batch[field][0,id], vmin=min, vmax=max, colors='black', alpha=0.7, linewidths=0.5)

def show(f, ax, min, max, cmap='viridis', title='') :
      ax.set_title(title)
      im = ax.imshow(f, cmap=cmap, vmin=min, vmax=max)
      return im
      #ax.contour(f, vmin=min, vmax=max, colors='black', alpha=0.4, linewidths=0.3)

# 1. Results visualisation
ks = [0, 17]
fig, axs = plt.subplots(2, len(ks)*2, figsize=(9, 10))
[a.axis('off') for a  in axs.ravel()]
axs[0,0].axis('on')
axs[1, 0].axis('on')
gen = 'constraint'
for j, field in enumerate(['toce', 'soce'] ):
    min = np.nanquantile(batch[field][:, ks, :, :], 0.01)
    max = np.nanquantile(batch[field][:, ks, :, :], 0.99)

    for i, k in enumerate(ks) :
        im = show(batch[field][0,k], axs[j, i], min, max, title=f'data {field} k={k}')
        _ = show(bg[gen][field][0, k], axs[j, i + len(ks)], min, max, title=f'gen {field} k={k}')

    fig.colorbar(im, ax=axs[j, -1])
plt.savefig('results.png')


#2. Variability of the states


fig, axs = plt.subplots(2, 2, figsize=(5, 10))
[a.axis('off') for a in axs.ravel()]
k = 0
for j, field in enumerate(['toce', 'soce']) :
    min = np.nanquantile(np.stack([v[field][:, k] for v in bg.values()]), 0.01)
    max = np.nanquantile(np.stack([v[field][:, k] for v in bg.values()]), 0.99)
    for i, gen in enumerate(['no_constraint', 'constraint']) :
        im = show(bg[gen][field].var(axis=0)[k], axs[i, j], cmap='turbo', min=None, max=None, title=f'{field} {gen} {k}')
        fig.colorbar(im, ax=axs[i, j])




plt.plot(np.nanmean(bg[gen][field], axis=(-2, -1)).T)
plt.show()
import pandas as pd


fig, axs = plt.subplots(2, 2, figsize=(10,10))
z = 0
for j, field in enumerate(['toce', 'soce']) :
    for i, gen in enumerate(['no_constraint', 'constraint']) :
        df_sample = pd.DataFrame(np.nanmean(batch[field][:,z, 5:-5, 5:-5], axis=-2)).melt(var_name='height')
        df_sample['type'] ='data'

        df_gen = pd.DataFrame(np.nanmean(bg[gen][field][:,z, 5:-5, 5:-5], axis=-2)).melt(var_name='height')
        df_gen['type'] = f'gen {gen}'
        df = pd.concat([df_sample, df_gen]).reset_index()

        sns.lineplot(data=df, x='height', y='value', hue='type', orient='x', ax=axs[i, j])
        plt.title(f'N-S profile {field})')
