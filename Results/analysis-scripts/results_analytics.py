import profile
from ipywidgets.widgets.widget_string import Label
import numpy as np
import os, sys
from data_analytics import get_transformed_data, split
from utils import get_dataloader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from functools import partial
from pathlib import Path
from tqdm import tqdm
from itertools import product
import pandas as pd
from dataclasses import dataclass
import napari

sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
os.environ['OCEANDATA'] = '/Volumes/LoCe/oceandata/'

def save_fig(fig, path) :
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path)

@dataclass
class m :
   path : str
   normalisation :  str

   def __str__(self) -> str:
       return f'{self.path} - {self.normalisation}'

models  = {'std training' :  m('z87envpm/epoch_4950.npy', 'std'),
           'min-max' :  m('hxdnrm4i/epoch_4950.npy', 'min-max'),
           'global-min-max' :  m('0z2hm5m9/epoch_3350.npy', 'global-min-max'),
           'min-max ft' :  m('ji71na3g/epoch_3350.npy', 'min-max'),
           'red' :  m('l5680eoz/epoch_4550.npy', 'min-max'),
           '3-std' :  m('tav0h83b/epoch_4950.npy', '3-std'),
           '7-std' :  m('wttzzdt9/epoch_4950.npy', '7-std')}
model = models['3-std']


config = TrainingConfig()
config.normalisation = model.normalisation
config.data_file = '/Users/emeunier/Documents/dino_1_4_degree_coarse_240125.tar'

train_dataloader = get_dataloader(config.data_file, batch_size=8,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)

batch.shape

#model_path = os.environ['OCEANDATA'] + f'models/dino-fusion/{model.path}'
#Large, no constraint : '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_no_constraints/20250131-110120.npy'
#Large, constraint : '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/20250203-175158.npy'
home='/Users/emeunier/Documents/tav0h83b/'
model_path = f'{home}/inference/infesteps_1000/constraints_no_constraints/20250203-141645.npy'
generated_batch = torch.tensor(np.load(model_path)) #

RE_CENTER_GENERATED = False
if RE_CENTER_GENERATED :
    generated_batch -= generated_batch.mean(axis=(-2, -1), keepdim=True)

#%% Un-normalisation : turn the batch to a dict

# Re-normalisation : bring back the data to it's initial scal
RENORMALISATION = True

# Without re-normalisation

if RENORMALISATION :
    extractor = train_dataloader.get_transform().uncall
    generated_samples = get_transformed_data(generated_batch, function=extractor)
    samples = get_transformed_data(batch, function=extractor)
else :
    extractor = partial(split, transform=train_dataloader.get_transform())
    generated_samples = get_transformed_data(generated_batch, function=extractor)
    samples = get_transformed_data(batch, function=extractor)
    # Manual mask application
    for k in samples.keys() :
        generated_samples[k][samples[k].isnan()]  = torch.nan

# Figure

def profile_comparison(samples, generated_samples, key, z, b=0) :
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2b = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    min, max = samples[key][b,z].nan_to_num(np.inf).amin(), samples[key][0,0].nan_to_num(-np.inf).amax()
    im1 = ax1.imshow(samples[key][b,z], vmin=min, vmax=max)
    ax1.set_title(f'Data {key} (z={z})')
    im2 = ax2.imshow(generated_samples[key][0,z], vmin=min, vmax=max)
    ax2.set_title(f'Generated ({model}) \n {key} (z={z})')
    fig.colorbar(im1, ax=ax2)

    ax2b.set_title('North-south profile')

    df_sample = pd.DataFrame(samples[key][:,z].nanmean(axis=-1)).melt(var_name='height')
    df_sample['type'] ='data'

    df_gen = pd.DataFrame(generated_samples[key][:,z].nanmean(axis=-1)).melt(var_name='height')
    df_gen['type'] ='gen'
    df = pd.concat([df_sample, df_gen]).reset_index()

    sns.lineplot(data=df, y='height', x='value', hue='type', orient='y', ax=ax2b)
    plt.title(f'N-S profile {key} ({model})')


    sns.histplot(data=samples[key][:,z].flatten(), label='Data', color='blue', alpha=0.5, ax=ax3)
    sns.histplot(data=generated_samples[key][:,z].flatten(), label='Generated', color='red', alpha=0.5, ax=ax3)
    plt.legend()
    plt.title(f'Distribution Comparison of Data vs Generated ({model}) Samples for z={z} {key} (over a batch)')
    return fig
profile_comparison(samples, generated_samples, 'soce.npy', 10, b=0);

FULL_GENERATION  = False # Generate images for the full profile

if FULL_GENERATION :
    for key, z in tqdm(product(['toce.npy', 'soce.npy'], range(36))):
            fig = profile_comparison(samples, generated_samples, key, z, 0);
            save_fig(fig, f'{model_path.replace(".npy", "")}_{config.normalisation}/{key}_z={z}_renormalised={RENORMALISATION}_recentered={RE_CENTER_GENERATED}.png')
            plt.close()

# Comparison vertical profiles

fig, axs = plt.subplots(1,2, figsize=(15,5))
fig.suptitle(f'vertical profiles blue : data orange :{model}')
for i, key in enumerate(['toce.npy', 'soce.npy']) :
    axs[i].set_title(f'{key}')
    axs[i].set_xlabel('Depth')
    axs[i].plot(samples[key].nanmean(axis=(-2, -1)).T, label='data', c='blue')
    axs[i].plot(generated_samples[key].nanmean(axis=(-2, -1)).T, label='generation', c='orange')


# Check max values from "info" file
key = 'toce.npy'
fig, axs = plt.subplots(2,1, figsize=(15,7))
fig.suptitle(f'Zonal integral -  {key}')

im1 = axs[0].imshow(samples[key][0].nanmean(axis=(-1)))
axs[0].invert_yaxis()
plt.colorbar(im1, ax=axs[0])
axs[0].set_title('Data')

im0 = axs[1].imshow(generated_samples[key][0].nanmean(axis=(-1)))
plt.colorbar(im0, ax=axs[1])
axs[1].invert_yaxis()
axs[1].set_title(f'{model}')




SAVE_CLEAN = True
if SAVE_CLEAN :
    # Deal with bottom boundary
    generated_samples['toce.npy'][:,:,1, :] = generated_samples['toce.npy'][:,:,2, :]
    generated_samples['soce.npy'][:,:,1, :] = generated_samples['soce.npy'][:,:,2, :]
    generated_samples['ssh.npy'][:,1, :] = generated_samples['ssh.npy'][:,2, :]

    save_dir = model_path.replace('.npy', '_clean/')
    Path(save_dir).mkdir(exist_ok=True)

    np.save(save_dir + 'toce.npy', generated_samples['toce.npy'].numpy())
    np.save(save_dir + 'soce.npy', generated_samples['soce.npy'].numpy())
    np.save(save_dir + 'ssh.npy', generated_samples['ssh.npy'][:, None].numpy())
    print(f'Saved clean to : {save_dir}')
