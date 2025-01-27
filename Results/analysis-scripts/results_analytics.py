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

def save_fig(fig, path) :
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path)


sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
os.environ['OCEANDATA'] = '/Volumes/LoCe/oceandata/'


config = TrainingConfig()

train_dataloader = get_dataloader(config.data_file, batch_size=8,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)

model = 'hxdnrm4i/epoch_4950.npy' #'z87envpm/epoch_4950.npy' #
model_path = os.environ['OCEANDATA'] + f'models/dino-fusion/{model}'
generated_batch = torch.tensor(np.load(model_path)) #

#%% Un-normalisation : turn the batch to a dict

# Re-normalisation : bring back the data to it's initial scal
RENORMALISATION = False

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
    for k in sample.keys() :
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
profile_comparison(samples, generated_samples, 'toce.npy', 0, b=0);

FULL_GENERATION  = False # Generate images for the full profile

if FULL_GENERATION :
    for key, z in tqdm(product(['toce.npy', 'soce.npy'], range(36))):
            fig = profile_comparison(samples, generated_samples, key, z, 0);
            save_fig(fig, f'{model_path.replace('.npy', '')}/{key}_z={z}.png')
            plt.close()


for k in range(18) :
    plt.hist(generated_samples['toce.npy'][:, k].flatten(), bins=50, alpha=0.5)
    plt.hist(samples['toce.npy'][:, k].flatten(), bins=50, alpha=0.5)
    plt.show()

# Comparison vertical profiles

fig, axs = plt.subplots(1,2, figsize=(15,5))
fig.suptitle(f'vertical profiles blue : data orange :{model}')
for i, key in enumerate(['toce.npy', 'soce.npy']) :
    axs[i].set_title(f'{key}')
    axs[i].set_xlabel('Depth')
    axs[i].plot(sample[key].nanmean(axis=(-2, -1)).T, label='data', c='blue')
    axs[i].plot(generated_samples[key].nanmean(axis=(-2, -1)).T, label='generation', c='orange')
# Check max values from "info" file

plt.figure(figsize=(15,5))
plt.plot(train_dataloader.get_transform().infos['max']['toce'][:, 0,0], label='global max data', c='blue')
#plt.plot(train_dataloader.get_transform().infos['min']['toce'][:, 0,0], label='global min data', c='blue')

plt.plot(generated_samples['toce.npy'].nan_to_num(-np.inf).amax(axis=(0, 2, 3)), label='batch max gen', c='orange')
plt.legend()
train_dataloader.get_transform().infos['max']['ssh'].max()
train_dataloader.get_transform().infos['min']['ssh'].min()
