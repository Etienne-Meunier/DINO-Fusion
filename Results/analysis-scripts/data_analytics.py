import torch
import sys, os

from torchvision import transforms
os.environ['OCEANDATA'] = '/Volumes/LoCe/oceandata/'

sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
from utils import get_dataloader, get_transform
from tools import *
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'


def get_raw_data(batch: dict) -> torch.Tensor:
    """Default function to extract data from untransformed batch"""
    return batch

def split(data, transform) :
    sample = transform.un_stride_concat(data, interpolation=False)
    for feature in ["soce","toce","ssh"]:
        array = transform.unpadData(sample[feature], **transform.padding)
        if array.shape[0] == 1 :
            array = array[0]
        #array = transform.replaceEdges(array, feature, val=np.nan)
        array[array == 0.0] = np.nan
        sample[feature] =  array
    sample = {key+'.npy' : val for key, val in sample.items()}
    return sample

def get_transformed_data(batch: torch.Tensor, function) :
    """Function to extract data from transformed batch"""
    samples = [function(b) for b in batch] # split(b, transform)
    sc = concat_dict(samples)
    return sc

if __name__ =='__main__' :
    config = TrainingConfig()
    config.normalisation = 'std'

    train_dataloader = get_dataloader(config.data_file, batch_size=8,
                                                    fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
    idt = iter(train_dataloader)

    #%% Un-normalisation : turn the batch to a dict
    # Re-normalisation : bring back the data to it's initial scal
    RENORMALISATION = False

    if RENORMALISATION :
        extractor = train_dataloader.get_transform().uncall
    else :
        extractor = partial(split, transform=train_dataloader.get_transform())


    #%% On a batch
    batch = next(idt)
    samples = get_transformed_data(batch, function=extractor)

    # Vertical profiles
    fig, axs = plt.subplots(2,1, figsize=(20,7))
    axs[0].set_title(f'soce {config.normalisation} - Renormalisation : {RENORMALISATION}')
    im0 = axs[0].imshow(samples['soce.npy'][1].nanmean(axis=-1))
    axs[0].invert_yaxis()
    plt.colorbar(im0, ax=axs[0])

    axs[1].set_title(f'toce {config.normalisation}  - Renormalisation : {RENORMALISATION}')
    im1 = axs[1].imshow(samples['toce.npy'][1].nanmean(axis=-1))
    axs[1].invert_yaxis()
    plt.colorbar(im1, ax=axs[1])

    #Distribution
    fig, axs = plt.subplots(2, 1, figsize=(18, 6))
    fig.suptitle(f'Depth distribution {config.normalisation} - Renormalisation : {RENORMALISATION}')
    for i, key in enumerate(['toce.npy', 'soce.npy']) :
        df = pd.DataFrame(samples[key][0, :-1, 5:-5, 5:-5].flatten(1).T)
        sns.violinplot(data=df, ax=axs[i])
        axs[i].set_xlabel('Depth layer')
        axs[i].set_ylabel(f'{key}')


    # %% Get global metrics
    keys = ['soce.npy', 'toce.npy', 'ssh.npy']
    areas = {'south': slice(5, 40), 'north': slice(170, 205)}

    #averages = calculate_area_averages(train_dataloader, keys, areas, data_extractor=lambda x : get_transformed_data(x, function=extractor))
    averages = calculate_area_averages(train_dataloader, keys, areas, data_extractor=lambda x : x)
    plot_area_distributions(averages, z_index=0, title='Data distribution')




    #%% Compute min-max over dataset
    #operations = {'max' : lambda x : x.nan_to_num(-torch.inf).amax((-2,-1)), 'min' : lambda x : x.nan_to_num(torch.inf).amin((-2,-1))}
    #op = operate_on_dataset(train_dataloader, operations, fields=keys, data_extractor=extractor)


    # Distribution

    bins = {'soce.npy' : 100, 'toce.npy' : 100}
    values = {'soce.npy' : np.zeros(100), 'toce.npy' : np.zeros(100)}
    for batch in tqdm(train_dataloader) :
        for key in ['toce.npy', 'soce.npy'] :
            nvalues, nbins = np.histogram(samples[key][:, :-1, 5:-5, 5:-5].nan_to_num_(0.0), bins=bins[key])
            if type(bins[key]) is int :
                bins[key] = nbins
            values[key] += nvalues

    fig, axs = plt.subplots(2, 1, figsize=(15,7))
    for i, key in enumerate(['toce.npy', 'soce.npy']) :
        axs[i].set_title(f'Value distribution (global) {key} {config.normalisation}  - Renormalisation : {RENORMALISATION}')
        axs[i].bar(bins[key][:-1], values[key], width=np.diff(bins[key]), align='edge',
                                                                        alpha=0.7,           # Add some transparency
                                                                        color='cornflowerblue',  # Nice blue color
                                                                        edgecolor='black',   # Black edges
                                                                        linewidth=0.5)       # Thinner edges
