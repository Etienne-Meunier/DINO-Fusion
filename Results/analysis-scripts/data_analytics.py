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

import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'


def get_raw_data(batch: dict) -> torch.Tensor:
    """Default function to extract data from untransformed batch"""
    return batch

def split(data, transform) :
    sample = transform.un_stride_concat(data, interpolate=False)
    for feature in ["soce","toce","ssh"]:
        array = transform.unpadData(sample[feature], **transform.padding)
        if array.shape[0] == 1 :
            array = array[0]
        #array = transform.replaceEdges(array, feature, val=np.nan)
        array[array == 0.0] = np.nan
        sample[feature] =  array
    sample = {key+'.npy' : val for key, val in sample.items()}
    return sample

def get_transformed_data(batch: torch.Tensor, transform) :
    """Function to extract data from transformed batch"""
    samples = [split(b, transform) for b in batch]
    sc = concat_dict(samples)
    return sc

if __name__ =='__main__' :
    #%%
    #%load_ext autoreload
    #%autoreload 2

    # %% Load data
    base_path= '/Volumes/LoCe/oceandata/models/dino-fusion/'
    training_tar = ''
    config = TrainingConfig()

    # %% Without normalisation
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size, fields=config.fields, transform=False, shuffle=False)
    extractor = get_raw_data
    idt = iter(train_dataloader)
    batch = next(idt)
    batch['soce.npy'].shape

    # %% With normalisation
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size,
                                                    fields=config.fields, transform=True, shuffle=False)
    idt = iter(train_dataloader)

    extractor = partial(get_transformed_data, transform=train_dataloader.get_transform())


    # %% Get global metrics
    keys = ['soce.npy', 'toce.npy', 'ssh.npy']
    areas = {'south': slice(5, 40), 'north': slice(170, 205)}

    averages = calculate_area_averages(train_dataloader, keys, areas, data_extractor=extractor)

    plot_area_distributions(averages, z_index=0, title='Data distribution')

    #%% With data from diffusion model
    model_gen = 'z87envpm/epoch_4950.npy'
    batch_diffusion = torch.tensor(np.load(os.environ['OCEANDATA'] + f'models/dino-fusion/{model_gen}'))
    averages = calculate_area_averages([batch_diffusion], keys, areas, data_extractor=extractor)
    plot_area_distributions(averages, z_index=0, title=f'Generated ({model_gen}) distribution')

    #%% Compute min-max over dataset
    operations = {'max' : lambda x : x.nan_to_num(-torch.inf).amax((-2,-1)), 'min' : lambda x : x.nan_to_num(torch.inf).amin((-2,-1))}
    op = operate_on_dataset(train_dataloader, operations, fields=keys, data_extractor=extractor)

    for key in op.keys() :
        print(f"{key} : {op[key]['min'].amin()} - {op[key]['max'].amax()}")


    for key in ['toce', 'soce', 'ssh'] :
        d = op[f'{key}.npy']['min'].amin(dim=0)[..., None, None].numpy()
        np.save(f'add_infos/min.{key}.npy', d)

        d = op[f'{key}.npy']['max'].amax(dim=0)[..., None, None].numpy()
        np.save(f'add_infos/max.{key}.npy', d)
np.load('add_infos/max.soce.npy').shape
