import torch
from tools import *
import numpy as np


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
