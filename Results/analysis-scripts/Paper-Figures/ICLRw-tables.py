import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import pandas as pd
from metrics import *


sys.path.append('../../../Diffusion_Model/')
sys.path.append('..')

from utils import get_dataloader

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

file_mask_LR=xr.open_dataset("../data/DINO_1deg_mesh_mask_renamed.nc")
file_mask_LR=file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

# 2. Get metrics
# T500, TBW and TDW during ML Training of the 8 lineages

stats = []

for gen in ['no_constraint', 'constraint', 'data'] :
    for field in ['toce', 'soce'] :
        for b in range(8) :
            data = file_mask_LR.e3t_0.copy()
            batch_raw =  batch if gen == 'data' else bg[gen]
            data[:] = batch_raw[field][b]

            base = {'source' : gen, 'index' : b, 'field' : field}
            stats.append(base | {'500m_30NS' : temperature_500m_30NS_metric(data, file_mask_LR).item(),
                                 'BWbox' : temperature_BWbox_metric(data, file_mask_LR).item(),
                                 'DWbox' : temperature_DWbox_metric(data, file_mask_LR).item()})


stats = pd.DataFrame(stats)
sns.relplot(data=stats, x='BWbox', y='DWbox', hue='source', col='field',facet_kws={'sharey': False, 'sharex': False})
