import numpy as np
from common import *


base = '/Volumes/LoCe/oceandata/models/dino-fusion/'


# Publi blandine

bland = 'Publication-Blandine/generated_npy_files/no_constraint_clean/toce.npy'
bland_toce = np.load(f'{base}/{bland}')

print('blandine')
np.nanmean(bland_toce[:,0], axis=(-1, -2))

data_toce = observer.unnormalized(batch)['toce']
print('data')
np.nanmean(data_toce[:8,0], axis=(-1, -2))


gen_path = f'{base}/tav0h83b/inference/default_eps/none_border0/151026-150424/generated.npy'
gen = torch.from_numpy(np.load(gen_path)).to(device)

gen_toce = observer.unnormalized(gen)['toce']

print('new_gen')
np.nanmean(gen_toce[:8,0], axis=(-1, -2))

fig, axs = plt.subplots(1,1)
sns.histplot(np.nanmean(data_toce[:,0], axis=(-1, -2)), ax=axs, label='data')
sns.histplot(np.nanmean(bland_toce[:,0], axis=(-1, -2)), ax=axs, label='bland')
sns.histplot(np.nanmean(gen_toce[:,0], axis=(-1, -2)), ax=axs, label='gen')


# Box metrics 

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric

file_mask_LR = load_mask()

box_metrics = {'BWbox': partial(temperature_BWbox_metric, depth_box=1500),
               'DWbox': partial(temperature_DWbox_metric, depth_box=1500)}



rB = {'data' : [], 'bland' : [], 'gen' : []}
rD = {'data' : [], 'bland' : [], 'gen' : []}

for i in range(8) :
    sample = file_mask_LR.e3t_0.copy()
    sample[:] = np.asarray(data_toce[i])
    rB['data'].append(box_metrics['BWbox'](sample, file_mask_LR).item())
    rD['data'].append(box_metrics['DWbox'](sample, file_mask_LR).item())

    sample = file_mask_LR.e3t_0.copy()
    sample[:] = np.asarray(bland_toce[i])
    rB['bland'].append(box_metrics['BWbox'](sample, file_mask_LR).item())
    rD['bland'].append(box_metrics['DWbox'](sample, file_mask_LR).item())

    sample = file_mask_LR.e3t_0.copy()
    sample[:] = np.asarray(gen_toce[i])
    rB['gen'].append(box_metrics['BWbox'](sample, file_mask_LR).item())
    rD['gen'].append(box_metrics['DWbox'](sample, file_mask_LR).item())

{k : np.array(rB[k]).mean() for k in rB.keys()}


{k : np.array(rD[k]).mean() for k in rD.keys()}

np.array(r['bland']).mean()

np.array(r['gen']).mean()
