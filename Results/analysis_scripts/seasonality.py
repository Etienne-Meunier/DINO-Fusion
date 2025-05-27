from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import torch
sys.path.append('../../Diffusion_Model/')

from configs.base_config import TrainingConfig

from utils import get_dataloader
mpl.rcParams['image.origin'] = 'lower'

# 1. Data load
config = TrainingConfig()
config.normalisation = '3-std'

train_dataloader = get_dataloader(config.data_file, batch_size=3000,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)
colors = plt.cm.viridis(np.linspace(0, 1, 17))
fig, axs = plt.subplots(3, 1, figsize=(10,8))
axs[0].set_title('ssh (north) [140:]')
axs[0].plot(batch[:, -1,   130:].mean(axis=(-2, -1)))
axs[1].set_title('surface temperature (north) [100:')
axs[1].plot(batch[:, 0,   100:].mean(axis=(-2, -1)))
axs[2].set_title('surface temperature (south) [:40]')
axs[2].plot(batch[:, 0,   :40].mean(axis=(-2, -1)))
plt.legend()
batch.shape

idx_key, idx_name = (0, 'surface_temp')
fields = {}
for name, function  in {'low' : torch.min, 'high' : torch.max}.items() :
    idx = function(batch[:, idx_key,   100:].mean(axis=(-2, -1)), 0).indices.item()
    np.save(f'../../Diffusion_Model/pipelines/{idx_name}_{name}.npy', batch[idx, idx_key].numpy())
    fields[f'{idx_name}_{name}'] = batch[idx, :]
plt.figure(figsize=(20, 5))
plt.plot(batch.mean(axis=(-2,-1)).T)
plt.show()
fields.keys()


plt.imshow(fields['surface_temp_high'][i, 7:-7, 7:-7] > 0.4)

s = 7
fig, axs = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Data temperature histogram (normalized)')
for i in range(5) :
    axs[i].set_title(f'layer : {i*2}')
    sns.histplot(fields['surface_temp_low'][i, s:-s, s:-s].flatten(), ax=axs[i], label='Data(Low surface temp)' if i ==0 else None)
    sns.histplot(fields['surface_temp_high'][i, s:-s, s:-s].flatten(), ax=axs[i], label='Data(High surface temp)'  if i ==0 else None)
fig.legend()

# See results
stemplow = np.load('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_gradient_zero_mean_conditional_generation_stemplow/20250318-191341.npy')
stemphigh = np.load('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_gradient_zero_mean_conditional_generation_stemphigh/20250318-191709.npy')
sshlow = np.load('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_gradient_zero_mean_conditional_generation_sshlow/20250318-172247.npy')
sshhigh = np.load('/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference/infesteps_1000/constraints_gradient_zero_mean_conditional_generation_sshhigh/20250318-172822.npy')


plt.figure(figsize=(20,5))
plt.plot(stemphigh[0].mean(axis=(-2, -1)).T, label='stemphigh')
plt.plot(stemplow[0].mean(axis=(-2, -1)).T, label='stemplow')
plt.plot(fields['surface_temp_low'].mean(axis=(-2, -1)).T, label='data_stemplow')
plt.plot(fields['surface_temp_high'].mean(axis=(-2, -1)).T, label='data_stemphigh')
plt.legend()

fig, axs = plt.subplots(2,37, figsize=(25, 5))
[a.axis('off')  for a in axs.flatten()]
for b in range(37) :
    im1 = axs[0, b].imshow(sshlow_gen[0,b], vmin=-1, vmax=1)
    axs[1, b].imshow(sshhigh_gen[0, b], vmin=-1, vmax=1)

import napari
v = napari.Viewer()
sshlow_gen[0].shape

plt.plot(stemphigh[0].mean(axis=(-2, -1)).T)
plt.plot(stemplow[0].mean(axis=(-2, -1)).T)

fig, ax = plt.subplots()
sns.histplot(stemplow[:, 0].flatten(), ax=ax, label='generated stemp low')
sns.histplot(stemphigh[:, 0].flatten(), ax=ax, label='generated stemp high')
fig.legend()

print("Number of zeros in stemplow:", np.sum(stemplow[:, 1] == 0))
print("Number of zeros in stemphigh:", np.sum(stemphigh[:, 1] == 0))

plt.imshow(stemplow[0, i, s:-s, s:-s] > 0.4)

fig, axs = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle('Generated data temperature histogram (normalized)')
for i in range(5) :
    axs[i].set_title(f'layer : {i*2}')
    sns.histplot(stemplow[:, i, s:-s, s:-s].flatten(), ax=axs[i], label='Condition(Low surface temp)' if i ==0 else None)
    sns.histplot(stemphigh[:, i, s:-s, s:-s].flatten(), ax=axs[i], label='Condition(High surface temp)'  if i ==0 else None)
fig.legend()

fig, ax = plt.subplots()
sns.histplot(stemplow[:, 1, 5:-5, 5:-5].flatten(), ax=ax, label='generated stemp low')
sns.histplot(stemphigh[:, 1, 5:-5, 5:-5].flatten(), ax=ax, label='generated stemp high')
fig.legend()



fig, ax = plt.subplots()
sns.histplot(stemplow[:, 2].flatten(), ax=ax, label='generated stemp low')
sns.histplot(stemphigh[:, 2].flatten(), ax=ax, label='generated stemp high')
fig.legend()
