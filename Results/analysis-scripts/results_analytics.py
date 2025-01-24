import numpy as np
import os, sys
from data_analytics import get_transformed_data
from utils import get_dataloader
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../../Diffusion_Model/')
from configs.base_config import TrainingConfig
os.environ['OCEANDATA'] = '/Volumes/LoCe/oceandata/'


config = TrainingConfig()


train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=False)
idt = iter(train_dataloader)
batch = next(idt)
batch.min()
config.normalisation

generated_batch = torch.tensor(np.load(os.environ['OCEANDATA'] + 'models/dino-fusion/z87envpm/epoch_4950.npy'))
generated_samples = get_transformed_data(generated_batch, transform=train_dataloader.get_transform())

sample = get_transformed_data(batch, transform=train_dataloader.get_transform())


# Manual mask application
for k in sample.keys() :
    generated_samples[k][sample[k].isnan()]  = torch.nan


#%% Imshow for both plot
fig, axs = plt.subplots(1, 2, figsize=(7, 5), constrained_layout=True)
im1 = axs[0].imshow(sample['toce.npy'][0,0], vmin=-2, vmax=2)
axs[0].set_title('Data toce (z=0)')
im2 = axs[1].imshow(generated_samples['toce.npy'][0,0], vmin=-2, vmax=2)
axs[1].set_title('Generated toce (z=0)')

fig.colorbar(im1, ax=axs[:])


#%% Distribution

key = 'toce.npy'
plt.figure(figsize=(10, 6))
sns.histplot(data=sample[key][0,0].flatten(), label='Data', color='blue', alpha=0.5)
sns.histplot(data=generated_samples[key][0,0].flatten(), label='Generated', color='red', alpha=0.5)
plt.legend()
plt.title(f'Distribution Comparison of Data vs Generated Samples for surface {key} (over a batch)')
