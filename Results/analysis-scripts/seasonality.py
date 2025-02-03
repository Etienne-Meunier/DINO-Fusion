from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
from configs.base_config import TrainingConfig

sys.path.append('../../Diffusion_Model/')
from utils import get_dataloader
mpl.rcParams['image.origin'] = 'lower'

# 1. Data load
config = TrainingConfig()
config.normalisation = '3-std'

data_file = '/Users/emeunier/Documents/dino_1_4_degree_coarse_240125.tar'
train_dataloader = get_dataloader(config.data_file, batch_size=100,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=False)
idt = iter(train_dataloader)
batch = next(idt)
colors = plt.cm.viridis(np.linspace(0, 1, 17))
fig, axs = plt.subplots(3, 1, figsize=(15,5))
axs[0].plot(batch[:, 0:17].mean(axis=(-2, -1)), label='temperature', c=colors.T)
axs[1].plot(batch[:, 17:-1].mean(axis=(-2, -1)), label='salinity')
axs[2].plot(batch[:, -1:].mean(axis=(-2, -1)), label='ssh')

#plt.plot(batch[:, 17].mean(axis=(-2, -1)), label='surface salinity')
