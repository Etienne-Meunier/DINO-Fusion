import sys, os
sys.path.append('../../Diffusion_Model/')
from utils import get_dataloader
from configs.base_config import TrainingConfig
os.environ['OCEANDATA'] = '/Volumes/LoCe/oceandata/'
import numpy as np


config = TrainingConfig()

train_dataloader = get_dataloader(config.data_file, batch_size=8,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
transform = train_dataloader.get_transform()
((transform.infos['max']['toce'] - transform.infos['mean']['toce'])/transform.infos['std']['toce'])[:-1].max()

v1 = np.abs(transform.infos['min']['toce']) [:-1, 0, 0]
v2 = transform.infos['max']['toce'][:-1, 0, 0]
(np.maximum(v1, v2) - transform.infos['mean']['toce'][:-1, 0, 0])/ (6 * transform.infos['std']['toce'][:-1, 0, 0])


v1 = np.abs(transform.infos['min']['soce']) [:-1, 0, 0]
v2 = transform.infos['max']['soce'][:-1, 0, 0]
(np.maximum(v1, v2) - transform.infos['mean']['soce'][:-1, 0, 0])/ (7 * transform.infos['std']['soce'][:-1, 0, 0])
v1 = np.abs(transform.infos['min']['ssh'])
v2 = transform.infos['max']['ssh']
(np.maximum(v1, v2) - transform.infos['mean']['ssh'])/ (7 * transform.infos['std']['ssh'])
transform.infos['min']['ssh'].shape
import matplotlib.pyplot as plt



fig, axs = plt.subplots(1,2, figsize=(15,5))
for i, key in enumerate(['soce', 'toce']) :
    axs[i].set_title(f'{key} infos')
    for info in ['mean', 'max', 'min', 'std'] :
        axs[i].plot(transform.infos[info][key][:,0,0], label=info)
fig.legend()
