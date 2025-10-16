# %% [markdown]
# In this experiment we condition the generation based on a given field 

# %%
from __init__ import PRP; import sys
sys.path.append(PRP + 'Diffusion_model')
sys.path.append(PRP + 'Results/analysis_scripts/')


from data_analytics import get_transformed_data

from configs.base_config import *
from utils import get_dataloader
import matplotlib.pyplot as plt

# %% [markdown]
# # Explore dataset

# %% Hey
config = TrainingConfig()
print(config)

train_dataloader = get_dataloader(config.data_file,
                                  batch_size=200,
                                  fields=config.fields,
                                  normalisation=config.normalisation)

idt = iter(train_dataloader)

extractor = train_dataloader.get_transform().uncall

# %%
batch_normalised = next(idt)
batch = get_transformed_data(batch_normalised, function=extractor)

# %%
plt.plot(batch['toce.npy'].nanmean(axis=(2, 3)).T)
plt.xlabel('Depth')
plt.ylabel('Toce')

# %%

# %% [markdown]
# ## Generate data

# %%

# %%

# %%

# %%
