# %% [markdown]
# In this experiment we condition the generation based on a given field 

# %%
from __init__ import PRP; import sys
sys.path.append(PRP)

from configs.base_config import *

# %% [markdown]
# # Explore dataset

# %% Hey
config = TrainingConfig()

train_dataloader = get_dataloader(config.data_file,
                                  batch_size=config.train_batch_size,
                                  fields=config.fields,
                                  normalisation=config.normalisation)

# %% [markdown]
#

# %%
