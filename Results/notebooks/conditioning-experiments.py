# %% [markdown]
# In this experiment we condition the generation based on a given field 

# %%
# %load_ext autoreload
# %autoreload 2 
    
from __init__ import PRP; import sys
sys.path.append(PRP + 'Diffusion_model')
sys.path.append(PRP + 'Results/analysis_scripts/')


from data_analytics import get_transformed_data

from configs.base_config import *
from utils import get_dataloader
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from visualisation_utils import *

# Suppress deprecation and future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
s = 7 # Borders
dl = 4

# %% [markdown]
# # Explore dataset

# %% Hey
config = TrainingConfig()
config.data_file='/Volumes/LoCe/oceandata//Dino-Fusion/dino_1_4_degree_coarse_240125.tar'
print(config)

train_dataloader = get_dataloader(config.data_file,
                                  batch_size=3000,
                                  fields=config.fields,
                                  normalisation=config.normalisation, shuffle=False)

idt = iter(train_dataloader)

extractor = train_dataloader.get_transform().uncall

# %%
batch_normalised = next(idt)
batch_unnormalised = get_transformed_data(batch_normalised, function=extractor)

# %%
ssq, lsq = 250, 50
idx_max = ssq + batch_normalised[ssq:ssq+lsq, 0, 100:].mean(axis=(-2, -1)).argmax()
idx_min = ssq + batch_normalised[ssq:ssq+lsq, 0, 100:].mean(axis=(-2, -1)).argmin()

# %%
plt.figure(figsize=(15, 5))
plt.plot(batch_normalised[:, 0, 100:].mean(axis=(-2, -1)))
plt.scatter(idx_max, batch_normalised[idx_max, 0, 100:].mean(axis=(-2, -1)), c='red')
plt.scatter(idx_min, batch_normalised[idx_min, 0, 100:].mean(axis=(-2, -1)), c='green')
plt.title('(Normalised) Surface temperature evolution over time')

# %%
fig, axs = plt.subplots(1,dl + 1 , figsize=(20,3))
fig.suptitle('Surface temperature (north) histogram - Data')
for i, depth in enumerate(range(dl)) :
    for name, idx, cm in [('vmin', idx_min, 'blue'),
                          ('vmax', idx_max, 'orange')] :
        histogram(batch_normalised[idx, depth, s:-s, s:-s],
                  ax=axs[i],
                  label= (name if i == 0 else None), cmap_name=cm)
    axs[i].set_xlabel(f'layer = {depth}')
axs[-1].set_title('Data profile')
axs[-1].set_xlabel('Depth layer')
axs[-1].plot(batch_normalised[idx_min, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmin')
axs[-1].plot(batch_normalised[idx_max, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmax')    
fig.legend()

# %% [markdown]
# ## Generate data

# %%
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from generate_images import get_constraints
from pipelines.constraints import *

# %%
model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/'
beta = 0.003
beta_type = 'constant'
inf_steps = 1000


# %%
class ConditionalGeneration_field():
    def __init__(self, field, index, name):
        self.field = field
        self.index = index
        self.name = name

    def apply(self, x, t=None):
        x[:, self.index] = self.field
        return x

    def __str__(self) : 
        return f'ConditionalGeneration_field {name}'


# %%
pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to('mps')

# %%

# %%
pipeline.constraints = [BorderZeroConstraint(),
                        GradientZeroMeanConstraint(beta=beta, beta_type=beta_type)]


generated_normalized = pipeline(batch_size=3,
                                num_inference_steps=inf_steps,
                                return_dict=False)[0]

# %%
print(pipeline.constraints)
fig, axs = plt.subplots(1,dl, figsize=(20,3))
fig.suptitle(f'Surface temperature histogram - Generation {pipeline}')
for i, depth in enumerate(range(dl)) :
    axs[i].set_xlabel(f'layer = {depth}')
    for idx in range(1) :
        histogram(generated_normalized[idx, depth, s:-s, s:-s].cpu(), ax=axs[i])
fig.legend()

# %%
idx = slice(2,3)
fig, axs = plt.subplots(1,dl, figsize=(20,3))
fig.suptitle(f'Surface temperature histogram \n- Generated {pipeline}')
for i, depth in enumerate(range(dl)) :
    for name, g, cm in [('vmin', generated_normalized_low.cpu(), 'blue'),
                          ('vmax', generated_normalized_high.cpu(), 'orange')] :
        histogram(g[idx, depth, s:-s, s:-s],
                  ax=axs[i],
                  label= (name if i == 0 else None), cmap_name=cm)
    axs[i].set_xlabel(f'layer = {depth}')
    fig.legend()

# %%
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Horizontal Means')
axs[0].set_title('Data')
axs[0].plot(batch_normalised[idx_min, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmin')
axs[0].plot(batch_normalised[idx_max, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmax')

axs[1].set_title('Generated')
axs[1].plot(generated_normalized_low[0, :, s:-s, s:-s].mean(axis=(-2, -1)).cpu(), label='vmin')
axs[1].plot(generated_normalized_high[0, :, s:-s, s:-s].mean(axis=(-2, -1)).cpu(), label='vmax')

# %% [markdown]
# Density

# %%
from metrics import get_density_at_surface_tensor
import torch

# %%
generated_unnormalized_high = get_transformed_data(generated_normalized_high.cpu(), function=extractor)
generated_unnormalized_low = get_transformed_data(generated_normalized_low.cpu(), function=extractor)

# %%
with torch.no_grad() : 
    density = get_density_at_surface_tensor(generated_unnormalized_high['toce.npy'],
                                            generated_unnormalized_high['soce.npy'], tmask=None)

# %%
batch_unnormalised

# %%
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Horizontal Means')
axs[0].set_title('Data')
axs[0].plot(batch_unnormalised['toce.npy'][idx_min, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmin')
axs[0].plot(batch_unnormalised['toce.npy'][idx_max, :, s:-s, s:-s].mean(axis=(-2, -1)), label='vmax')

axs[1].set_title('Generated')
axs[1].plot(generated_unnormalized_low['toce.npy'][0, :, s:-s, s:-s].mean(axis=(-2, -1)).cpu(), label='vmin')
axs[1].plot(generated_unnormalized_high['toce.npy'][0, :, s:-s, s:-s].mean(axis=(-2, -1)).cpu(), label='vmax')

# %%
plt.plot(density[0].nanmean(axis=(-2,-1)).detach())

# %%
