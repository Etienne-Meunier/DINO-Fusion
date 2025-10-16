#%%

import numpy as np
from data_analytics import get_transformed_data, split
from utils import get_dataloader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from pathlib import Path
import pandas as pd
from metrics import *
from functools import partial
from matplotlib.cm import get_cmap

from configs.base_config import TrainingConfig


def save_fig(fig, path) :
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path)


config = TrainingConfig()
config.normalisation = '3-std'
config.data_file = '../../../DATA_DINOFusion/dino_1_4_degree_coarse_240125.tar'

train_dataloader = get_dataloader(config.data_file, batch_size=8,
                                                fields=config.fields, normalisation=config.normalisation, transform=True, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)


home='../../../test-generate-img'
model_path = 'ablation/infesteps_1000/constraints_border_zero_gradient_zero_mean'
extension = ".npy"
#model_path_list = [f for f in os.listdir(f'{home}/{model_path}') if os.path.isfile(os.path.join(f'{home}/{model_path}', f)) and f.endswith(extension)]
model_path_list = ['beta_1e-05_20250522-141925.npy','beta_0.0001_20250515-153542.npy', 'beta_0.0005_20250515-160357.npy', 
                   'beta_0.001_20250515-100900.npy','beta_0.003_20250515-102223.npy', 
                   'beta_0.004_20250515-102916.npy', 'beta_0.005_20250515-103610.npy', 
                   'beta_0.006_20250515-104231.npy','beta_0.01_20250515-110955.npy', 
                   'beta_0.05_20250515-114604.npy', 'beta_0.1_20250515-123850.npy']

#model_path_list = [ 'beta_0.001_20250515-100900.npy', 'beta_0.002_20250515-101513.npy', 
#                   'beta_0.003_20250515-102223.npy', 'beta_0.004_20250515-102916.npy', 
#                   'beta_0.005_20250515-103610.npy', 'beta_0.006_20250515-104231.npy',
#                   'beta_0.007_20250515-104922.npy', 'beta_0.008_20250515-105604.npy',
#                   'beta_0.009_20250515-110258.npy', 'beta_0.01_20250515-110955.npy', ]

generated_batch_list = []

for filename in model_path_list:
    generated_batch = torch.tensor(np.load(f'{home}/{model_path}/{filename}')) 
    generated_batch_list.append(generated_batch)

RE_CENTER_GENERATED = False
if RE_CENTER_GENERATED :
    generated_batch -= generated_batch.mean(axis=(-2, -1), keepdim=True)

# Un-normalisation : turn the batch to a dict

# Re-normalisation : bring back the data to it's initial scale
RENORMALISATION = True

# Without re-normalisation
generated_samples_list = []

if RENORMALISATION :
    extractor = train_dataloader.get_transform().uncall
    for generated_batch in generated_batch_list:
        generated_samples = get_transformed_data(generated_batch, function=extractor)
        generated_samples_list.append(generated_samples)
    samples = get_transformed_data(batch, function=extractor)
else :
    extractor = partial(split, transform=train_dataloader.get_transform())
    generated_samples = get_transformed_data(generated_batch, function=extractor)
    samples = get_transformed_data(batch, function=extractor)
    # Manual mask application
    for k in samples.keys() :
        generated_samples[k][samples[k].isnan()]  = torch.nan


betas = []
for filename in model_path_list:
    beta_value = filename.split('_')[1]
    betas.append(beta_value)

# Figure
# plot temperature and salinity evolution
cmap = get_cmap("tab10") 
fig, axs = plt.subplots(1,2, figsize=(15,5))
fig.suptitle('vertical profiles blue : data')
for i, key in enumerate(['toce.npy', 'soce.npy']) :
    axs[i].set_title(f'{key}')
    axs[i].set_xlabel('Depth')
    axs[i].plot(samples[key].nanmean(axis=(-2, -1)).T, label='data', c='blue')
    for idx, generated_samples in enumerate(generated_samples_list):
        color = cmap(idx % cmap.N)  
        axs[i].plot(generated_samples[key].nanmean(axis=(-2, -1)).T, label=f'gen beta {betas[idx]}', c=color)



#%%

SAVE_CLEAN = False
if SAVE_CLEAN :
    # Deal with bottom boundary
    for idx, generated_samples in enumerate(generated_samples_list):
        generated_samples['toce.npy'][:,:,1, :] = generated_samples['toce.npy'][:,:,2, :]
        generated_samples['soce.npy'][:,:,1, :] = generated_samples['soce.npy'][:,:,2, :]
        generated_samples['ssh.npy'][:,1, :] = generated_samples['ssh.npy'][:,2, :]

        beta_value = betas[idx]

        save_dir = (home +'/'+ f'ablation/exploration_plot/beta_{beta_value}/')
        Path(save_dir).mkdir(exist_ok=True)

        np.save(save_dir + 'toce.npy', generated_samples['toce.npy'].numpy())
        np.save(save_dir + 'soce.npy', generated_samples['soce.npy'].numpy())
        np.save(save_dir + 'ssh.npy', generated_samples['ssh.npy'][:, None].numpy())
        print(f'Saved clean to : {save_dir}')

# %%

#deal with border clean
bg = {}
for idx, generated_samples in enumerate(generated_samples_list):
    generated_samples['toce.npy'][:,:,1, :] = generated_samples['toce.npy'][:,:,2, :]
    generated_samples['soce.npy'][:,:,1, :] = generated_samples['soce.npy'][:,:,2, :]
    generated_samples['ssh.npy'][:,1, :] = generated_samples['ssh.npy'][:,2, :]

    bg[betas[idx]] = {'ssh': generated_samples['ssh.npy'].numpy(),
                'soce': generated_samples['soce.npy'].numpy(),
                'toce': generated_samples['toce.npy'].numpy()}
    
#format original data
train_dataloader = get_dataloader(config.data_file, batch_size=8, transform=False, shuffle=True) #batch_size=200
idt = iter(train_dataloader)
batch = next(idt)

for k in ['toce', 'soce', 'ssh'] :
    batch[k] = batch.pop(f'{k}.npy')

#%%

#metrics for physical consistency 

file_mask_LR=xr.open_dataset("./data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
file_mask_LR=file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

stats = []

metrics = {'BWbox' : partial(temperature_BWbox_metric, depth_box=1500),
           'DWbox' : partial(temperature_DWbox_metric, depth_box=1500)}

for gen in list(bg.keys()) + ['data']:
    batch_raw =  batch if gen == 'data' else bg[gen]
    for field in ['toce', 'soce'] :
        for bi, b in  enumerate(batch_raw[field]) :
            data = file_mask_LR.e3t_0.copy()
            data[:] = b
            base = {'source' : gen, 'index' : bi, 'field' : field}
            for key, f in metrics.items() :
                stats.append(base | {'metric' : key , 'value' : f(data, file_mask_LR).item()})
stats = pd.DataFrame(stats)
stats

def compute_density_error(batch, file_mask_LR) :
    """
        Take a batch with toce (b, z, h, w) and soce along with the maks
        and return the pourcent of volume unstable per element in the batch
    """
    bsize = batch['soce'].shape[0]

    data = file_mask_LR.e3t_0.copy().expand_dims({'batch' : bsize})
    soce = data.copy()
    soce[:] = batch['soce']
    toce = data.copy()
    toce[:] = batch['toce']

    tmask = data.copy()
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)

    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0).expand_dims({'batch' : bsize})

    density = get_density_at_surface(toce, soce, tmask)
    errors_density = (((density.diff('depth') < 0.0) * volume * tmask).sum(['nav_lat', 'nav_lon', 'depth']) / (volume * tmask).sum(['nav_lat', 'nav_lon', 'depth']))*100
    return errors_density.values


densities_errors = compute_density_error(batch, file_mask_LR)

densities_errors_stats={} 
densities_errors_stats['data']= densities_errors

for key in bg.keys():
    densities_errors_constraint = compute_density_error(bg[key], file_mask_LR)
    densities_errors_stats[key] = densities_errors_constraint

for key, v in densities_errors_stats.items() :
    df = pd.DataFrame({'index' : np.arange(densities_errors.shape[0]), 'value' : v})
    df['source'] = key
    df['metric'] ='density_errors'
    df['field'] = 'density'
    stats = pd.concat([stats, df])
stats.pivot_table(index='source', columns=['metric', 'field'], values='value', aggfunc='std')
stats.pivot_table(index='source', columns=['metric', 'field'], values='value')
sns.histplot(data=stats.query('(metric == "density_errors") & (field=="density") & (source!="no_constraint")'), x='value', hue='source')#, col='field')


stats.query('(metric == "density_errors") & (field=="density")')

stats.query('(metric == "density_errors") & (field=="density")').groupby('source')['value'].mean()

def create_pivot_with_stats(stats_df):
    mean_pivot = pd.pivot_table(stats_df,
                              index='source',
                              columns=['metric', 'field'],
                              values='value',
                              aggfunc='mean')

    std_pivot = pd.pivot_table(stats_df,
                             index='source',
                             columns=['metric', 'field'],
                             values='value',
                             aggfunc='std')

    return mean_pivot, std_pivot

def format_value_with_std(mean, std, precision=1):
    """Format value with compact scientific notation for small standard deviations"""
    mean_str = f"{mean:.{precision}f}"

    if std < 0.1:
        std_str = f"{std:.1e}"
        std_str = std_str.replace('e-0', 'e-')
    else:
        std_str = f"{std:.{precision}f}"

    return f"{mean_str} $\\pm$ {std_str}"

def create_latex_table(mean_df, std_df, betas):
    # Define row order
    row_order = ['data'] + betas
    
    row_names = {beta: f'beta={beta}' for beta in betas}
    row_names['data'] = 'Data'

    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l|cc|cc|c}",
        "\\hline",
        "& \\multicolumn{2}{c|}{Bottom-Water} & \\multicolumn{2}{c|}{Deep-Water} & Density \\\\",
        "Source & $\\mathcal{S}$ & $\\mathcal{T}$ & $\\mathcal{S}$ & $\\mathcal{T}$ & Errors \\\\",
        "\\hline"
    ]

    # Add data rows in specified order
    for idx in row_order:
        row_values = []
        for col in [('BWbox', 'soce'), ('BWbox', 'toce'),
                    ('DWbox', 'soce'), ('DWbox', 'toce'),
                    ('density_errors', 'density')]:
            mean_val = mean_df.loc[idx, col]
            std_val = std_df.loc[idx, col]
            row_values.append(format_value_with_std(mean_val, std_val))

        row = f"{row_names[idx]} & " + " & ".join(row_values) + " \\\\"
        latex.append(row)

    latex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Statistical analysis of water masses and density errors. Values are presented as mean $\\pm$ standard deviation.}",
        "\\label{tab:water_masses}",
        "\\end{table}"
    ])

    return "\n".join(latex)

# Usage:
mean_pivot, std_pivot = create_pivot_with_stats(stats)
latex_table = create_latex_table(mean_pivot, std_pivot, betas)
print(latex_table)




# %%
#compute variability of the states --> per point or as a mean variable (only over 10 first levels?)

#Variability metrics 
def compute_variability_metrics(data, z=10):
    point_var = {}
    point_var_mean = {}
    var_mean = {}

    for key in bg.keys():
        #point by point
        point_var[key] = {'ssh': np.var(data[key]['ssh'], axis=0),
                          'soce': np.var(data[key]['soce'], axis=0),
                          'toce': np.var(data[key]['toce'], axis=0)
                          }
        point_var_mean[key] = {'ssh': np.nanmean(point_var[key]['ssh']),
                               'soce': np.nanmean(point_var[key]['soce'][:,:,:]),
                               'toce': np.nanmean(point_var[key]['toce'][:,:,:])
                               }
        #variability to state mean
        var_mean[key] = {'ssh': np.var(np.nanmean(data[key]['ssh'], axis=(1,2)), axis=0),
                            'soce': np.var(np.nanmean(data[key]['soce'][:,0:10,:,:], axis=(1,2,3)), axis=0),
                            'toce': np.var(np.nanmean(data[key]['toce'][:,0:10,:,:], axis=(1,2,3)), axis=0)
                            }
        
    return point_var, point_var_mean, var_mean

point_var, point_var_mean, var_mean = compute_variability_metrics(bg, z=10)


#%%

# plot the variability depending on the beta average on one coordinate

def plot_variability(var_dict, field, z=0):
    fig, axs = plt.subplots(1, len(bg.keys()), figsize=(15, 10))

    for i, key in enumerate(bg.keys()):
        if field=='ssh':
            im = axs[i].imshow(var_dict[key][field][:,:], cmap='viridis')
        else:
            im = axs[i].imshow(var_dict[key][field][z,:,:], cmap='viridis')
        axs[i].set_title(f"B: {key}")
        axs[i].axis('off')
    

    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.1, pad=0.1)


plot_variability(point_var, 'soce')
plot_variability(point_var, 'toce')
plot_variability(point_var, 'ssh')



#%%
#Exploration plots as functions of beta's values

df_mean = pd.DataFrame.from_dict(point_var_mean, orient="index")
df_mean

df_var_mean = pd.DataFrame.from_dict(var_mean, orient="index")
df_var_mean

plt.plot(df_mean['toce'], label='toce')
plt.plot(df_mean['soce'], label='soce')
plt.plot(df_mean['ssh'], label='ssh')
plt.yscale('log')
plt.xlabel('Beta')
plt.ylabel('Mean point by point variance')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()

plt.plot(df_var_mean['toce'], label='toce')
plt.plot(df_var_mean['soce'], label='soce')
plt.plot(df_var_mean['ssh'], label='ssh')
plt.yscale('log')
plt.xlabel('Beta')
plt.ylabel('Overall mean variance')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()

#%%
# 3D exploration plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

betas_num = np.array([float(beta) for beta in betas])
density_errors = mean_pivot['density_errors']['density'].values[:-1]

sc = ax.scatter(df_mean['soce'].values, density_errors, betas_num,  c=betas_num, cmap='viridis', s=50)
ax.set_zlabel('Beta')
ax.set_xlabel('Mean point by point varibility')
ax.set_ylabel('Density errors')

cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Betas')



# %%
betas_num = np.array([float(beta) for beta in betas])
density_errors = mean_pivot['density_errors']['density'].values[:-1]
mean_variability = df_mean['soce'].values


plt.figure(figsize=(10, 6))
scatter = plt.scatter(density_errors, mean_variability, c=betas_num, cmap='viridis', s=50)
plt.plot(density_errors, mean_variability, linestyle='-', color='gray', alpha=0.7)

cbar = plt.colorbar(scatter, pad=0.1)
cbar.set_label('Beta')

plt.xlabel('Density Errors')
plt.ylabel('Mean Point by Point Variability')
plt.title('2D Scatter Plot: Variability vs. Density Errors (Colored by Beta)')
plt.grid(True, linestyle='--', alpha=0.6)

