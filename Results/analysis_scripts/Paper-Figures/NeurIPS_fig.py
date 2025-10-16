#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import xarray as xr
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import matplotlib.colors as mcolors

sys.path.append('../../../Diffusion_Model/')
sys.path.append('../')
from utils import get_dataloader
from metrics import get_density_at_surface
mpl.rcParams['image.origin'] = 'lower'
import pandas as pd

# 1. Data load
home = '../../../../'
model_path = f'{home}test-generate-img/'


path = {
    'constraint_C1' : f'{model_path}/ablation/ablation_chamon_gradient_zero_mean_density/eta_0.005_20250804-144748_clean',
    'constraint_C2' : f'{model_path}/ablation/again/beta_0_eta_0.1_20250804-185452_clean',
    'no_constraint' : f'{home}/tav0h83b/inference/infesteps_1000/constraints_no_constraints/20250131-110120_clean',
    'constraint_TS' : f'{model_path}inference/infesteps_1000/constraints_border_zero_gradient_zero_mean/beta_0_20250815-160022_clean'
    }

bg = {'constraint_C1' : [], 'constraint_C2' : [], 'no_constraint' : []}

for key, p in path.items() :
      bg[key] = {'ssh' : np.load(p + '/ssh.npy'),
                 'soce' : np.load(p + '/soce.npy'),
                 'toce' : np.load(p + '/toce.npy')}

data_file = '../../../../DATA_DINOFusion/dino_1_4_degree_coarse_240125.tar'
train_dataloader = get_dataloader(data_file, batch_size=8, transform=False, shuffle=True)
idt = iter(train_dataloader)
batch = next(idt)
for k in ['toce', 'soce', 'ssh'] :
    batch[k] = batch.pop(f'{k}.npy')

#enable to save results


save = False

#%%
# Fig 1. Visualisation of the fields / Quality of the generation

lat = np.linspace(-70, 70, 199)
lon = np.linspace(-60, 0, 62)
lon_grid, lat_grid = np.meshgrid(lon, lat)

idx1=0
idx2=17

# Data
temp_data = [batch['toce'][1][idx1]]
salt_data = [batch['soce'][1][idx1]]

for j in [idx1, idx2]:
    if j==idx2:
        temp_data = np.concat((temp_data,[batch['toce'][1][j]]))
        salt_data = np.concat((salt_data,[batch['soce'][1][j]]))
    for i in ['constraint_C1', 'constraint_C2']:
        temp_data = np.concat((temp_data,[bg[i]['toce'][3][j]]))
        salt_data = np.concat((salt_data,[bg[i]['soce'][3][j]]))

titles_temp = ['a) data at surface', 'b) C1 at surface', 'c) C2 at surface', 'd) data at 340 m', 'e) C1 at 340 m', 'f) C2 at 340 m']
titles_salt = ['g)', 'h)', 'i)', 'j)', 'k)', 'l)']
cmap_temp = cmocean.cm.thermal
cmap_salt = cmocean.cm.haline

levels_temp = np.linspace(0, 27, 28)
levels_salt = np.linspace(32.2, 36.8, 24)

# Projection setup
proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
fig, axs = plt.subplots(2, 6, figsize=(21, 10), sharex=True, sharey=True, dpi=600, subplot_kw={'projection': proj})  # high dpi for printing

# Global font size settings
plt.rcParams.update({
    'font.size': 16,         # default font size
    'axes.titlesize': 20,    # title font size
    'axes.labelsize': 20,    # axis label font size
    'xtick.labelsize': 18,   # x-axis tick label size
    'ytick.labelsize': 18,   # y-axis tick label size
    'legend.fontsize': 14
})

# Top row
cf_temp = axs[0,0].contourf(lon, lat, temp_data[0], levels=levels_temp, cmap=cmap_temp, extend='both')
#axs[0,0].set_ylabel('latitude [°N]')

for i, ax in enumerate(axs[0]):
    ax.contourf(lon, lat, temp_data[i], levels=levels_temp, cmap=cmap_temp, extend='both', transform=ccrs.PlateCarree())
    ax.set_title(titles_temp[i])
    ax.coastlines(alpha = 0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = False   # turn off top labels
    gl.right_labels = False # turn off right labels
    gl.bottom_labels = False 
    if i!=0:
        gl.left_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}


cbar_temp = fig.colorbar(cf_temp, ax=axs[0,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar_temp.set_label('conservative temperature [°C]')
cbar_temp.ax.tick_params(labelsize=16)  # colorbar ticks font size

# Bottom row
cf_salt = axs[1,0].contourf(lon, lat, salt_data[0], levels=levels_salt, cmap=cmap_salt, extend='both')
#axs[1,0].set_ylabel('latitude [°N]')

for i, ax in enumerate(axs[1]):
    ax.contourf(lon, lat, salt_data[i], levels=levels_salt, cmap=cmap_salt, extend='both', transform=ccrs.PlateCarree())
    ax.coastlines(alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-60, 0, -70, 70], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color='black', alpha=0.7, linestyle='--')
    gl.top_labels = False   # turn off top labels
    gl.right_labels = False # turn off right labels
    if i!=0:
        gl.left_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    #ax.set_xlabel('longitude [°E]')
    ax.set_title(titles_salt[i])
cbar_salt = fig.colorbar(cf_salt, ax=axs[1,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar_salt.set_label('absolute salinity [g/kg]')
cbar_salt.ax.tick_params(labelsize=16)


plt.show()

if save: 
    fig.savefig('results_generated_fields.png', dpi=600, bbox_inches='tight')  # high dpi and tight bounding box


# %%

# Fig 2. Density of the generated fields (can be used also for T and S)

# Compute density of the profil
file_mask_LR = xr.open_dataset("../data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
mask = file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})

def compute_density(batch, file_mask_LR) :
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

    density = get_density_at_surface(toce, soce, tmask)
    return density


density = compute_density(batch, mask)
batch['density'] = density

for i, constraint in enumerate(bg.keys()):
    density = compute_density(bg[constraint], mask)
    bg[constraint]['density'] = density

#mean proportional to volume
def get_mean(field, mask, dim=['nav_lat', 'nav_lon']): 
    e1t = mask.e1t.squeeze()
    e2t = mask.e2t.squeeze()
    e3t = mask.e3t_0.squeeze()
    tmask = mask.tmask.squeeze()
    volume = e1t * e2t * e3t * tmask
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    if field.shape[0]>1:
        volume = volume.expand_dims({'batch' : field.shape[0]})

    return (field*volume).sum(dim=dim)/volume.sum(dim=dim)

#Plots ------------------------
depth = mask.e3t_0.depth[:-1]
longitude = np.linspace(-70, 70, 199)
depth_grid, lon_grid = np.meshgrid(depth, longitude, indexing='ij')

data = get_mean(batch['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]
gen_C1 = get_mean(bg['constraint_C1']['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]
gen_C2 = get_mean(bg['constraint_C2']['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]
uncons = get_mean(bg['no_constraint']['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]

vmin = min(
    (data - 1000).min().values,
    (gen_C1 - 1000).min().values,
    (gen_C2 - 1000).min().values
)
vmax = max(
    (data - 1000).max().values,
    (gen_C1 - 1000).max().values,
    (gen_C2 - 1000).max().values
)

levels = np.linspace(vmin, vmax, 11)

fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
titles = ['a) data', 'b) generated with constraint C1', 'c) generated with constraint C2', 'd) generated without constraint']

all_data = [data, gen_C1, gen_C2, uncons]
for i, ax in enumerate(axs):
    # Plot 1
    cf = ax.contourf(lon_grid, depth_grid, all_data[i]-1000, levels=levels, cmap='YlGnBu_r', extendfrac='auto', extendrect=True)
    ax.contour(lon_grid, depth_grid, all_data[i]-1000, levels=levels, colors='k', linewidths=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('depth [m]', fontsize =18)
    ax.set_title(titles[i])
axs[3].set_xlabel('latitude', fontsize=18)

# Single colorbar for all
cbar = fig.colorbar(cf, ax=axs, orientation='vertical',
                    spacing='uniform')
cbar.set_label(r'potential density anomaly $\sigma_2$ [kg m$^{-3}$ − 1000]', fontsize=18)
cbar.ax.tick_params(labelsize=16)
plt.show()

if save:
    fig.savefig('results_density_vs_depth.png', dpi=600, bbox_inches='tight')

#%%
# Fig 3. Vertical stratification density

depth = mask.e3t_0.depth[:-1]
longitude = np.linspace(-70, 70, 199)
depth_grid, lon_grid = np.meshgrid(depth, longitude, indexing='ij')

data = get_mean(batch['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]
gen_C1 = get_mean(bg['constraint_C1']['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]
gen_C2 = get_mean(bg['constraint_C2']['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]

vmin = min(
    (data - 1000).min().values,
    (gen_C1 - 1000).min().values,
    (gen_C2 - 1000).min().values
)
vmax = max(
    (data - 1000).max().values,
    (gen_C1 - 1000).max().values,
    (gen_C2 - 1000).max().values
)

levels = np.linspace(vmin, vmax, 11)

fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)


# Plot 1
cf1 = axs[0].contourf(lon_grid, depth_grid, data-1000, levels=levels, cmap='YlGnBu_r', extendfrac='auto', extendrect=True)
axs[0].contour(lon_grid, depth_grid, data-1000, levels=levels, colors='k', linewidths=0.5)
axs[0].invert_yaxis()
axs[0].set_ylabel('depth [m]')
axs[0].set_title('a) data')

# Plot 2
cf2 = axs[1].contourf(lon_grid, depth_grid, gen_C1-1000, levels=levels, cmap='YlGnBu_r', extendfrac='auto', extendrect=True)
axs[1].contour(lon_grid, depth_grid, gen_C1-1000, levels=levels, colors='k', linewidths=0.5)
axs[1].invert_yaxis()
axs[1].set_ylabel('depth [m]')
axs[1].set_title('b) generated with constraint C1')

# Plot 3
cf3 = axs[2].contourf(lon_grid, depth_grid, gen_C2-1000, levels=levels, cmap='YlGnBu_r', extendfrac='auto', extendrect=True)
axs[2].contour(lon_grid, depth_grid, gen_C2-1000, levels=levels, colors='k', linewidths=0.5)
axs[2].invert_yaxis()
axs[2].set_ylabel('depth [m]')
axs[2].set_title('c) generated with constraint C2')
axs[2].set_xlabel('latitude')

# Single colorbar for all
cbar = fig.colorbar(cf1, ax=axs, orientation='vertical',
                    label=r'potential density anomaly $\sigma_2$ [kg m$^{-3}$ − 1000]',
                    spacing='uniform')
cbar.ax.tick_params(labelsize=16)
plt.show()

if save:
    fig.savefig('results_density_vs_depth.png', dpi=300, bbox_inches='tight')


#%%
# Fig 4. Variability 
import matplotlib.gridspec as gridspec
sns.set_theme(style='white')

lat = np.linspace(-70, 70, 199)
lon = np.linspace(-60, 0, 62)
k = 0  # depth index

gens = ['no_constraint', 'constraint_C2', 'constraint_TS']
fields = ['toce', 'soce']
titles = [
    ['a) without constraint', 'b) with constrained density', 'c) with constrained T,S'],
    ['d) without constraint', 'e) with constrained density', 'f) with constrained T,S']
]

fig = plt.figure(figsize=(20, 17), dpi=300)

plt.rcParams.update({
    'font.size': 24,         # default font size
    'axes.titlesize': 22,    # title font size
    'axes.labelsize': 22,    # axis label font size
    'xtick.labelsize': 24,   # x-axis tick label size
    'ytick.labelsize': 24,   # y-axis tick label size
    'legend.fontsize': 16
})

gs = gridspec.GridSpec(
    2, 4, 
    width_ratios=[1, 0.2, 1, 1],  
    wspace=0.2
)

axs = np.array([
    [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3])
    ],
    [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3])
    ]
])

for j, field in enumerate(fields):

    imgs = [np.nanvar(bg[gen][field], axis=0)[k] for gen in gens] 

    vmin_1, vmax_1 = np.nanquantile(np.stack(imgs), [0.01, 0.99])
    vmin_2, vmax_2 = np.nanquantile(np.stack(imgs[1:]), [0.01, 0.99])

    vmin = [vmin_1, vmin_2, vmin_2]
    vmax = [vmax_1, vmax_2, vmax_2]

    for i, (gen, img) in enumerate(zip(gens, imgs)):
        cmap = cmocean.cm.thermal if field == 'toce' else cmocean.cm.haline

        #vmin, vmax = np.nanquantile(img, [0.01, 0.99])

        im = axs[j, i].imshow(
            img,
            origin='lower',                                
            extent=(lon.min(), lon.max(), lat.min(), lat.max()),  
            aspect='auto',
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmax[i]
        )

        axs[j, i].set_title(titles[j][i], fontsize=24)
        axs[j, i].set_xlabel('longitude [°E]')
        

        if i!=2:
            axs[j, i].set_ylabel('latitude [°N]')
            yticks = np.linspace(lat.min(), lat.max(), 7) 
            axs[j, i].set_yticks(yticks)
            axs[j, i].set_yticklabels([f'{t:.0f}°' for t in yticks])

        xticks = np.linspace(lon.min(), lon.max(), 4)  
        axs[j, i].set_xticks(xticks)
        axs[j, i].set_xticklabels([f'{t:.0f}°' for t in xticks])


        if i==0 or i ==2:
            cbar = fig.colorbar(im, ax=axs[j, i], orientation='vertical', fraction=0.035, pad=0.02)
            if field == 'toce':
                cbar.set_label('variance of SST[C$^{2}$])', fontsize=24) 
                cbar.ax.tick_params(labelsize=20) 
            else: 
                cbar.set_label(' variance of SSS[g$^{2}$/ kg$^{-2}$])', fontsize=24)
                cbar.ax.tick_params(labelsize=20)

plt.show()

if save:
    fig.savefig('results_variability.png', dpi=300, bbox_inches='tight')


# %%
#quantitative variability metrics
def compute_variability_metrics(data, z=10):
    point_var = {}
    point_var_mean = {}
    var_mean = {}

    for key in data.keys():
        #point by point
        point_var[key] = {'ssh': np.var(data[key]['ssh'], axis=0),
                          'soce': np.var(data[key]['soce'], axis=0),
                          'toce': np.var(data[key]['toce'], axis=0)
                          }
        point_var_mean[key] = {'ssh': np.nanmean(point_var[key]['ssh']),
                               'soce': np.nanmean(point_var[key]['soce'][:,:,:]),
                               'toce': np.nanmean(point_var[key]['toce'][:,:,:])
                               }
        
    return point_var, point_var_mean

point_var, point_var_mean= compute_variability_metrics(bg)


df = pd.DataFrame(point_var_mean)
df = df.round(4)

latex_table = df.to_latex(
    index=True,           
    caption="Error metrics under different constraints.",
    label="tab:constraints",
    column_format="lcccc"  
)

print(latex_table)

