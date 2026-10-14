#%%
# Paper Figure 2 (left panel) : temperature and salinity at the surface and at 340 m,
# training data vs generations with constraints C1 and C2.
from pathlib import Path
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from load_data import load_generations, load_data_batch, get_args, LABELS
mpl.rcParams['image.origin'] = 'lower'

s = get_args('Temperature and salinity at the surface and at 340 m')
figures_path = s['figures_path']
gens = s['gens'] or s['spatial']
bg = load_generations({k : s['path'][k] for k in gens})
batch = load_data_batch()

save = True

#%%
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
    for i in gens:
        temp_data = np.concat((temp_data,[bg[i]['toce'][3][j]]))
        salt_data = np.concat((salt_data,[bg[i]['soce'][3][j]]))

ncols = 2 * (len(gens) + 1)
letters = 'abcdefghijklmnopqrstuvwxyz'
titles_temp = [textwrap.fill(f'{letters[i]}) {name} at {depth}', 18) for i, (depth, name) in
               enumerate([(d, n) for d in ['surface', '340 m'] for n in ['data'] + [LABELS[g] for g in gens]])]
titles_salt = [f'{letters[ncols + i]})' for i in range(ncols)]
cmap_temp = cmocean.cm.thermal
cmap_salt = cmocean.cm.haline

levels_temp = np.linspace(0, 27, 28)
levels_salt = np.linspace(32.2, 36.8, 24)

# Projection setup
proj = ccrs.Orthographic(central_longitude=-30, central_latitude=0)
fig, axs = plt.subplots(2, ncols, figsize=(3.5 * ncols, 10), sharex=True, sharey=True, dpi=600, subplot_kw={'projection': proj})  # high dpi for printing

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

    ax.set_title(titles_salt[i])
cbar_salt = fig.colorbar(cf_salt, ax=axs[1,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar_salt.set_label('absolute salinity [g/kg]')
cbar_salt.ax.tick_params(labelsize=16)


plt.show()

if save:
    Path(figures_path).mkdir(exist_ok=True, parents=True)
    fig.savefig(f'{figures_path}/results_generated_fields.png', dpi=600, bbox_inches='tight')  # high dpi and tight bounding box
