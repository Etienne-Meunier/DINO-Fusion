#%%
# Zonally averaged potential density (referenced at the surface) of the generated fields
# computed with the EOS from temperature and salinity.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from load_data import load_generations, load_data_batch, load_mask, get_args, LABELS, UNCONSTRAINED
from metrics import get_density_at_surface
mpl.rcParams['image.origin'] = 'lower'

s = get_args('Zonally averaged density of the generated states')
figures_path = s['figures_path']
gens = s['gens'] or s['density']
bg = load_generations({k : s['path'][k] for k in gens})
batch = load_data_batch()
mask = load_mask()

save = True

#%%
# Compute density of the profil

def compute_density(batch, file_mask_LR) :
    """
        Take a batch with toce (b, z, h, w) and soce along with the maks
        and return the density of each element in the batch
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

all_data = [get_mean(batch['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1]]
all_data += [get_mean(bg[gen]['density'], mask, dim=['nav_lon']).mean(dim='batch')[:-1] for gen in gens]
titles = [f'{l}) {name}' for l, name in zip('abcdefghijklmnopqrstuvwxyz', ['data'] + [LABELS[gen] for gen in gens])]

# Colour range from the data and the constrained generations
scaled = [d for d, name in zip(all_data, ['data'] + gens) if name not in UNCONSTRAINED]
vmin = min((d - 1000).min().values for d in scaled)
vmax = max((d - 1000).max().values for d in scaled)

levels = np.linspace(vmin, vmax, 11)

ncols = 1 if len(all_data) <= 4 else 2
nrows = int(np.ceil(len(all_data) / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4 * nrows), sharex=True, squeeze=False)
axs = axs.T.flatten() if ncols == 1 else axs.flatten()

for i, ax in enumerate(axs):
    if i >= len(all_data) :
        ax.axis('off')
        continue
    cf = ax.contourf(lon_grid, depth_grid, all_data[i]-1000, levels=levels, cmap='YlGnBu_r', extendfrac='auto', extendrect=True)
    ax.contour(lon_grid, depth_grid, all_data[i]-1000, levels=levels, colors='k', linewidths=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('depth [m]', fontsize =18)
    ax.set_title(titles[i])
    if i >= len(all_data) - ncols :
        ax.set_xlabel('latitude', fontsize=18)

# Single colorbar for all
cbar = fig.colorbar(cf, ax=axs, orientation='vertical',
                    spacing='uniform')
cbar.set_label(r'potential density anomaly $\sigma_0$ [kg m$^{-3}$ − 1000]', fontsize=18)
cbar.ax.tick_params(labelsize=16)
plt.show()

if save:
    Path(figures_path).mkdir(exist_ok=True, parents=True)
    fig.savefig(f'{figures_path}/results_density_vs_depth.png', dpi=600, bbox_inches='tight')
