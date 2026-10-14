# %% [markdown]
# # Zonally averaged potential density of NEMO restarts
#
# - Generated states (initial restarts of the NEMO runs) : paper Figure 3
# - States after 10 years of NEMO integration : paper Figure 4 / Figure 7

# %%
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/a_posteriori/dinostics/')
from experiment import NemoExperiment

import matplotlib.pyplot as plt
import numpy as np
import cmocean as cm
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

figures_path = f'{PRP}/Results/a_posteriori/figures/'
Path(figures_path).mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load experiments

# %%
base_path = '/lustre/fsstor/projects/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/'
blandine_base = '/lustre/fsn1/projects/rech/omr/romr004/Blandine/DINO/Diffusion_runs/'
blandine_base_david = '/lustre/fsn1/projects/rech/omr/uym68qx/DINO/Blandine/'

# %%
expR4 = NemoExperiment(
    path= base_path,
    experiment_name='1_4degree',
    restarts=['restart25', 'restart26', 'restart27', 'restart28','restart29'],
    files = ['T_3D']
)

expR4.data['inst_T_3D'] = expR4.data['T_3D']
expR4.data['T_3D'] = expR4.data['inst_T_3D'].groupby('t.year').mean(dim='t')
expR4.data['T_3D'] = expR4.data['T_3D'].rename({'year': 't'})
expR4.data['T_3D'] = expR4.data['T_3D'].rename({'toce_inst': 'toce', 'soce_inst': 'soce'})

expR4.description = 'Simulation 1/4 degree'
expR4.namelist['namrun']['nn_itend'] = 12960000 # Here we didn't have the right end for some reason

expnoC = NemoExperiment(
    path= blandine_base,
    experiment_name='Constraint_no_C',
    restarts=['restart0'],
    files = ['inst_T_3D', 'T_3D']
)

expnoC.description = 'No constraint'

expCTS = NemoExperiment(
    path= blandine_base,
    experiment_name='Constraint_TS',
    restarts=['restart0'],
    files = ['inst_T_3D', 'T_3D']
)

expCTS.description = 'Constraint TS'

expC1 = NemoExperiment(
    path= blandine_base_david,
    experiment_name='',
    restarts=['restart25_C1'],
    files = ['inst_T_3D', 'T_3D']
)

expC1.namelist['namrun']['cn_ocerst_in'] = 'diffusion_states/' + expC1.namelist['namrun']['cn_ocerst_in']
expC1.description = 'Constraint C1'

expC2 = NemoExperiment(
    path= blandine_base_david,
    experiment_name='',
    restarts=['restart25_C2'],
    files = ['inst_T_3D', 'T_3D']
)

expC2.namelist['namrun']['cn_ocerst_in'] = 'diffusion_states/' + expC2.namelist['namrun']['cn_ocerst_in']
expC2.description = 'Constraint C2'


exps = [expnoC, expCTS, expC1, expC2]

# %% [markdown]
# # Plotting functions

# %%
def prepare_restart(exp, restart_path) :
    """
    Load restart file, convert coordinates and compute Sigma
    """
    rn = exp.open_restart(restart_path)

    # New coordinate z_c
    z_c = exp.data['inst_T_3D'].toce_inst.z_c
    rn = rn.drop_vars('nav_lev')
    rn = rn.rename({"nav_lev": "z_c", "y" : "y_c", "x" : "x_c"})
    rn = rn.assign_coords(z_c=z_c)
    return rn

def compute_sigma_mean(exp, rn) :
    # Compute Sigma
    s = exp.diagnostics.get_rho(T=rn.tn,
                                S=rn.sn,
                                z=2000.).where(exp.mask.tmask==1.0).mean('x_c')

    s = s.assign_coords(lat=exp.data['inst_T_3D'].gphit.isel(x_c=10),
                        depth_plot=('z_c',expnoC.mask.gdepw_1d.shift(z_f=-1, fill_value=4250.).values)) - 1000
    return s.isel(time_counter=0).compute()

def add_sigma_plot(ax, sigma_mean, title) :
    levels = np.linspace(22.,28.5,14)
    a = sigma_mean.plot.contourf(levels=levels, cmap=cm.cm.deep_r, x='lat',
                                 y='depth_plot', ax=ax, add_colorbar=False, yincrease=False)
    contour_1 = sigma_mean.plot.contour(levels=levels, colors='k', linewidths=0.75,
                                        x='lat', y='depth_plot', ax=ax, yincrease=False)

    # Vertical dashed lines
    ax.axvline(x=-65, color='gray', linestyle='dashed')
    ax.axvline(x=-45, color='gray', linestyle='dashed')
    ax.set_title('')
    ax.set_title(title)
    ax.set_ylabel('depth (m)')
    ax.set_ylim(3500,0)


    # Compute x-average between lat indices
    lat_vals = sigma_mean.lat.values
    idx_min = np.argmin(np.abs(lat_vals + 65))  # Find index for -65
    idx_max = np.argmin(np.abs(lat_vals + 45))  # Find index for -45
    subset = sigma_mean.isel(y_c=slice(idx_min, idx_max)).mean(dim='y_c')

    # Inset plot (on the right)
    inset_ax = inset_axes(ax, width="15%", height="100%", loc='right', borderpad=2)
    inset_ax.plot(subset, subset['depth_plot'], color='black', linewidth=2)
    inset_ax.patch.set_alpha(0.75)


    # **Ensure y-axis matches main plot**
    inset_ax.set_ylim(ax.get_ylim())
    inset_ax.set_yticks(ax.get_yticks())
    inset_ax.set_yticklabels([])  # Remove y labels for clarity
    inset_ax.set_xticklabels([])  # Remove x labels
    inset_ax.grid(True, linestyle="--", alpha=0.5)

    return a

# %%
plt.rcParams.update({'font.size': 22})

# %% [markdown]
# # Generated states (initial restarts)

# %%
initial_states = '/lustre/fsn1/projects/rech/omr/romr004/data/'

fig, axs = plt.subplots(1,len(exps), figsize=(35, 5), sharey=True, constrained_layout=True)

plots = []
for i, exp in enumerate(exps) :
    plots.append({'title' : f'{exp.description}', 'exp' : exp, 'path' : initial_states + exp.namelist['namrun']['cn_ocerst_in'] + '.nc', 'ax' : axs[i]})

for plot in plots:
    rn = prepare_restart(plot['exp'], plot['path'])
    sigma_mean = compute_sigma_mean(plot['exp'], rn)

    quadset = add_sigma_plot(plot['ax'], sigma_mean, plot['title'])

cbar = fig.colorbar(quadset, ax=axs, location='right', fraction=0.025, pad=0.02)
cbar.set_label(r'potential density anomaly')
fig.savefig(f'{figures_path}/generated_states.png')

# %% [markdown]
# # States after 10 years of integration

# %%
sexps = [expnoC, expC2]

fig, axs = plt.subplots(1,len(sexps) + 1, figsize=(30, 5), sharey=True, constrained_layout=True)

plots = []
plots.append({'exp' : expR4, 'path' :  expR4.path + expR4.restarts[-1] +'/DINO_12960000_restart.nc', 'title' : 'Target stratification', 'ax' : axs[0]})

for i, exp in enumerate(sexps) :
    plots.append({'title' : f'{exp.description}', 'exp' : exp, 'path' : exp.path + exp.restarts[-1] +'/DINO_11232000_restart.nc', 'ax' : axs[i+1]})

for plot in plots:
    rn = prepare_restart(plot['exp'], plot['path'])
    sigma_mean = compute_sigma_mean(plot['exp'], rn)

    add_sigma_plot(plot['ax'], sigma_mean, plot['title'])

cbar = fig.colorbar(quadset, ax=axs, location='right', fraction=0.025, pad=0.02)
cbar.set_label(r'potential density anomaly')
fig.savefig(f'{figures_path}/states_after_10y_integration.png')
