#%%
# Paper Figure 2 (right panel) : point by point variance of the generated surface temperature and salinity,
# and paper Table 2 : mean point by point variance.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmocean
import pandas as pd
from load_data import load_generations, get_args, LABELS

s = get_args('Variance of the generated surface fields')
figures_path = s['figures_path']
bg = load_generations(s['path'])

save = True

#%%
sns.set_theme(style='white')

lat = np.linspace(-70, 70, 199)
lon = np.linspace(-60, 0, 62)
k = 0  # depth index

# First generation has its own colour scale, the others share one
gens = s['gens'] or s['variability']
fields = ['toce', 'soce']
n = len(gens)
letters = 'abcdefghijklmnopqrstuvwxyz'
titles = [[f'{letters[j*n + i]}) {LABELS[gen]}' for i, gen in enumerate(gens)] for j in range(len(fields))]

fig = plt.figure(figsize=(20 * (n + 0.2) / 3.2, 17), dpi=300)

plt.rcParams.update({
    'font.size': 24,         # default font size
    'axes.titlesize': 22,    # title font size
    'axes.labelsize': 22,    # axis label font size
    'xtick.labelsize': 24,   # x-axis tick label size
    'ytick.labelsize': 24,   # y-axis tick label size
    'legend.fontsize': 16
})

gs = gridspec.GridSpec(
    2, n + 1,
    width_ratios=[1, 0.2] + [1] * (n - 1),
    wspace=0.2
)

axs = np.array([[fig.add_subplot(gs[j, 0])] + [fig.add_subplot(gs[j, i + 1]) for i in range(1, n)] for j in range(2)])

for j, field in enumerate(fields):

    imgs = [np.nanvar(bg[gen][field], axis=0)[k] for gen in gens]

    vmin_1, vmax_1 = np.nanquantile(np.stack(imgs), [0.01, 0.99])
    vmin_2, vmax_2 = np.nanquantile(np.stack(imgs[1:]), [0.01, 0.99])

    vmin = [vmin_1] + [vmin_2] * (n - 1)
    vmax = [vmax_1] + [vmax_2] * (n - 1)

    for i, (gen, img) in enumerate(zip(gens, imgs)):
        cmap = cmocean.cm.thermal if field == 'toce' else cmocean.cm.haline

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


        if i < 2:
            axs[j, i].set_ylabel('latitude [°N]')
            yticks = np.linspace(lat.min(), lat.max(), 7)
            axs[j, i].set_yticks(yticks)
            axs[j, i].set_yticklabels([f'{t:.0f}°' for t in yticks])

        xticks = np.linspace(lon.min(), lon.max(), 4)
        axs[j, i].set_xticks(xticks)
        axs[j, i].set_xticklabels([f'{t:.0f}°' for t in xticks])


        if i == 0 or i == n - 1:
            cbar = fig.colorbar(im, ax=axs[j, i], orientation='vertical', fraction=0.035, pad=0.02)
            if field == 'toce':
                cbar.set_label('variance of SST[C$^{2}$])', fontsize=24)
                cbar.ax.tick_params(labelsize=20)
            else:
                cbar.set_label(' variance of SSS[g$^{2}$/ kg$^{-2}$])', fontsize=24)
                cbar.ax.tick_params(labelsize=20)

plt.show()

if save:
    Path(figures_path).mkdir(exist_ok=True, parents=True)
    fig.savefig(f'{figures_path}/results_variability.png', dpi=300, bbox_inches='tight')


# %%
#quantitative variability metrics
def compute_variability_metrics(data, z=10):
    point_var = {}
    point_var_mean = {}

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
print(df.T.to_string())

latex_table = df.to_latex(
    index=True,
    caption="Error metrics under different constraints.",
    label="tab:constraints",
    column_format="lcccc"
)

print(latex_table)
