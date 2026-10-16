from common import *

import cmocean.cm as cm

file_mask_LR = load_mask()

lon = file_mask_LR.glamt.values - 60
lat = file_mask_LR.gphit.values
tmask = file_mask_LR.tmask.values.astype(bool)
z_deep = int(np.argmin(np.abs(file_mask_LR.gdept_1d.values - 1050)))
levels_depth = {0: 'surface', z_deep: f'{file_mask_LR.gdept_1d.values[z_deep]:.0f} m'}

fields = {source: {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
          for source, x in sources.items()}

style = {'toce': dict(levels=np.linspace(0, 27, 28), cmap=cm.thermal, extend='both',
                      label='conservative temperature [°C]'),
         'soce': dict(levels=np.linspace(32.2, 36.8, 24), cmap=cm.haline, extend='both',
                      label=r'absolute salinity [g kg$^{-1}$]')}


def setup_ax(ax, row, col, title, row_label):
    """Gray land background, source as column title, labels on the outer panels only."""
    ax.set_facecolor('lightgray')
    if row == 0:
        ax.set_title(title, fontsize=9)
    if col == 0:
        ax.set_ylabel(f'{row_label}\nlatitude [°N]')
    else:
        ax.set_yticks([])
    if row == ax.figure.axes[0].get_gridspec().nrows - 1:
        ax.set_xlabel('longitude [°E]')
    else:
        ax.set_xticks([])


# Fig A: one sample of each source, T and S at the surface and at depth
rows = [(f, z) for f in ['toce', 'soce'] for z in levels_depth]
fig_state, axs = plt.subplots(len(rows), len(sources), figsize=(2.2 * len(sources), 4.5 * len(rows)),
                              constrained_layout=True, squeeze=False)
for i, (field, z) in enumerate(rows):
    s = style[field]
    for j, source in enumerate(sources):
        snapshot = np.where(tmask[z], fields[source][field][0, z], np.nan)
        cf = axs[i, j].contourf(lon, lat, snapshot, levels=s['levels'], cmap=s['cmap'], extend=s['extend'])
        setup_ax(axs[i, j], i, j, source, f'{field} at {levels_depth[z]}')
    fig_state.colorbar(cf, ax=axs[i, :], label=s['label'])
plt.show()


# Fig B: per-pixel variance over the batch of SST and SSS, shared colour scale per row
var_labels = {'toce': (cm.thermal, r'variance of SST [K$^{2}$]'),
              'soce': (cm.haline, r'variance of SSS [g$^{2}$ kg$^{-2}$]')}
variances = {field: {source: np.where(tmask[0], fields[source][field][:, 0].var(axis=0), np.nan)
                     for source in sources} for field in var_labels}

fig_var, axs = plt.subplots(len(var_labels), len(sources), figsize=(2.2 * len(sources), 4.5 * len(var_labels)),
                            constrained_layout=True, squeeze=False)
for i, (field, (cmap, label)) in enumerate(var_labels.items()):
    vmax = np.nanmax(variances[field]['data'])
    levels = np.linspace(0, vmax, 30)
    for j, source in enumerate(sources):
        cf = axs[i, j].contourf(lon, lat, variances[field][source], levels=levels, cmap=cmap, extend='max')
        setup_ax(axs[i, j], i, j, source, f'{field} variance')
    fig_var.colorbar(cf, ax=axs[i, :], label=label)
plt.show()
