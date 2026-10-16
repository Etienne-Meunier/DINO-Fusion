from common import *

sns.histplot(batch[:, 0].nanmean((-2, -1)))
sns.histplot(gen[:, 0].nanmean((-2, -1)))

surface_data_T = observer.unnormalized(batch)['toce'][:,0].nanmean(dim=(-2, -1))
surface_gen_T = observer.unnormalized(gen)['toce'][:,0].nanmean(dim=(-2, -1))

sns.histplot(surface_data_T)
sns.histplot(surface_gen_T)


quantiles = np.linspace(0, 1, 5)

surface_data = observer.unnormalized(batch)['toce'][:, 0]
surface_gen = observer.unnormalized(gen)['toce'][:, 0]

rows = {'data': (surface_data, surface_data_T), 'gen': (surface_gen, surface_gen_T)}
selection = {}
for name, (fields, means) in rows.items():
    order = torch.argsort(means)
    idx = (quantiles * (len(order) - 1)).round().astype(int)
    selection[name] = (fields[order[idx]], means[order[idx]])

vmin = min(s[0].nanquantile(0.01).item() for s in selection.values())
vmax = max(s[0].nanquantile(0.99).item() for s in selection.values())

fig, axs = plt.subplots(2, len(quantiles), figsize=(4 * len(quantiles), 8), squeeze=False)
for i, (name, (fields, means)) in enumerate(selection.items()):
    for j in range(len(quantiles)):
        ax = axs[i, j]
        im = ax.pcolormesh(fields[j].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f'{name} q={quantiles[j]:.2f} - mean {means[j].item():.2f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, label='surface toce')
