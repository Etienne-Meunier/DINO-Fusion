from common import *

fields = {'data': observer.density_gradient(batch).clamp(0.0).nansum(axis=(1))[0],
          'gen': observer.density_gradient(gen).clamp(0.0).nansum(axis=(1))[0]}

vmin = min(f.nanquantile(0.01).item() for f in fields.values())
vmax = max(f.nanquantile(0.99).item() for f in fields.values())

fig, axs = plt.subplots(1, 2)
for ax, (name, f) in zip(axs, fields.items()):
    im = ax.imshow(f, vmin=vmin, vmax=vmax)
    ax.set_title(name)
fig.colorbar(im, ax=axs, label='summed density gradient')

fig, axs = plt.subplots(2, 1)

axs[0].plot(observer.density_gradient(batch).clamp(0.0).nansum(axis=(-2, -1))[0])
axs[1].plot(observer.density_gradient(gen).clamp(0.0).nansum(axis=(-2, -1))[0])

grad_density = observer.density_gradient(batch)
(observer.volume_w[None] * grad_density.clamp(0.0)).nansum() / observer.volume_w.nansum()


grad_density_gen = observer.density_gradient(gen)
(observer.volume_w[None] * grad_density_gen.clamp(0.0)).nansum() / observer.volume_w.nansum()

losses['dist_isoton'](batch)

losses = {name : get_loss(name, observer) for name in losses}

density = observer.density(gen, result='strided')

density_iso = losses['dist_isoton'].isoton(density)

fig, axs = plt.subplots(1,3)
axs[0].plot(density[0,:, 50, 50])
axs[1].plot(density_iso[0,:, 50, 50])
axs[2].plot(observer.level_volume('strided'))

observer.level_volume('strided')
observer.volume.nanmean(axis=(-2,-1))
