"""
Is the surface bias of the unconstrained generations a season the model over-represents ?

Data: level-0 T fields of the 1800 training states (figures/seasonality.npz, period 36 states = 1 year), averaged into a
36-phase climatology. Generations: the 200 unconstrained states of the README (no projection), the b16 optimal `none`
and `none_border0` runs, and runs/c7_c7100-like C7 runs. For every member: phase of the nearest climatology map
(area-weighted RMSE on the ocean), that RMSE, and the global level-0 T. Then zonal-mean level-0 T against latitude,
compared with the envelope of the 36 phases. Saves figures/season_match.png.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

mask = load_mask()
ocean = mask.tmask.values[0].astype(bool)
area = (mask.e1t * mask.e2t).values * ocean
lat = mask.gphit.isel(nav_lon=10).values
d = np.load(HERE / 'figures' / 'seasonality.npz')
T = np.nan_to_num(d["all_T0map"])
phase = np.arange(len(T)) % 36
clim = np.stack([T[phase == p].mean(0) for p in range(36)])
GEN200 = f'{INFERENCE}/infesteps_1000/constraints_no_constraints/20250203-141645_clean/toce.npy'


def level0_T(x) :
    """Level-0 T maps (b, y, x) of a normalised batch."""
    return observer.unnormalized(torch.as_tensor(x))['toce'][:, 0].detach().cpu().numpy()


def match(maps) :
    """Nearest phase, RMSE to it, global mean T0 of each (b, y, x) map."""
    err = np.sqrt((((maps[:, None] - clim[None]) ** 2) * area).sum((2, 3)) / area.sum())
    return err.argmin(1), err.min(1), (np.where(ocean, maps, 0) * area).sum((1, 2)) / area.sum()


def zonal(maps) :
    return (np.where(ocean, maps, 0) * area).sum(-1) / np.maximum(area.sum(-1), 1e-9)


sets = {'data (all 1800)' : T,
        'none, 200 (README)' : np.load(GEN200, mmap_mode='r')[:, 0].astype(np.float32),
        'none b16' : level0_T(gens['none']),
        'none_border0 b16' : level0_T(gens['none_border0']),
        'C7 eta 80 b16' : level0_T(gens['negative_grad_density_huber_border0_80'])}
print(f"climatology phases : global T0 {[round(float(v), 2) for v in match(clim)[2][::6]]} (every 6th)")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for name, maps in sets.items() :
    maps = np.nan_to_num(np.asarray(maps))
    ph, err, g = match(maps)
    counts = np.bincount(ph, minlength=36)
    print(f'{name:22s} n {len(maps):4d} T0 {g.mean():6.2f} +/- {g.std():5.2f} | RMSE to nearest phase {np.median(err):5.2f} '
          f'(median) | phase hist by quarter {[int(c) for c in counts.reshape(4, 9).sum(1)]}')
    if name.startswith('data') :
        continue
    axes[0].hist(ph, bins=np.arange(37) - 0.5, histtype='step', density=True, label=name)
    axes[1].hist(err, bins=30, histtype='step', density=True, label=name)
    axes[2].plot(lat, zonal(maps).mean(0), label=name)
z = zonal(clim)
axes[2].fill_between(lat, z.min(0), z.max(0), color='gray', alpha=0.4, label='data, 36 phases')
axes[0].set(xlabel='nearest phase (10-day index in the year)', title='a) Season of nearest climatology')
axes[1].set(xlabel='RMSE to nearest phase [C]', title='b) Distance to nearest season')
axes[2].set(xlabel='latitude', ylabel='zonal mean level-0 T [C]', title='c) Zonal mean surface T')
axes[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'season_match.png', dpi=150)
