"""
Functional seasonal test : does a sampler produce winter and summer states, each consistent with itself ?

Reference : the 1800 training level-0 T maps (figures/seasonality.npz, period 36 states) averaged into a 36-phase
climatology. For each member of a run, the level-0 T pattern with its area-weighted global mean removed is matched to
the phase climatology patterns (also without their mean) :
    rmse     : area-weighted RMSE of the pattern to the nearest phase [C] (data members : about 0.35 / see the data row)
    phase    : the nearest phase, its coverage given as the number of distinct phases and the counts per quarter
    T0       : the member's global level-0 T, against T0(phase) the global T of the matched phase climatology : slope and
               offset of the least-squares line T0 = a T0(phase) + b, mean offset T0 - T0(phase), correlation
Runs are given as names of runs/ or keys of SOURCES ; `data` draws N_DATA training states.
Prints the table and saves figures/season_test.png (T0 against T0(phase), one panel per run).

    python season_test.py data centred_none centred_c7010 optimal:zero_mean_border0
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
import numpy as np, torch, matplotlib.pyplot as plt
from common import observer, load_mask, INFERENCE, generation_dirs

N_DATA = 64
mask = load_mask()
ocean = mask.tmask.values[0].astype(bool)
area = (mask.e1t * mask.e2t).values * ocean
d = np.load(HERE / 'figures' / 'seasonality.npz')
T = np.nan_to_num(d['all_T0map'])
phase = np.arange(len(T)) % 36
gmean = lambda m : (m * area).sum((-2, -1)) / area.sum()
clim = np.stack([T[phase == p].mean(0) for p in range(36)])
clim_T0 = gmean(clim)
clim_pat = (clim - clim_T0[:, None, None]) * ocean


def level0(name) :
    """Level-0 T maps (b, y, x) of a run : `data`, `optimal:<constraint>` (paper b16 runs) or a runs/ folder."""
    if name == 'data' :
        idx = np.random.default_rng(0).choice(len(T), N_DATA, replace=False)
        return T[idx]
    if name.startswith('optimal:') :
        x = np.load(Path(generation_dirs()[name.split(':')[1]]) / 'generated.npy')
    else :
        x = np.load(HERE / 'runs' / name / 'generated.npy')
    return np.nan_to_num(observer.unnormalized(torch.as_tensor(x))['toce'][:, 0].numpy())


def test(maps) :
    """Per-member nearest phase, pattern RMSE to it, member T0 and matched-phase T0."""
    T0 = gmean(maps)
    pat = (maps - T0[:, None, None]) * ocean
    err = np.sqrt((((pat[:, None] - clim_pat[None]) ** 2) * area).sum((-2, -1)) / area.sum())
    p = err.argmin(1)
    return p, err.min(1), T0, clim_T0[p]


names = sys.argv[1:] or ['data', 'centred_none']
fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), squeeze=False, sharex=True)
print(f'{"run":28s} {"n":>3s} {"rmse":>5s} {"phases":>6s} {"quarters":>16s} {"T0":>13s} {"slope":>6s} {"offset":>7s} {"corr":>5s}')
for ax, name in zip(axes[0], names) :
    p, err, T0, T0p = test(level0(name))
    a, b = np.polyfit(T0p, T0, 1) if np.ptp(T0p) > 0 else (np.nan, np.nan)
    corr = np.corrcoef(T0p, T0)[0, 1] if np.ptp(T0p) > 0 else np.nan
    quarters = np.bincount(p, minlength=36).reshape(4, 9).sum(1)
    print(f'{name:28s} {len(T0):3d} {np.median(err):5.2f} {len(np.unique(p)):6d} {str(quarters.tolist()):>16s} '
          f'{T0.mean():6.2f}+/-{T0.std():4.2f} {a:6.2f} {(T0 - T0p).mean():+7.2f} {corr:5.2f}')
    ax.scatter(T0p, T0, s=10)
    lim = [clim_T0.min() - 0.05, clim_T0.max() + 0.05]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel('T0 of matched phase [C]', fontsize=8)
axes[0, 0].set_ylabel('member T0 [C]', fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'season_test.png', dpi=150)
