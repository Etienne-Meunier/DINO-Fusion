"""
Seasonal consistency in the 35-60N band : surface temperature and upper-ocean stratification must tell the same season.

Reference : zonal-mean T of levels 0-13 of the 1800 training states (figures/seasonality_3d.npz), averaged into a
36-phase climatology. Two band features per state : band T0 (level 0) and the stratification T0 - T7 (level 7, 87 m).
Each member is matched to the nearest phase by each feature separately ; a seasonal state gets the same quarter from
both, a regional bias does not. Prints, per run, the mean +/- std of both features and the quarter counts.

    python band_test.py data centred_none optimal:zero_mean_border0
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
import numpy as np, torch
from common import observer, load_mask, generation_dirs

K, BAND, N_DATA = 14, (35, 60), 64
mask = load_mask()
lat = mask.gphit.isel(nav_lon=10).values
area = (mask.e1t * mask.e2t * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[:K]
T = np.load(HERE / 'figures' / 'seasonality_3d.npz')['T']
ph = np.arange(len(T)) % 36
clim = np.stack([T[ph == p].mean(0) for p in range(36)])
band = (lat > BAND[0]) & (lat < BAND[1])


def feats(Tz) :
    """Band T0 and T0 - T7 (n,) of zonal-mean T (n, K, y)."""
    b = Tz[:, :, band].mean(-1)
    return b[:, 0], b[:, 0] - b[:, 7]


def zonal(name) :
    """Zonal-mean T (n, K, y) of `data`, `optimal:<constraint>` or a runs/ folder."""
    if name == 'data' :
        return T[np.random.default_rng(0).choice(len(T), N_DATA, replace=False)]
    path = f'{generation_dirs()[name[8:]]}/generated.npy' if name.startswith('optimal:') else HERE / 'runs' / name / 'generated.npy'
    x = np.nan_to_num(observer.unnormalized(torch.as_tensor(np.load(path)))['toce'].numpy()[:, :K])
    return (x * area).sum(-1) / np.maximum(area.sum(-1), 1e-9)


f0, fs = feats(clim)
for name in sys.argv[1:] or ['data', 'centred_none'] :
    a, s = feats(zonal(name))
    pa = np.abs(a[:, None] - f0[None]).argmin(1)
    ps = np.abs(s[:, None] - fs[None]).argmin(1)
    q = lambda p : np.bincount(p, minlength=36).reshape(4, 9).sum(1)
    print(f'{name:24s} band T0 {a.mean():5.2f}+/-{a.std():4.2f}  T0-T7 {s.mean():5.2f}+/-{s.std():4.2f} | '
          f'quarters from T0 {q(pa)}, from T0-T7 {q(ps)}, same quarter {np.mean(pa // 9 == ps // 9):.2f}')
