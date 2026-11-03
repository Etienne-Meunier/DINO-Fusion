"""
Sampler ablation, unconstrained, b8 seed 0 : the final batches of traj_test.py (runs/traj_<tag>_x.npy), land and padding
set to 0 before evaluation (BorderZero after the last step for every variant).

Columns : density error E, level-0 T (volume-weighted) mean and member std, level-0 sigma_0 bias to the data batch,
BW and DW box T. Prints the table and saves figures/sampler_ablation.txt.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *

from functools import partial
from metrics import compute_density_error, temperature_BWbox_metric, temperature_DWbox_metric
from pipelines.losses import BorderZero

VARIANTS = {'raw' : 'land from the model, raw noise (= optimal/none)',
            'raw_call' : 'land from the model, centred noise',
            'border0' : 'BorderZero every step, raw noise (= optimal/none_border0)',
            'border0_call' : 'BorderZero every step, centred noise',
            'bnoise' : 'noisy land, raw noise',
            'bnoise_call' : 'noisy land, centred noise (centred sampler)'}
mask = load_mask()
ocean0 = mask.tmask.values[0].astype(bool)
volume0 = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[0]
project = BorderZero(observer)
boxes = {'BW T' : partial(temperature_BWbox_metric, depth_box=1500), 'DW T' : partial(temperature_DWbox_metric, depth_box=1500)}


def scores(x) :
    """Per-member E, T0, sigma_0 level 0, BW T, DW T of a normalised batch."""
    f = {k : v.detach().cpu().numpy() for k, v in observer.unnormalized(project(torch.as_tensor(x))).items()}
    sig = compute_density(f, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values[:, 0] - 1000
    mean0 = lambda v : np.nansum(np.where(ocean0, v, 0) * volume0, axis=(1, 2)) / volume0.sum()
    out = {'E' : np.asarray(compute_density_error(f, mask), dtype=float), 'T0' : mean0(f['toce'][:, 0]), 'sig0' : mean0(sig)}
    for k, m in boxes.items() :
        vals = []
        for b in f['toce'] :
            sample = mask.e3t_0.copy()
            sample[:] = np.asarray(b)
            vals.append(m(sample, mask).item())
        out[k] = np.asarray(vals)
    return out


ref = scores(batch)
rows = {'data' : ref}
for tag in VARIANTS :
    f = HERE / 'runs' / f'traj_{tag}_x.npy'
    if f.exists() :
        rows[tag] = scores(np.load(f))
table = pd.DataFrame({tag : {'sampler' : VARIANTS.get(tag, 'data batch'),
                             'E' : f"{r['E'].mean():.1f} ± {r['E'].std(ddof=1):.1f}",
                             'T0' : f"{r['T0'].mean():.2f} ± {r['T0'].std(ddof=1):.2f}",
                             'bias0' : f"{r['sig0'].mean() - ref['sig0'].mean():+.2f}",
                             'BW T' : f"{r['BW T'].mean():.2f} ± {r['BW T'].std(ddof=1):.2f}",
                             'DW T' : f"{r['DW T'].mean():.2f} ± {r['DW T'].std(ddof=1):.2f}"} for tag, r in rows.items()}).T
pd.set_option('display.width', 250)
print(table.to_string())
(HERE / 'figures' / 'sampler_ablation.txt').write_text(table.to_string() + '\n')
