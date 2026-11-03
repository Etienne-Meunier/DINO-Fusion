"""
Time series of the level-0 volume-weighted T, S and sigma_0 of every training state, in archive order (assumed
chronological), from the raw (unnormalised) arrays of the training tar and of the `_summer` tar.
Saves figures/seasonality.npz with, per archive, arrays T0, S0, keys and the level-0 T and S fields (n, y, x).
"""
import io, sys, tarfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
import xarray as xr
from __init__ import PRP

mask = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
mask = mask.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})
v0 = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[0]
ROOT = '/Volumes/LoCe/oceandata/Dino-Fusion/'


def scan(name) :
    """Level-0 mean T and S of every state of the tar `name`, keyed by sample key, in archive order."""
    out = {'toce' : {}, 'soce' : {}, 'shape' : None, 'fields' : {'toce' : {}, 'soce' : {}}}
    with tarfile.open(ROOT + name) as tar :
        for m in tar :
            if m.name.startswith('infos/') or not m.isfile() :
                continue
            key, field, _ = m.name.split('/')[-1].split('.')
            if field not in ('toce', 'soce') :
                continue
            a = np.load(io.BytesIO(tar.extractfile(m).read()))
            out['shape'] = a.shape
            out['fields'][field][key] = a[0].astype(np.float32)
            out[field][key] = np.nansum(np.where(v0 > 0, a[0], 0) * v0) / v0.sum()
    keys = sorted(out['toce'])
    fields = {f : np.stack([out['fields'][f][k] for k in keys]) for f in ('toce', 'soce')}
    return keys, np.array([out['toce'][k] for k in keys]), np.array([out['soce'][k] for k in keys]), out['shape'], fields


res = {}
for name in ['dino_1_4_degree_coarse_240125.tar', 'dino_1_4_degree_coarse_240125_summer.tar'] :
    keys, T0, S0, shape, fields = scan(name)
    print(name, len(keys), shape, f'T0 {T0.mean():.3f} +/- {T0.std():.3f} [{T0.min():.2f}, {T0.max():.2f}]',
          f'S0 {S0.mean():.4f} +/- {S0.std():.4f}')
    tag = 'summer' if 'summer' in name else 'all'
    res |= {f'{tag}_keys' : np.array(keys), f'{tag}_T0' : T0, f'{tag}_S0' : S0,
            f'{tag}_T0map' : fields['toce'], f'{tag}_S0map' : fields['soce']}
(HERE / 'figures').mkdir(exist_ok=True)
np.savez(HERE / 'figures' / 'seasonality.npz', **res)
