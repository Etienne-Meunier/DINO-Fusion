"""
Area-weighted zonal-mean T and S of the levels 0 to K-1 of every training state, in archive order (period 36 states),
from the raw arrays of the training tar. Saves figures/seasonality_3d.npz : T, S (n, K, y) and keys.
"""
import io, sys, tarfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
import xarray as xr
from __init__ import PRP

K = 14
mask = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
area = (mask.e1t * mask.e2t * mask.tmask).transpose('nav_lev', 'y', 'nav_lon').values[:K]
out = {'toce' : {}, 'soce' : {}}
with tarfile.open('/Volumes/LoCe/oceandata/Dino-Fusion/dino_1_4_degree_coarse_240125.tar') as tar :
    for m in tar :
        if m.name.startswith('infos/') or not m.isfile() :
            continue
        key, field, _ = m.name.split('/')[-1].split('.')
        if field in out :
            a = np.nan_to_num(np.load(io.BytesIO(tar.extractfile(m).read()))[:K])
            out[field][key] = ((a * area).sum(-1) / np.maximum(area.sum(-1), 1e-9)).astype(np.float32)
keys = sorted(out['toce'])
np.savez(HERE / 'figures' / 'seasonality_3d.npz', keys=np.array(keys), T=np.stack([out['toce'][k] for k in keys]),
         S=np.stack([out['soce'][k] for k in keys]))
print(len(keys), out['toce'][keys[0]].shape)
