"""Level-mean T stats for runs/*/generated.npy (and pooled per generator type), vs data and vs the Feb-2025 200-state run.
z = offset / (std/sqrt(n)); zb = (mean - mean_200run) / bootstrap sd of n-state means from the 200-state run."""
PRP = '/Users/emeunier/Desktop/Projets/DINO-Fusion/'
import sys, os, glob
sys.path += [f'{PRP}/Diffusion_Model/', PRP]
import numpy as np, torch, xarray as xr
from configs.base_config import TrainingConfig
from DataLoader import get_infos
from TransformFields import TransformationPipeline
OUT = os.path.dirname(os.path.abspath(__file__))
config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='cpu')
mesh = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').isel(time_counter=0)
tmask = mesh.tmask.values.astype(bool); depth = mesh.gdept_1d.values
K = [int(np.abs(depth - d).argmin()) for d in (26, 74, 230, 2892)]
def lm(x):
    out = []
    with torch.no_grad():
        for g in torch.as_tensor(np.asarray(x)).float():
            v = transform.__uncall__(g)['toce'].numpy()
            out.append([v[k][tmask[k]].mean() if tmask[k].any() else np.nan for k in range(36)])
    return np.array(out)
d = np.load(f'{OUT}/data_level_means.npz')['toce']; dm, ds = d.mean(0), d.std(0)
L = np.load(f'{OUT}/run_stats_levels.npz'); t200 = L['infesteps_1000__constraints_no_constraints__20250203-141645.npy']
rng = np.random.default_rng(0)
def row(name, t):
    n = len(t); off = t.mean(0) - dm; z = off / (t.std(0, ddof=1) / np.sqrt(n))
    bm = np.array([t200[rng.choice(200, n, replace=False)].mean(0) for _ in range(5000)])
    zb = (t.mean(0) - bm.mean(0)) / bm.std(0)
    v = ~np.isnan(off)
    print(f'{name:28s} n={n:3d} rms_off={np.sqrt(np.nanmean(off**2)):.2f} spread={np.nanmean(t.std(0)[v]/ds[v]):6.1f} '
          f'off[26,74,230,2892]={np.round(off[K],2)} z={np.round(z[K],1)} zb200={np.round(zb[K],1)} rms_zb={np.sqrt(np.nanmean(zb**2)):.1f}', flush=True)
pools = {}
for p in sorted(glob.glob(f'{OUT}/runs/*/generated.npy')):
    name = p.split('/')[-2]; t = lm(np.load(p)); np.save(p.replace('generated.npy', 'level_T.npy'), t)
    row(name, t)
    key = 'cpu' if name.startswith('cpu') else ('mps+cpugen' if 'cpugen' in name else 'mps+mpsgen')
    pools.setdefault(key, []).append(t)
for k, v in pools.items(): row('POOL ' + k, np.concatenate(v))
row('REF feb2025 200 (first 16)', t200[:16])
