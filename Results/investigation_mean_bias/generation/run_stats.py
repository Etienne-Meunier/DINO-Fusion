"""Basin-mean T (and S) per level of every unconstrained generation vs a 200-state data batch.

Usage: python run_stats.py [extra.npy ...]  -> prints table, writes run_stats.csv and run_stats_levels.npz
z = (gen_mean - data_mean) / (gen_std / sqrt(n)), spread ratio = mean over levels of gen_std / data_std.
"""
PRP = '/Users/emeunier/Desktop/Projets/DINO-Fusion/'
import sys, os, glob, json
sys.path += [f'{PRP}/Diffusion_Model/', f'{PRP}/Results/analysis_scripts/', PRP]
import numpy as np, torch, xarray as xr
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline

OUT = os.path.dirname(os.path.abspath(__file__))
I = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference'
config = TrainingConfig()
torch.manual_seed(0)
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='cpu')
mesh = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').isel(time_counter=0)
tmask = mesh.tmask.values.astype(bool)
depth = mesh.gdept_1d.values
KEY = [int(np.abs(depth - d).argmin()) for d in (26, 74, 230, 2892)]


def level_means(x):
    """x normalised (b, 37, 208, 64) -> {toce, soce : (b, 36)} unweighted ocean mean per level."""
    out = {'toce': [], 'soce': []}
    with torch.no_grad():
        for g in torch.as_tensor(x).float():
            s = transform.__uncall__(g)
            for f in out:
                v = s[f].numpy()
                out[f].append([v[k][tmask[k]].mean() if tmask[k].any() else np.nan for k in range(36)])
    return {f: np.array(v) for f, v in out.items()}


cache = f'{OUT}/data_level_means.npz'
if os.path.exists(cache):
    data = dict(np.load(cache))
else:
    dl = get_dataloader(config.data_file, transform, batch_size=200, shuffle=True)
    data = level_means(next(iter(dl)))
    np.savez(cache, **data)

runs = sorted(set(
    [p for p in glob.glob(f'{I}/**/generated.npy', recursive=True) if '/none' in p or 'no_constraint' in p]
    + glob.glob(f'{I}/infesteps_1000/constraints_no_constraints/*.npy')
    + glob.glob(f'{I}/infesteps_1000/2025*.npy')) - {p for p in glob.glob(f'{I}/infesteps_1000/*contraint*')}) + sys.argv[1:]
runs = [r for r in runs if '/._' not in r]

rows, levels = [], {}
dm, ds = data['toce'].mean(0), data['toce'].std(0)
for r in runs:
    x = np.load(r, mmap_mode='r')
    lm = level_means(x)
    t = lm['toce']; n = len(t)
    off = t.mean(0) - dm
    z = off / (t.std(0, ddof=1) / np.sqrt(n))
    valid = ~np.isnan(off)
    args = {}
    for j in ('args.json', 'metrics.json'):
        p = os.path.join(os.path.dirname(r), j)
        if os.path.exists(p):
            a = json.load(open(p)); args = a.get('args', a)
    row = dict(run=r.replace(I + '/', ''), n=n, steps=args.get('inf_steps', args.get('steps', '?')),
               proj=args.get('projection', '?'), rms_off=np.sqrt(np.nanmean(off ** 2)),
               max_absz=np.nanmax(np.abs(z)), spread_ratio=np.nanmean(t.std(0)[valid] / ds[valid]),
               **{f'off{depth[k]:.0f}': off[k] for k in KEY}, **{f'z{depth[k]:.0f}': z[k] for k in KEY},
               S_rms_off=np.sqrt(np.nanmean((lm['soce'].mean(0) - data['soce'].mean(0)) ** 2)))
    rows.append(row); levels[row['run']] = t
    print(' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row.values()), flush=True)

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/run_stats.csv', index=False)
np.savez(f'{OUT}/run_stats_levels.npz', depth=depth, **{k.replace('/', '__'): v for k, v in levels.items()})
print('data T level means at key levels', [f'{depth[k]:.0f}m {dm[k]:.2f}+-{ds[k]:.2f}' for k in KEY])
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.round(2).to_string())
