"""
Eval density error (float64, volume-weighted, file levels, as `compute_density_error`) split by inversion magnitude, and
realism metrics, for every saved run of `test_new_sampler_clip`, to compare the constraints at equal distance to the
unconstrained run. Output : `step1_all_runs.csv`.
"""
from pathlib import Path
import sys, glob, os
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pd_explore import ocean_mask, distance, variance_split, roughness
from step1_bins import R, HERE, BIN_NAMES, physical, split_E

observer = Observer('data', TrainingConfig())
mask = ocean_mask(observer)
x_none = torch.tensor(np.load(f'{R}/none/generated.npy'))
rows = []
for p in sorted(glob.glob(f'{R}/*/generated.npy')) :
    x = torch.tensor(np.load(p))
    if not torch.isfinite(x).all() or x.shape != x_none.shape :
        continue
    T, S = physical(x, observer)
    E, E_bins, _, _, _ = split_E(T, S, observer.volume, torch.float64)
    with torch.no_grad() :
        rows.append({'run' : os.path.basename(os.path.dirname(p)), 'E' : E.mean().item(),
                     **{f'E_{n}' : E_bins[:, i].mean().item() for i, n in enumerate(BIN_NAMES)},
                     'dist_none' : distance(x, x_none, mask).mean().item(), 'rough' : roughness(x, mask)['all'],
                     **variance_split(x, mask)})
df = pd.DataFrame(rows).sort_values('E')
df.to_csv(HERE / 'step1_all_runs.csv', index=False)
pd.set_option('display.width', 250)
print(df.to_string(index=False, float_format=lambda v : f'{v:.3g}'))
