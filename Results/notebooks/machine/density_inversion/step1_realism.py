"""
Realism metrics of the Step-1 sets (pd_explore helpers) next to the eval density error, to compare the constraint runs at
equal cost : distance to the unconstrained run, level-mean / anomaly variance, roughness, strided pair-fraction of
inversions (the `inversions` column of FINDINGS_short). Output : `step1_realism.csv`.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pd_explore import ocean_mask, distance, variance_split, inversion_fraction, roughness
from step1_bins import R, RUNS, HERE

observer = Observer('data', TrainingConfig())
mask = ocean_mask(observer)
x_none = torch.tensor(np.load(f'{R}/none/generated.npy'))
E = pd.read_csv(HERE / 'step1_bins.csv').query('dtype == "f64"').groupby('set').E.mean()
rows = []
for name, run in RUNS.items() :
    x = torch.tensor(np.load(f'{R}/{run}/generated.npy'))
    with torch.no_grad() :
        rows.append({'set' : name, 'E_eval' : E[name], 'pair_inv' : inversion_fraction(x, observer).mean().item() * 100,
                     'pair_inv_0.01' : inversion_fraction(x, observer, tol=0.01).mean().item() * 100,
                     'dist_none' : distance(x, x_none, mask).mean().item(), 'rough' : roughness(x, mask)['all'],
                     **variance_split(x, mask)})
df = pd.DataFrame(rows)
df.to_csv(HERE / 'step1_realism.csv', index=False)
print(df.to_string(float_format=lambda v : f'{v:.4g}'))
