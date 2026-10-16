"""
Split the eval density error into a profile part and an anomaly part, without sampling. Each state is written as
level mean profile (horizontal mean of T and S over ocean cells, per level) + horizontal anomalies. E is recomputed
(float64, eval weighting) on hybrids : generated profile + DATA anomalies, and DATA mean profile + generated anomalies.
Also reports the per-sample spread of E. Output : `step2_swap.csv`.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from step1_bins import R, HERE, physical, split_E

RUNS = {'NONE' : 'none', 'NGD' : 'negative_grad_density_sqrt_eta0.3_eps0', 'ZM' : 'zero_mean_eta1_eps2.6e-3',
        'ISO' : 'dist_isoton_sqrt_eta5e-6_eps3.5e-5', 'ZM_ISO_1.33' : 'combo_iso5e-6_zm0.1_eps1.33',
        'X64J' : 'combo_iso5e-6_zm0.1_eps1.66e-1_joint0.01'}


def split(F) :
    """Level mean (b, k, 1, 1) over ocean cells and anomaly (b, k, i, j) of a physical field with NaN on land."""
    m = F.nanmean(dim=(2, 3), keepdim=True)
    return m, F - m


def E_of(T, S, volume) :
    return split_E(T, S, volume, torch.float64)[0]


config = TrainingConfig()
observer = Observer('data', config)
vol = observer.volume
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
x_data = states[np.random.default_rng(0).choice(len(states), 64, replace=False)]
del states
Td, Sd = physical(x_data, observer)
(mTd, aTd), (mSd, aSd) = split(Td.double()), split(Sd.double())
mTd_bar, mSd_bar = mTd.mean(0, keepdim=True), mSd.mean(0, keepdim=True)

rows = []
for name, run in RUNS.items() :
    T, S = physical(torch.tensor(np.load(f'{R}/{run}/generated.npy')), observer)
    (mT, aT), (mS, aS) = split(T.double()), split(S.double())
    b = len(T)
    E = E_of(T, S, vol)
    E_dataprof = E_of(mTd_bar + aT, mSd_bar + aS, vol)                 # generated anomalies on the mean data profile
    E_genprof = E_of(mT + aTd[:b], mS + aSd[:b], vol)                   # generated profile under data anomalies
    E_genprof_flat = E_of(mT.expand_as(T), mS.expand_as(S), vol)        # generated profile alone, no anomalies
    rows.append({'set' : name, 'E' : E.mean().item(), 'E_std' : E.std().item(), 'E_min' : E.min().item(), 'E_max' : E.max().item(),
                 'E_dataprof_genanom' : E_dataprof.mean().item(), 'E_genprof_dataanom' : E_genprof.mean().item(),
                 'E_genprof_only' : E_genprof_flat.mean().item()})
rows.append({'set' : 'DATA', 'E' : E_of(Td, Sd, vol).mean().item(), 'E_std' : E_of(Td, Sd, vol).std().item(),
             'E_dataprof_genanom' : E_of(mTd_bar + aTd, mSd_bar + aSd, vol).mean().item(),
             'E_genprof_only' : E_of(mTd.expand_as(Td), mSd.expand_as(Sd), vol).mean().item()})
df = pd.DataFrame(rows)
df.to_csv(HERE / 'step2_swap.csv', index=False)
pd.set_option('display.width', 250)
print(df.to_string(index=False, float_format=lambda v : f'{v:.3g}'))
