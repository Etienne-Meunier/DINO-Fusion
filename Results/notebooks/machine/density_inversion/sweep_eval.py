"""
Metrics of the negative_grad_density_huber (C7) eta sweep, each run paired with the unconstrained run of the same seed
and batch (same noise), as required by the inspector review :
    E (eval, float64) with its |g| bins, E_tiny = share of E with |g| < 1.2e-4, deep near-neutral volume
    (0 <= g < 1e-3, pairs of levels 25-34), level-mean / anomaly variance, roughness, distance to the paired none
    (RMS over ocean cells), paired per-sample E difference (mean, std), final lbd and int L dt ~ (lbd_end - lbd0) / eta.
References : the paired none, optimal mean_density / zero_mean against none b16, 16 data states.
Output : `sweep_eval.csv`.
"""
from pathlib import Path
import sys, json
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pd_explore import ocean_mask, variance_split, roughness
from step1_bins import HERE, BIN_NAMES, physical, split_E

INF = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b/inference'
NONE = {8 : f'{INF}/sweep_baselines/none_border0_b8/151026-145216', 16 : f'{INF}/optimal/none_border0/151026-150424'}
REFS = {'mean_density' : f'{INF}/optimal/mean_density_border0/171026-012335',
        'zero_mean' : f'{INF}/optimal/zero_mean_border0/171026-014033'}
SWEEPS = [f'{INF}/sweep_density_huber/negative_grad_density_huber_border0', f'{INF}/optimal/negative_grad_density_huber_border0',
          f'{INF}/optimal/negative_grad_density_huber_border0_30', f'{INF}/optimal/negative_grad_density_huber_border0_80',
          f'{INF}/sweep_density_huber_b16/negative_grad_density_huber_border0']


def metrics(x, x_none, observer, mask) :
    T, S = physical(x, observer)
    E, E_bins, _, _, g = split_E(T, S, observer.volume, torch.float64)
    w = observer.volume[1:].double()
    total = observer.volume.sum().double()
    deep = g[:, 25:35]
    near0_deep = 100 * (((deep >= 0) & (deep < 1e-3)) * w[25:35]).nansum(dim=(1, 2, 3)) / total
    row = {'E' : E.mean().item(), 'E_std' : E.std().item(),
           **{f'E_{n}' : E_bins[:, i].mean().item() for i, n in enumerate(BIN_NAMES)},
           'E_tiny_share' : (E_bins[:, 0].sum() / E.sum().clamp(min=1e-30)).item(),
           'near0_deep' : near0_deep.mean().item(), 'rough' : roughness(x, mask)['all'], **variance_split(x, mask)}
    if x_none is not None :
        T0, S0 = physical(x_none, observer)
        E0 = split_E(T0, S0, observer.volume, torch.float64)[0]
        row.update({'dist' : ((x - x_none)[:, mask] ** 2).mean(dim=1).sqrt().mean().item(),
                    'dE_paired' : (E - E0).mean().item(), 'dE_paired_std' : (E - E0).std().item()})
    return row


def load(run) :
    return torch.tensor(np.load(f'{run}/generated.npy'))


if __name__ == '__main__' :
    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    nones = {b : load(r) for b, r in NONE.items()}

    rows = [{'run' : f'none_b{b}', 'eta' : 0.0, 'batch' : b, **metrics(x, None, observer, mask)} for b, x in nones.items()]
    for name, r in REFS.items() :
        x = load(r)
        rows.append({'run' : name, 'eta' : json.load(open(f'{r}/args.json'))['eta'], 'batch' : len(x),
                     **metrics(x, nones[len(x)], observer, mask)})
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    xd = states[np.random.default_rng(3).choice(len(states), 16, replace=False)]
    del states
    rows.append({'run' : 'data', 'eta' : 0.0, 'batch' : 16, **metrics(xd, None, observer, mask)})

    for r in sorted(r for s in SWEEPS for r in Path(s).glob('*/')) :
        if not (r / 'generated.npy').exists() :
            continue
        a = json.load(open(r / 'args.json'))
        x = load(r)
        lbd = np.load(r / 'lbd.npy')
        rows.append({'run' : f'C7_{r.name}', 'eta' : a['eta'], 'batch' : len(x), **metrics(x, nones[len(x)], observer, mask),
                     'lbd_end' : float(np.median(lbd)), 'int_L_dt' : float(np.median((lbd - a['lbd0']) / a['eta']))})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'sweep_eval.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['run', 'eta', 'batch', 'E', 'E_std', 'dE_paired', 'dE_paired_std', 'E_tiny_share', 'near0_deep', 'dist',
            'level_mean', 'anomaly', 'rough', 'lbd_end', 'int_L_dt']
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda v : f'{v:.4g}'))
