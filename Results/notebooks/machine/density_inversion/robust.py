"""
Robustness of the projected states before any guided sampling (review round after the gates).
    1. eps(t) refinement : q50 / q90 of each loss on x0_hat of 48 data states, 3 noise draws, on a t grid dense near 0.
    2. Projected states of the gate points (cold start, BorderZero), saved to `robust_states/` :
        near0 split : volume share with 0 <= g < 1.2e-4 (float32 ulp of rho) and 0 <= g < 1e-3, per depth band of the pair,
        share of E coming from inversions with |g| < 1.2e-4,
        re-noise test : noise to t in RENOISE (3 draws), one UNet x0_hat, E(x0_hat) ; a fragile pile-up gives a higher E
        than the other losses at comparable clean E.
Outputs : `robust_eps.csv`, `robust_near0.csv`, `robust_renoise.csv`.
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
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import BorderZero
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pd_explore import ocean_mask
from frontier import get_candidate, project
from transfer import MODEL, E_of
from transfer2 import x0_hat_of
from step1_bins import R, HERE, physical, split_E

EPS_TS = [0, 5, 10, 20, 30, 50, 100, 200, 300, 450, 600, 800, 999]
POINTS = [('C0', 10.0), ('C1', 1.0), ('C7_1e-4', 1e-3), ('C7_1e-4', 3e-3)]
RENOISE = [10, 20, 50]
BANDS = {'0-9' : slice(0, 10), '10-24' : slice(10, 25), '25-34' : slice(25, 35)}
ULP = 1.2e-4
DRAWS = 3


def near0_rows(x, observer, label) :
    """Volume shares (% of total ocean volume) of near-neutral pairs and the tiny-inversion part of E, per depth band."""
    T, S = physical(x, observer)
    E, _, _, _, g = split_E(T, S, observer.volume, torch.float64)
    w = observer.volume[1:].double()
    total = observer.volume.sum().double()
    tiny_inv = ((g < 0) & (g > -ULP) & torch.isfinite(g)) * w
    row = {'state' : label, 'E' : E.mean().item(), 'E_tiny' : (100 * tiny_inv.nansum(dim=(1, 2, 3)) / total).mean().item()}
    for name, sl in BANDS.items() :
        gb, wb = g[:, sl], w[sl]
        row[f'near0_ulp_{name}'] = (100 * (((gb >= 0) & (gb < ULP)) * wb).nansum(dim=(1, 2, 3)) / total).mean().item()
        row[f'near0_1e-3_{name}'] = (100 * (((gb >= 0) & (gb < 1e-3)) * wb).nansum(dim=(1, 2, 3)) / total).mean().item()
        row[f'E_{name}'] = (100 * (((gb < 0) & torch.isfinite(gb)) * wb).nansum(dim=(1, 2, 3)) / total).mean().item()
    return row


if __name__ == '__main__' :
    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    border = BorderZero(observer)
    pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL)
    gen = torch.Generator().manual_seed(0)
    losses = {n : get_candidate(n, observer) for n in sorted({p[0] for p in POINTS})}

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    xd = states[np.random.default_rng(4).choice(len(states), 48, replace=False)]
    del states

    rows = []
    for t in EPS_TS :
        with torch.no_grad() :
            x0s = [xd] if t == 0 else [torch.cat([x0_hat_of(pipeline, c, t, gen)[1].detach() for c in xd.split(8)])
                                       for _ in range(DRAWS)]
            for n, l in losses.items() :
                v = torch.cat([l(x0) for x0 in x0s])
                rows.append({'t' : t, 'loss' : n, 'q50' : v.quantile(0.5).item(), 'q90' : v.quantile(0.9).item(),
                             'E_x0hat' : torch.cat([E_of(x0, observer) for x0 in x0s]).mean().item()})
        print('eps', t, {r['loss'] : f"{r['q90']:.3g}" for r in rows[-len(losses):]}, flush=True)
        pd.DataFrame(rows).to_csv(HERE / 'robust_eps.csv', index=False)

    x0 = torch.tensor(np.load(f'{R}/none/generated.npy'))
    out = HERE / 'robust_states'
    out.mkdir(exist_ok=True)
    sets = {'none' : x0, 'data' : xd[:8]}
    for n, lam in POINTS :
        f = out / f'{n}_{lam:g}.npy'
        if f.exists() :
            sets[f'{n}_{lam:g}'] = torch.tensor(np.load(f))
        else :
            sets[f'{n}_{lam:g}'] = border(project(x0, losses[n], lam, mask, 300))
            np.save(f, sets[f'{n}_{lam:g}'].numpy())
        print('projected', n, lam, flush=True)

    near0 = pd.DataFrame([near0_rows(x, observer, k) for k, x in sets.items()])
    near0.to_csv(HERE / 'robust_near0.csv', index=False)
    rn = []
    for t in RENOISE :
        for d in range(DRAWS) :
            for k, x in sets.items() :
                with torch.no_grad() :
                    x0_hat = x0_hat_of(pipeline, x, t, gen)[1]
                rn.append({'state' : k, 't' : t, 'draw' : d, 'E_clean' : E_of(x, observer).mean().item(),
                           'E_renoised' : E_of(x0_hat, observer).mean().item(),
                           'dist' : ((x0_hat - x)[:, mask] ** 2).mean(dim=1).sqrt().mean().item()})
        print('renoise', t, flush=True)
    rn = pd.DataFrame(rn)
    rn.to_csv(HERE / 'robust_renoise.csv', index=False)

    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v : f'{v:.3g}'))
    print(near0.to_string(index=False, float_format=lambda v : f'{v:.3g}'))
    print(rn.groupby(['state', 't'])[['E_clean', 'E_renoised', 'dist']].mean().unstack('t').to_string(float_format=lambda v : f'{v:.3g}'))
