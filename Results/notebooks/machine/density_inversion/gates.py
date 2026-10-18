"""
Gates of PLAN2.md for the candidates at the lam that brings the unconstrained states into the acceptable band
(E between 2.4 and 9.3 %), from a cold start (check of the warm start of `frontier.py`) :
    on the 8 unconstrained states x0 : E and |g| bins, distance, roughness, level-mean / anomaly variance,
        push on already stable pairs : median and mean of g_new - g_old over pairs with g_old >= 0,
        pile-up near 0 : volume share of pairs with 0 <= g_new < 1e-3 vs x0 and data,
    on 16 data states with the same lam : distance moved and change of E.
Output : `gates.csv`.
"""
from pathlib import Path
import sys, argparse
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
from pd_explore import ocean_mask, variance_split, roughness
from frontier import get_candidate, project
from step1_bins import R, HERE, BIN_NAMES, physical, split_E

POINTS = {'C0' : [10.0], 'C1' : [1.0], 'C1+C5' : [1.0], 'C7_1e-4' : [1e-3, 3e-3], 'C6_1e-4' : [1e-2]}


def g64(x, observer) :
    """g = rho[k+1] - rho[k] in float64 on the file levels, NaN on dry pairs."""
    T, S = physical(x, observer)
    return split_E(T, S, observer.volume, torch.float64)


def stats(x, x0, observer, mask) :
    E, E_bins, _, _, g = g64(x, observer)
    _, _, _, _, g0 = g64(x0, observer)
    w = observer.volume[1:].double()
    stable = (g0 >= 0) & torch.isfinite(g0)
    shift = (g - g0)[stable]
    near0 = (((g >= 0) & (g < 1e-3)) * w).nansum(dim=(1, 2, 3)) / w.sum() * 100
    return {'E' : E.mean().item(), **{f'E_{n}' : E_bins[:, i].mean().item() for i, n in enumerate(BIN_NAMES)},
            'dist' : ((x - x0)[:, mask] ** 2).mean(dim=1).sqrt().mean().item(),
            'stable_shift_median' : shift.median().item(), 'stable_shift_mean' : shift.mean().item(),
            'near0' : near0.mean().item(), 'rough' : roughness(x, mask)['all'], **variance_split(x, mask)}


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--only', nargs='*', default=None, help='subset of POINTS keys')
    args = parser.parse_args()

    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    border = BorderZero(observer)
    x0 = torch.tensor(np.load(f'{R}/none/generated.npy'))
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    xd = states[np.random.default_rng(3).choice(len(states), 16, replace=False)]
    del states

    rows = [{'loss' : 'none', 'lam' : 0.0, 'set' : 'NONE', **stats(x0, x0, observer, mask)},
            {'loss' : 'none', 'lam' : 0.0, 'set' : 'DATA', **stats(xd, xd, observer, mask)}]
    out = HERE / ('gates.csv' if args.only is None else f'gates_{"_".join(args.only)}.csv'.replace('+', 'p'))
    for name, lams in POINTS.items() :
        if args.only and name not in args.only :
            continue
        loss = get_candidate(name, observer)
        for lam in lams :
            x = border(project(x0, loss, lam, mask, args.iters))
            rows.append({'loss' : name, 'lam' : lam, 'set' : 'NONE', **stats(x, x0, observer, mask)})
            y = border(project(xd, loss, lam, mask, args.iters))
            rows.append({'loss' : name, 'lam' : lam, 'set' : 'DATA', **stats(y, xd, observer, mask)})
            print(name, lam, {k : round(v, 5) for k, v in rows[-2].items() if k in ('E', 'dist', 'stable_shift_mean', 'near0')},
                  'data', {k : round(v, 5) for k, v in rows[-1].items() if k in ('E', 'dist')}, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
