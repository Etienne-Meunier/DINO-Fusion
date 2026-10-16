"""
Local metrics of the generations that no constraint acts on directly : fraction of vertical density inversions
(`pd_explore.inversion_fraction`), horizontal roughness (`pd_explore.roughness`) and the variance split (`pd_explore.variance_split`), for random batches of training
states and for every saved run of the sweep (including the post-hoc projections).

Output : `<fig>/local_metrics.csv` one row per run (name 'data' for the training batches)
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, glob, os
import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pd_explore import ocean_mask, variance_split, inversion_fraction, roughness


def metrics(x, observer, mask) :
    """Row of local metrics of a batch x (b, c, i, j)."""
    with torch.no_grad() :
        inv = inversion_fraction(x, observer)
        inv_tol = inversion_fraction(x, observer, tol=0.01)
        rough = roughness(x, mask)
    return {'inversions' : inv.mean().item(), 'inversions_0.01' : inv_tol.mean().item(), **variance_split(x, mask),
            **{f'rough_{k}' : v for k, v in rough.items()}}


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=str, nargs='*',
                        default=[f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/',
                                 f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/test_new_sampler_clip/'])
    parser.add_argument('--fig', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n_data_batches', type=int, default=20)
    args = parser.parse_args()

    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    rows = []

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    rng = np.random.default_rng(0)
    for k in range(args.n_data_batches) :
        rows.append({'folder' : 'data', 'name' : 'data', **metrics(states[rng.choice(len(states), args.batch, replace=False)], observer, mask)})
    del states

    for out in args.out :
        for p in sorted(glob.glob(f'{out}/*/generated.npy')) :
            x = torch.tensor(np.load(p))
            if torch.isfinite(x).all() :
                rows.append({'folder' : os.path.basename(os.path.normpath(out)), 'name' : os.path.basename(os.path.dirname(p)), **metrics(x, observer, mask)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{args.fig}/local_metrics.csv', index=False)
    print(df.groupby(['folder', 'name']).median().to_string())
