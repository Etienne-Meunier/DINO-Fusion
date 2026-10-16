"""
Variance over the batch split into the per-level spatial mean and the spatial anomalies (`pd_explore.variance_split`),
for random batches of training states and for every run of the sweep, to see which part of the variability each
constraint removes.

Output : `<fig>/variance_split.csv` one row per run (and 'data' rows), columns total / level_mean / anomaly
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
from pd_explore import ocean_mask, variance_split


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=str, default=f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/')
    parser.add_argument('--fig', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n_data_batches', type=int, default=50)
    args = parser.parse_args()

    config = TrainingConfig()
    mask = ocean_mask(Observer('data', config))
    rows = []

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    rng = np.random.default_rng(0)
    for k in range(args.n_data_batches) :
        rows.append({'name' : 'data', **variance_split(states[rng.choice(len(states), args.batch, replace=False)], mask)})
    del states

    for p in sorted(glob.glob(f'{args.out}/*/generated.npy')) :
        x = torch.tensor(np.load(p))
        if torch.isfinite(x).all() :
            rows.append({'name' : os.path.basename(os.path.dirname(p)), **variance_split(x, mask)})

    df = pd.DataFrame(rows)
    df['level_mean_share'] = df['level_mean'] / df['total']
    df.to_csv(f'{args.fig}/variance_split.csv', index=False)
    print(df.groupby('name')[['total', 'level_mean', 'anomaly', 'level_mean_share']].median().to_string())
