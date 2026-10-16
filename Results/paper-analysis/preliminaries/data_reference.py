"""
Reference values of the sweep metrics (`pd_explore.diversity`, `pd_explore.distance`) on the training states, to read
the metrics of the generations : diversity of random batches of training states, and distance between two random
training states.

Output : `<out>/data_reference.json` {'diversity' : {group : [value per batch]}, 'distance' : [value per pair]}
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, json
import numpy as np
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pd_explore import ocean_mask, diversity, distance


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_keep', type=int, default=256, help='number of training states kept, drawn uniformly')
    parser.add_argument('--batch', type=int, default=8, help='batch size of the diversity')
    parser.add_argument('--n_batches', type=int, default=200)
    parser.add_argument('--out', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    args = parser.parse_args()

    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    dataloader = get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)
    states = torch.cat([b for b in dataloader])
    rng = np.random.default_rng(0)
    states = states[rng.choice(len(states), size=min(args.n_keep, len(states)), replace=False)]
    mask = ocean_mask(Observer('data', config))

    div = [diversity(states[rng.choice(len(states), args.batch, replace=False)], mask) for _ in range(args.n_batches)]
    pairs = rng.choice(len(states), size=(args.n_batches, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    dist = distance(states[pairs[:, 0]], states[pairs[:, 1]], mask).numpy().tolist()

    ref = {'diversity' : {k : [d[k] for d in div] for k in div[0]}, 'distance' : dist, 'n_states' : len(states)}
    json.dump(ref, open(f'{args.out}/data_reference.json', 'w'))
    print('diversity', {k : f'{np.median(v):.3f}' for k, v in ref['diversity'].items()}, 'distance', f'{np.median(dist):.3f}')
