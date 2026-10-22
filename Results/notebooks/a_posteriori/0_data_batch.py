#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 : form a batch of DATA_BATCH training states, the reference of the a posteriori runs. Restarting NEMO from T, S
and SSH only (velocities at rest) is itself a perturbation : running the same pipeline on data states measures it.

The states are the DATA_BATCH states that follow the data batch of `paper/common.py` (its `batch_size` first states of
the shuffle seeded with DATA_SEED) in the same shuffle, so they are disjoint from the batch the median criterion of
step 1 compares to. Outputs in <INFERENCE>/<PREFIX>/<DATA_RUN>/<ddmmyy-HHMMSS>/ :
    states.npy  normalised states (DATA_BATCH, channels, y, x), as the generated.npy of a generation (the other name keeps
                them out of the generations loaded by the paper, graphs and viewer scripts)
    args.json   source of the states
Step 1 then selects the median member of this batch as for a generation, steps 2 to 4 run on it unchanged.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/notebooks/paper/')

import argparse
import os
import time
from glob import glob

import numpy as np
import torch
from common import INFERENCE, PREFIX, RUN, batch, batch_size, config, transform
from DataLoader import DATA_SEED, get_dataloader
from settings import DATA_BATCH, DATA_RUN, DATA_STATES, write_json

parser = argparse.ArgumentParser(description='Form the batch of training states of the a posteriori reference runs')
parser.add_argument('--force', action='store_true', help='form a new batch even if one exists')
args = parser.parse_args()

existing = glob(f'{INFERENCE}/{PREFIX}/{DATA_RUN}/*/{DATA_STATES}')
if existing and not args.force:
    sys.exit(f'{existing} exists, --force to form a new batch')

states = next(iter(get_dataloader(config.data_file, transform, batch_size=batch_size + DATA_BATCH, shuffle=True, seed=DATA_SEED)))
if not torch.equal(states[:batch_size], batch):
    sys.exit('the shuffle does not reproduce the data batch of paper/common.py, the states would not be disjoint from it')
states = states[batch_size:].cpu().numpy()

out = f'{INFERENCE}/{PREFIX}/{DATA_RUN}/{time.strftime("%d%m%y-%H%M%S")}'
os.makedirs(out)
np.save(f'{out}/{DATA_STATES}', states)
print(f'Saved {out}/{DATA_STATES} {states.shape}')
write_json(f'{out}/args.json', dict(
    source='data', data_file=config.data_file, data_seed=DATA_SEED, batch=DATA_BATCH,
    states=f'{batch_size} to {batch_size + DATA_BATCH - 1} of the shuffle (buffer 1000) seeded with data_seed, after the '
           f'{batch_size} states of the data batch of paper/common.py', model=RUN, loss=None, projection=None))
