#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 : select the median member of every generation of the sweep (`paper/common.py` RUN / PREFIX, 'optimal' by
default), and of the batch of training states of step 0 (DATA_RUN) if there is one, with the criterion of
`paper/fig1_density_profile.py`, and export it for the conversion to a NEMO restart.

Outputs in <generation folder>/aposteriori/ :
    {toce,soce,ssh}.npy  un-normalised median member, shapes (1, 36, 199, 62) and (1, 1, 199, 62)
    selection.json       member index, RMSE, source generation and model
A generation already converted or launched (conversion.json / launch.json) is skipped, unless --force.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/notebooks/paper/')

import argparse
import json
import os
import time
from glob import glob
import numpy as np
import torch
from common import RUN, PREFIX, INFERENCE, batch, gens, gens_args, generation_dirs, load_mask, median_member, observer, rmse, zonal_sections
from post_processing import save_clean
from settings import APOSTERIORI, CONVERSION, DATA_RUN, DATA_STATES, LAUNCH, MODEL_REL, SELECTION, timestamp, write_json

CRITERION = ('upper median of the unweighted RMSE of the volume-weighted zonal sigma_0 section of each member to the '
             'batch-mean section of a data batch (paper/fig1_density_profile.py)')

parser = argparse.ArgumentParser(description='Export the median member of every generation of the sweep')
parser.add_argument('--force', action='store_true', help='overwrite the selection of generations already converted or launched')
parser.add_argument('--constraints', nargs='*', default=None, help='constraints (and DATA_RUN) to select, default all')
args = parser.parse_args()

file_mask_LR = load_mask()
data_mean = np.nanmean(zonal_sections(batch, file_mask_LR), axis=0)
dirs = generation_dirs(PREFIX, INFERENCE)
runs, runs_args = dict(gens), dict(gens_args)
data_states = glob(f'{INFERENCE}/{PREFIX}/{DATA_RUN}/*/{DATA_STATES}')
if data_states:
    dirs[DATA_RUN] = max((os.path.dirname(p) for p in data_states), key=timestamp)
    runs[DATA_RUN] = torch.from_numpy(np.load(f'{dirs[DATA_RUN]}/{DATA_STATES}'))
    with open(f'{dirs[DATA_RUN]}/args.json') as f:
        runs_args[DATA_RUN] = json.load(f)

for constraint, x in runs.items():
    if args.constraints is not None and constraint not in args.constraints:
        continue
    out = f'{dirs[constraint]}/{APOSTERIORI}'
    done = [f for f in (CONVERSION, LAUNCH) if os.path.exists(f'{out}/{f}')]
    if done and not args.force:
        print(f'{constraint} : {", ".join(done)} in {out}, skip (--force to overwrite the selection)')
        continue
    errors = rmse(zonal_sections(x, file_mask_LR), data_mean)
    m = median_member(errors)
    print(f'{constraint} : median member {m} / {len(errors)}, RMSE {errors[m]:.3f} (mean {errors.mean():.3f})')

    save_clean(x[m:m + 1], out, observer.normalizer)
    write_json(f'{out}/{SELECTION}', dict(
        constraint=constraint, member=m, rmse=float(errors[m]), rmse_mean=float(errors.mean()), rmse_members=errors.tolist(),
        criterion=CRITERION, data_batch_size=len(batch), model=RUN, model_rel=MODEL_REL, prefix=PREFIX,
        generation_rel=os.path.relpath(dirs[constraint], os.environ['OCEANDATA']), generation_args=runs_args[constraint],
        created=time.strftime('%Y-%m-%d %H:%M:%S')))
