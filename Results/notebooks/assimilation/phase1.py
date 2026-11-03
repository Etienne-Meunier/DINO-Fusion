"""
Shared constants and helpers of the phase 1 scripts of the assimilation study (see plan.md): the cache of the last-year
data states (SST sources and data reference), the normalisation of a data state into the padded model space, and the
run folders, one per sampler of 1_generate_sst.py (RUN_SETS: centred in OUT1, standard in OUT1_standard).

Run directly, builds the cache.

Inputs:
    TrainingConfig().data_file  training tar, unnormalised NNNNN.{toce,soce,ssh}.npy, time-ordered
Outputs in OUT1:
    data_states_<first>_<last>.npz  indices, toce, soce (n, 36, 199, 62) and ssh (n, 199, 62) of the last YEAR states,
                                    full file grid, float32, land as stored in the tar
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')

import os
import tarfile

import numpy as np
import torch

from seasonal import DATA_FILE, N_STATES, OUT, SUMMER, WINTER, YEAR, read

OUT1 = OUT.replace('phase0', 'phase1')
RUN_SETS = {'centred': OUT1, 'standard': f'{OUT1}_standard'}
INDICES1 = np.arange(N_STATES - YEAR, N_STATES)
CACHE1 = f'{OUT1}/data_states_{INDICES1[0]}_{INDICES1[-1]}.npz'
SOURCES = {'summer': INDICES1[INDICES1 % YEAR == SUMMER][0], 'winter': INDICES1[INDICES1 % YEAR == WINTER][0]}


def load_states():
    """Dict of the cached last-year states (see the module docstring), built from the tar if missing."""
    if not os.path.exists(CACHE1):
        with tarfile.open(DATA_FILE) as tar:
            fields = {f: np.stack([read(tar, i, f) for i in INDICES1]).astype(np.float32) for f in ['toce', 'soce', 'ssh']}
        os.makedirs(OUT1, exist_ok=True)
        np.savez(CACHE1, indices=INDICES1, **fields)
    return dict(np.load(CACHE1))


def normalised_state(states, index, observer):
    """State `index` of `states` (load_states) in the padded model space of `observer`.normalizer, (37, 208, 64)
    float32 tensor on the normalizer device, land and padding 0."""
    k = int(np.flatnonzero(states['indices'] == index)[0])
    return observer.normalizer({f: torch.from_numpy(states[f][k].copy()) for f in ['toce', 'soce', 'ssh']}).float()


def ocean_mask(observer):
    """Ocean cells of the padded model space, (37, 208, 64) float, 1 on ocean."""
    return observer.normalizer.padder((~observer.normalizer.masker.mask).float())


if __name__ == '__main__':
    print({k: v.shape for k, v in load_states().items()}, SOURCES)
