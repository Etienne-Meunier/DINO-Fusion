#!/usr/bin/env python3
"""One-off: the reference density of the paper-clean figures, the batch-mean zonal sigma_0 section of the 200-state
data batch, saved once so that no figure script needs the DataLoader.

The batch is the first 200 states of the shuffle of the training archive seeded with DATA_SEED (TrainingConfig,
TransformationPipeline, get_dataloader with batch_size=200, shuffle=True). Its section is the volume-weighted (e1t e2t
e3t_0 of the 1° reference grid) zonal mean of TEOS-10 sigma_0 (`metrics.get_density_at_surface`, the EOS of the
training constraint, float64), last (land) level dropped, averaged over the members: (z_c - 1, y_c), as `common.zonal_sections`. Written to
OUT/reference_density.npy with its description in OUT/reference_density.json; refuses to overwrite them. Not imported
by anything: `python build_reference_density.py`.
"""
from __init__ import PRP; import sys
sys.path += [f'{PRP}/Diffusion_Model/', PRP, f'{PRP}/Results/a_posteriori/dinostics/', f'{PRP}/Results/analysis_scripts/']

import json
import os
import subprocess
import warnings
from datetime import datetime

import numpy as np
import torch
import xarray as xr

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    from experiment import NemoExperiment
from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from metrics import get_density_at_surface

sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')
from settings import REFERENCES_REL

BATCH_SIZE = 200
OUT = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference/optimal/reference_density'


def sections(x, grid, observer):
    """Volume-weighted zonal mean of sigma_0 of a normalized batch x on the T grid of `grid`, (member, z_c - 1, y_c)."""
    with torch.no_grad():
        fields = {k: v.cpu().numpy() for k, v in observer.unnormalized(x).items() if k in ('toce', 'soce')}
    mask = grid.mask
    ds = xr.Dataset({k: (('member', 'z_c', 'y_c', 'x_c'), v) for k, v in fields.items()},
                    coords={c: mask[c] for c in ('z_c', 'y_c', 'x_c')})
    sigma = (get_density_at_surface(ds.toce.astype(np.float64), ds.soce.astype(np.float64), np.ones(())) - 1000
             ).where(mask.tmask == 1)
    volume = (mask.e1t * mask.e2t * mask.e3t_0).where(mask.tmask == 1.0, 0.).transpose('z_c', 'y_c', 'x_c')
    section = (sigma * volume).sum('x_c') / volume.where(sigma.notnull()).sum('x_c')
    return section.transpose('member', 'z_c', 'y_c').values[:, :-1]


def main():
    """Load the data batch, compute its batch-mean section and write the .npy and .json of OUT."""
    npy, desc = f'{OUT}/reference_density.npy', f'{OUT}/reference_density.json'
    if os.path.exists(npy) or os.path.exists(desc):
        sys.exit(f'{OUT} already holds a reference density, not overwritten')
    grid = NemoExperiment(f'{os.environ["OCEANDATA"]}/{REFERENCES_REL}/', '1deg_restart', restarts=[''], files=[])
    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='cpu')
    batch = next(iter(get_dataloader(config.data_file, transform, batch_size=BATCH_SIZE, shuffle=True,
                                     seed=DATA_SEED)))
    reference = np.nanmean(sections(batch, grid, Observer('data', config, device='cpu')), axis=0)

    script = os.path.abspath(__file__)
    git = lambda *a: subprocess.run(['git', '-C', PRP, *a], capture_output=True, text=True).stdout.strip()
    commit = git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain', '--', script) else '')
    os.makedirs(OUT, exist_ok=True)
    np.save(npy, reference)
    with open(desc, 'w') as f:
        json.dump({'data_file': config.data_file, 'data_seed': DATA_SEED, 'batch': BATCH_SIZE,
                   'states': f'first {BATCH_SIZE} states of the shuffle seeded with data_seed (get_dataloader, '
                             f'shuffle=True)',
                   'quantity': 'batch-mean volume-weighted (e1t e2t e3t_0) zonal mean of sigma_0 = rho - 1000 '
                               '[kg m-3], last (land) level dropped',
                   'eos': 'TEOS-10 polynomial of Results/analysis_scripts/metrics.get_density_at_surface (the EOS of '
                          'the training constraint, pipelines/State.py), float64',
                   'grid': f'{REFERENCES_REL}/1deg_restart', 'dims': ['z_c', 'y_c'], 'shape': list(reference.shape),
                   'script': f'{script} @ {commit}', 'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                  f, indent=2)
    print(f'saved {npy} {reference.shape}, sigma_0 in [{np.nanmin(reference):.3f}, {np.nanmax(reference):.3f}]')


if __name__ == '__main__':
    main()
