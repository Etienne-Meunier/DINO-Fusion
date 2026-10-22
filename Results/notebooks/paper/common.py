#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared state for the paper scripts: data batch, losses and the generations of one sweep of one run.

Imported with ``from common import *`` so an interactive session loads the data once. The model and sweep
are RUN / PREFIX, overridable with the PAPER_RUN / PAPER_PREFIX environment variables.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')
sys.path.append(PRP)

import json
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, eps_data, get_loss
from metrics import get_density_at_surface

device = 'cpu'
batch_size = 200

RUN = os.environ.get('PAPER_RUN', 'tav0h83b')
PREFIX = os.environ.get('PAPER_PREFIX', 'optimal')
INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/{RUN}/inference'
FIGURES = f'{os.path.dirname(os.path.abspath(__file__))}/figures'
NAMES = {'data': 'Data',
         'none': 'No constraint',
         'none_border0': 'Boundary projection',
         'zero_mean_border0': 'T,S mean constraint',
         'negative_grad_density_huber_border0': 'Increasing density constraint',
         'mean_density_border0': 'Mean density constraint',
         'negative_grad_density_huber_border0_30': 'Increasing density constraint, eta = 30',
         'negative_grad_density_huber_border0_80': 'Increasing density constraint, eta = 80'}


def generation_dirs(prefix=PREFIX, inference=INFERENCE):
    """Folder ``<inference>/<prefix>/<constraint>/<timestamp>/`` holding the generated.npy of every run, keyed by constraint.

    A run is expected to have a single timestamp: when it has several, an error is printed and the latest one is kept.
    """
    dirs = {}
    for run_dir in sorted(glob(f'{inference}/{prefix}/*/')):
        constraint = os.path.basename(os.path.normpath(run_dir))
        versions = sorted(glob(f'{run_dir}/*/generated.npy'),
                          key=lambda v: datetime.strptime(os.path.basename(os.path.dirname(v)), '%d%m%y-%H%M%S'))
        if not versions:
            continue
        if len(versions) > 1:
            print(f'ERROR : {prefix}/{constraint} has {len(versions)} versions '
                  f'({", ".join(os.path.basename(os.path.dirname(v)) for v in versions)}), keeping the latest')
        dirs[constraint] = os.path.dirname(versions[-1])
    return dirs


def load_generations(prefix=PREFIX, inference=INFERENCE, device=device):
    """Load the generation of every run of `generation_dirs`.

    Returns (gens, args) dicts keyed by constraint name, `args` holding the matching args.json.
    """
    gens, args = {}, {}
    for constraint, path in generation_dirs(prefix, inference).items():
        gens[constraint] = torch.from_numpy(np.load(f'{path}/generated.npy')).to(device)
        with open(f'{path}/args.json') as f:
            args[constraint] = json.load(f)
    return gens, args


def load_mask():
    """Return the 1 degree DINO mesh mask with dimensions renamed to (depth, nav_lat, nav_lon)."""
    file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
    return file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})


def compute_density(fields, file_mask_LR):
    """Potential density referenced at the surface, (batch, depth, nav_lat, nav_lon), of unnormalized 'toce', 'soce'."""
    bsize = fields['soce'].shape[0]
    data = file_mask_LR.e3t_0.copy().expand_dims({'batch': bsize})
    soce, toce, tmask = data.copy(), data.copy(), data.copy()
    soce[:] = np.asarray(fields['soce'])
    toce[:] = np.asarray(fields['toce'])
    tmask[:] = np.repeat(file_mask_LR.tmask.values[None], bsize, 0)
    return get_density_at_surface(toce, soce, tmask)


def zonal_sections(x, file_mask_LR):
    """Volume-weighted zonal mean of sigma_0 - 1000 of a normalized batch x, (batch, depth - 1, nav_lat), last (land) level dropped."""
    fields = {k: v.detach().cpu().numpy() for k, v in observer.unnormalized(x).items()}
    volume = (file_mask_LR.e1t * file_mask_LR.e2t * file_mask_LR.e3t_0 * file_mask_LR.tmask).squeeze()
    volume = volume.transpose('depth', 'nav_lat', 'nav_lon')
    section = (compute_density(fields, file_mask_LR) * volume).sum('nav_lon') / volume.sum('nav_lon')
    return (section.transpose('batch', 'depth', 'nav_lat')[:, :-1] - 1000).values


def rmse(sections, reference):
    """Unweighted RMSE of each section (b, z, y) to `reference` (z, y), over the cells that are ocean in both."""
    return np.sqrt(np.nanmean((sections - reference[None]) ** 2, axis=(1, 2)))


def median_member(errors):
    """Index of the member at the median of `errors`, the upper median for an even batch."""
    return int(np.argsort(errors)[len(errors) // 2])


config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
dataloader = get_dataloader(config.data_file, transform, batch_size=batch_size, shuffle=True, seed=DATA_SEED)
observer = Observer('data', config, device=device)
losses = {name: get_loss(name, observer) for name in LOSSES}

batch = next(iter(dataloader))
gens, gens_args = load_generations()

sources = {'data': batch} | gens
