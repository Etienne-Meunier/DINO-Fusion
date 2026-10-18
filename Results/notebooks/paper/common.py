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
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, eps_data, get_loss

device = 'cpu'
batch_size = 200

RUN = os.environ.get('PAPER_RUN', 'tav0h83b')
PREFIX = os.environ.get('PAPER_PREFIX', 'optimal')
INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/{RUN}/inference'
FIGURES = f'{os.path.dirname(os.path.abspath(__file__))}/figures'
NAMES = {'data': 'Data',
         'none': 'No constraint',
         'none_border0': 'No constraint, border 0',
         'zero_mean_border0': 'TS constraint',
         'mean_density_border0': 'Density constraint',
         'negative_grad_density_huber_border0': 'Density monotonicity constraint'}


def load_generations(prefix=PREFIX, inference=INFERENCE, device=device):
    """Load the generation of every run under ``<inference>/<prefix>/<constraint>/<timestamp>/generated.npy``.

    A run is expected to have a single timestamp: when it has several, an error is printed and the latest
    one is kept. Returns (gens, args) dicts keyed by constraint name, `args` holding the matching args.json.
    """
    gens, args = {}, {}
    for run_dir in sorted(glob(f'{inference}/{prefix}/*/')):
        constraint = os.path.basename(os.path.normpath(run_dir))
        versions = sorted(glob(f'{run_dir}/*/generated.npy'))
        if not versions:
            continue
        if len(versions) > 1:
            print(f'ERROR : {prefix}/{constraint} has {len(versions)} versions '
                  f'({", ".join(os.path.basename(os.path.dirname(v)) for v in versions)}), keeping the latest')
        path = versions[-1]
        gens[constraint] = torch.from_numpy(np.load(path)).to(device)
        with open(f'{os.path.dirname(path)}/args.json') as f:
            args[constraint] = json.load(f)
    return gens, args


def load_mask():
    """Return the 1 degree DINO mesh mask with dimensions renamed to (depth, nav_lat, nav_lon)."""
    file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
    return file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})


config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
dataloader = get_dataloader(config.data_file, transform, batch_size=batch_size, shuffle=True)
observer = Observer('data', config, device=device)
losses = {name: get_loss(name, observer) for name in LOSSES}

batch = next(iter(dataloader))
gens, gens_args = load_generations()

sources = {'data': batch} | gens
