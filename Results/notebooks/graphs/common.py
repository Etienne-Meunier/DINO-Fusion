#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared state for the graph scripts: data batch, losses and every generation of a prefix.

Imported with ``from common import *`` so an interactive session loads the data once.
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
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, eps_data, get_loss

device = 'cpu'
batch_size = 200

RUN = 'tav0h83b'
PREFIX = 'optimal'
INFERENCE = f'{os.environ["OCEANDATA"]}/models/dino-fusion/{RUN}/inference'


def load_generations(prefix=PREFIX, inference=INFERENCE, latest_only=True, device=device):
    """Load every generation under ``<inference>/<prefix>/<constraint>/<timestamp>/generated.npy``.

    Returns (gens, args) dicts keyed by constraint name when `latest_only`, else by
    'constraint/timestamp'. `args` holds the matching args.json content.
    """
    gens, args = {}, {}
    for path in sorted(glob(f'{inference}/{prefix}/*/*/generated.npy')):
        timestamp = os.path.basename(os.path.dirname(path))
        constraint = os.path.basename(os.path.dirname(os.path.dirname(path)))
        key = constraint if latest_only else f'{constraint}/{timestamp}'
        gens[key] = torch.from_numpy(np.load(path)).to(device)
        with open(f'{os.path.dirname(path)}/args.json') as f:
            args[key] = json.load(f)
    return gens, args


def load_mask():
    """Return the 1 degree DINO mesh mask with dimensions renamed to (depth, nav_lat, nav_lon)."""
    file_mask_LR = xr.open_dataset(f'{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc').sel(time_counter=0)
    return file_mask_LR.rename({'nav_lev': 'depth', 'y': 'nav_lat', 'x': 'nav_lon'})


config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
dataloader = get_dataloader(config.data_file, transform, batch_size=batch_size, shuffle=True, seed=DATA_SEED)
observer = Observer('data', config, device=device)
losses = {name: get_loss(name, observer) for name in LOSSES}

batch = next(iter(dataloader))
gens, gens_args = load_generations()

sources = {'data': batch} | gens
