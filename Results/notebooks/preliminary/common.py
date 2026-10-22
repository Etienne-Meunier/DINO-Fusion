#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:27:04 2026

@author: emeunier
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from configs.base_config import TrainingConfig
from DataLoader import DATA_SEED, get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, get_loss
import os


device = 'cpu'
losses=list(LOSSES)
batch = 200

config = TrainingConfig()
transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=device)
dataloader = get_dataloader(config.data_file, transform, batch_size=batch, shuffle=True, seed=DATA_SEED)
observer = Observer('data', config, device=device)

idt = iter(dataloader)

batch = next(idt)

models_dir = f'{os.environ["OCEANDATA"]}models/dino-fusion/tav0h83b/'
gen_path = f'{models_dir}/inference/151026/none/151026-150424/generated.npy'
gen = torch.from_numpy(np.load(gen_path)).to(device)