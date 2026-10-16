#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep of the constraint strength (eps as a fraction of eps_data, eta, lbd0) for the paper constraints, all with the BorderZero projection :
    C1 : mean_density, TS : zero_mean
Outputs of `generate` in {model_path}/inference/{prefix}/{loss}_border0/{time}/, the sweep values are in args.json.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import os
from itertools import product
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from generate import generate
from pipelines.losses import eps_data


model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b'
prefix = 'sweep_strength'
volume_weighted = False
batch = 8
inf_steps = 1000

losses = ['mean_density', 'zero_mean']
eps_scales = [1.0, 0.1]  # eps = scale * eps_data, eps_data ~ max of the loss on data
eta_values = [1e-3, 1e-2, 1e-1, 1.0]
lbd0_values = [0.0, 0.1, 1.0]

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to(device)
pipeline.scheduler.set_timesteps(inf_steps)
observer = Observer('data', TrainingConfig(), device=device)

runs = list(product(eps_scales, eta_values, lbd0_values, losses))
for i, (scale, eta, lbd0, loss) in enumerate(runs) :
    eps = scale * eps_data[volume_weighted][loss]
    print(f'[{i + 1}/{len(runs)}] loss={loss} eps={eps} eta={eta} lbd0={lbd0}')
    generate(pipeline, observer, f'{model_path}/inference/{prefix}', loss=loss, volume_weighted=volume_weighted,
             eps=eps, eta=eta, lbd0=lbd0, batch=batch, projection='border_zero',
             info=dict(model_path=model_path, prefix=prefix))
