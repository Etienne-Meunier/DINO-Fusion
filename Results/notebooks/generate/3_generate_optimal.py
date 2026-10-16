#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained sampling with the selected (optimal) strength for each paper constraint, all with the BorderZero projection.
Outputs of `generate` in {model_path}/inference/{prefix}/{loss}_border0/{time}/
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import os
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from generate import generate


model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b'
prefix = 'optimal'
volume_weighted = False
batch = 16
inf_steps = 1000

constraints = {
    'C1': dict(loss='mean_density', eta=1e-2, lbd0=1.0, eps='default'),
    'TS': dict(loss='zero_mean', eta=1e-1, lbd0=1.0, eps='default'),
}

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to(device)
pipeline.scheduler.set_timesteps(inf_steps)
observer = Observer('data', TrainingConfig(), device=device)

for name, params in constraints.items() :
    print(f'{name} : {params}')
    generate(pipeline, observer, f'{model_path}/inference/{prefix}', volume_weighted=volume_weighted, batch=batch,
             projection='border_zero', info=dict(model_path=model_path, prefix=prefix, constraint=name), **params)
