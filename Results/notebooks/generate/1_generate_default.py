#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained sampling for each loss with its default eps (`eps_data`), volume_weighted=False.
Outputs of `generate` in {model_path}/inference/{prefix}/{loss}/{time}/
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import os
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.losses import LOSSES
from generate import generate


model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b'
prefix = 'default_eps'
volume_weighted = False
batch = 16
inf_steps = 1000

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to(device)
pipeline.scheduler.set_timesteps(inf_steps)
observer = Observer('data', TrainingConfig(), device=device)

for loss in LOSSES :
    generate(pipeline, observer, f'{model_path}/inference/{prefix}', loss=loss, volume_weighted=volume_weighted,
             batch=batch, info=dict(model_path=model_path, prefix=prefix), projection='none')
