#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep of eta for the monotonicity-only density loss negative_grad_density_huber (C7 of
`Results/notebooks/machine/density_inversion/`), raw mode, eps = eps_data, lbd0 = 1, BorderZero, as the optimal runs.
The eta values are given on the command line so that the runs can be launched one at a time.
Outputs of `generate` in {model_path}/inference/{prefix}/negative_grad_density_huber_border0/{time}/, eta in args.json.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import os
import argparse
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from generate import generate
from pipelines.losses import eps_data


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--eta', type=float, nargs='+', required=True)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--prefix', type=str, default='sweep_density_huber')
args = parser.parse_args()

model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b'
loss = 'negative_grad_density_huber'
volume_weighted = False
inf_steps = 1000
lbd0 = 1.0
eps = eps_data[volume_weighted][loss]

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to(device)
pipeline.scheduler.set_timesteps(inf_steps)
observer = Observer('data', TrainingConfig(), device=device)

for i, eta in enumerate(args.eta) :
    print(f'[{i + 1}/{len(args.eta)}] loss={loss} eps={eps} eta={eta} lbd0={lbd0}')
    generate(pipeline, observer, f'{model_path}/inference/{args.prefix}', loss=loss, volume_weighted=volume_weighted,
             eps=eps, eta=eta, lbd0=lbd0, batch=args.batch, projection='border_zero',
             info=dict(model_path=model_path, prefix=args.prefix, constraint='C7'))
