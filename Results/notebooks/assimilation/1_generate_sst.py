#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 of the assimilation study (see plan.md): generate DINO states with the diffusion model conditioned on the SST
(toce level 0) of a fixed data state, unconstrained or with one constraint, on one of two samplers (--sampler):
    centred  : the sampler of `machine/surface_bias/experiments` (exp.ddpm_step: centred noise, noisy land cells along
               the trajectory), the projection only after the last step, the constraint on the field with land and
               padding set to 0 (exp.OceanOnly) for zero_mean;
    standard : the DDPM sampler `pipelines.sampling.sample` of generate.py and the paper "optimal" runs (no centred
               noise, no noisy land), the projection after every step, the loss of `pipelines.losses.get_loss`.

The whole batch shares the SST of the source state (phase0 SUMMER or WINTER state of the last year, phase1.SOURCES),
normalised into the padded model space. After every denoising step (and its guidance step and projection), the ocean
cells of channel 0 are overwritten:
    noisy : by scheduler.add_noise(sst0, z, t - 1), the observation at the noise level of the new state, z fresh;
    clean : by sst0.
After the last step both give sst0. The initial noise gets the same overwrite at the first timestep. The z of the
noisy overwrite come from their own generator, so the trajectory noise draws are those of the unconditioned run of the
same sampler, seed and batch (exp.sample or sampling.sample). Centred runs made before this separate generator are not
member-paired with the unconditioned references.

    python 1_generate_sst.py --season summer --variant none --eta 0 --eps 0 --name summer_none
    python 1_generate_sst.py --sampler standard --projection none --season summer --variant none --eta 0 --eps 0 \
        --name summer_none
    python 1_generate_sst.py --sampler standard --season winter --variant c7 --eta 30 --eps 8.6e-8 --name winter_C7

Inputs:
    MODEL (tav0h83b) pipeline, phase1.CACHE1 (last-year data states)
Outputs in RUN_SETS[sampler]/<name>/ (or --out/<name>/):
    generated.npy (b, 37, 208, 64) normalised, lbd.npy and trace.npy (steps, 2) mean lbd and loss if constrained,
    args.json every setting, the loss name, the source state index and the runtime.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/notebooks/machine/surface_bias/experiments/')

import argparse
import os
import json
import time

import numpy as np
import torch
from exp import MODEL, OceanOnly, ddpm_step
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.losses import BorderZero, ZeroMean, get_loss
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer

from phase1 import RUN_SETS, SOURCES, load_states, normalised_state, ocean_mask

LOSS_NAMES = {'c7': 'negative_grad_density_huber', 'zero_mean': 'zero_mean'}


def observation_injector(sch, sst0, ocean, batch_size, mode, seed):
    """
    Function inject(x, t_noise) returning x with the ocean cells of channel 0 replaced by sst0 (208, 64) at the noise
    level of timestep t_noise of the scheduler `sch` (noisy `mode`), or by sst0 itself (clean `mode` or t_noise < 0).
    The noise z of the noisy mode is drawn from a generator of its own, seeded seed + 2**32. ocean (37, 208, 64) the
    ocean mask.
    """
    device = sst0.device
    generator = torch.Generator(device).manual_seed(seed + 2 ** 32)
    obs_cells = ocean[0] > 0
    sst = sst0.expand(batch_size, -1, -1)

    def inject(x, t_noise):
        if mode == 'clean' or t_noise < 0:
            y = sst
        else:
            z = torch.randn(sst.shape, generator=generator, device=device)
            y = sch.add_noise(sst, z, torch.tensor([int(t_noise)]))
        x = x.clone()
        x[:, 0] = torch.where(obs_cells, y, x[:, 0])
        return x

    return inject


def check_consecutive(sch):
    """Assert that the timesteps of `sch` are consecutive (set_timesteps(num_train_timesteps), or a tail of them): the
    overwrite after timestep t uses the noise level t - 1."""
    assert torch.all(sch.timesteps[:-1] - sch.timesteps[1:] == 1), 'the SST samplers need consecutive timesteps'


def guidance(sch, x, t, noise_pred, loss_c, lbd, eps, constraint_mode='raw'):
    """Primal-dual guidance term of `sampling.sample` at timestep t: loss_c on the clipped Tweedie estimate of x,
    constraint g = loss - eps ('raw') or sqrt(loss) - sqrt(eps) ('sqrt'), and the gradient of sum(lbd * g) with
    respect to x (x requires grad and noise_pred depends on it). Returns loss (b,), g (b,), grad (b, c, i, j)."""
    abar = sch.alphas_cumprod[t].to(x.device)
    x0_hat = ((x - (1 - abar).sqrt() * noise_pred) / abar.sqrt()).clamp(-sch.config.clip_sample_range,
                                                                         sch.config.clip_sample_range)
    loss = loss_c(x0_hat)
    g = loss - eps if constraint_mode == 'raw' else (loss + 1e-12).sqrt() - eps ** 0.5
    grad, = torch.autograd.grad((lbd * g).sum(), x)
    return loss, g, grad


def sample_sst(pipeline, batch_size, sst0, ocean, mode='noisy', loss_c=None, eta=0., eps=0., projection=None, seed=0,
               lbd0=1.):
    """
    Centred sampler: `exp.sample` restricted to one constraint (no reference, no t_on, no grad_weights) with the SST
    overwrite of the module docstring, the guidance step on the ocean cells only, `projection` after the last step.
    sst0 (208, 64) the normalised source SST, ocean (37, 208, 64) the ocean mask. Returns x (b, 37, 208, 64), lbd (b,)
    or None, trace (steps, 2) of mean lbd and loss or None.
    """
    sch = pipeline.scheduler
    check_consecutive(sch)
    device = pipeline.device
    generator = torch.Generator(device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    centre = lambda z: z - (z * ocean).sum((-2, -1), keepdim=True) / ocean.sum((-2, -1), keepdim=True) * ocean
    inject = observation_injector(sch, sst0, ocean, batch_size, mode, seed)

    x = inject(centre(torch.randn(shape, generator=generator, device=device)), int(sch.timesteps[0]))
    lbd = None if loss_c is None else torch.full((batch_size,), float(lbd0), device=device)
    trace = []
    for t in pipeline.progress_bar(sch.timesteps):
        x = x.detach().requires_grad_(loss_c is not None)
        with torch.set_grad_enabled(loss_c is not None):
            noise_pred = pipeline.unet(x, t).sample
        if loss_c is not None:
            loss, g, grad = guidance(sch, x, t, noise_pred, loss_c, lbd, eps)
        with torch.no_grad():
            x = ddpm_step(sch, noise_pred, t, x, generator, ocean)
            if loss_c is not None:
                x = x - grad * ocean
                lbd = (lbd + eta * g).clamp(min=0)
                trace.append(torch.stack([lbd.mean(), loss.mean()]).cpu())
            x = inject(x, int(t) - 1)
            if t == 0 and projection is not None:
                x = projection(x)
    return x.detach(), lbd, (torch.stack(trace).numpy() if trace else None)


def sample_sst_standard(pipeline, batch_size, sst0, ocean, mode='noisy', loss_c=None, eta=0., eps=0.,
                        constraint_mode='raw', projection=None, seed=0, lbd0=0.):
    """
    Standard sampler: `pipelines.sampling.sample` (same arguments, constraint_mode its `mode`) with the SST overwrite
    of the module docstring after the DDPM step, the guidance step and `projection`, applied at every step. sst0
    (208, 64) the normalised source SST, ocean (37, 208, 64) the ocean mask. Returns x (b, 37, 208, 64), lbd (b,) or
    None, trace (steps, 2) of mean lbd and loss or None.
    """
    sch = pipeline.scheduler
    check_consecutive(sch)
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    inject = observation_injector(sch, sst0, ocean, batch_size, mode, seed)

    x = inject(torch.randn(shape, generator=generator, device=pipeline.device), int(sch.timesteps[0]))
    lbd = None if loss_c is None else torch.full((batch_size,), float(lbd0), device=pipeline.device)
    trace = []
    for t in pipeline.progress_bar(sch.timesteps):
        x = x.detach().requires_grad_(loss_c is not None)
        with torch.set_grad_enabled(loss_c is not None):
            noise_pred = pipeline.unet(x, t).sample
        if loss_c is not None:
            loss, g, grad = guidance(sch, x, t, noise_pred, loss_c, lbd, eps, constraint_mode)
        with torch.no_grad():
            x = sch.step(noise_pred, t, x, generator=generator).prev_sample
            if loss_c is not None:
                x = x - grad
                lbd = (lbd + eta * g).clamp(min=0)
                trace.append(torch.stack([lbd.mean(), loss.mean()]).cpu())
            if projection is not None:
                x = projection(x)
            x = inject(x, int(t) - 1)
    return x.detach(), lbd, (torch.stack(trace).numpy() if trace else None)


def make_loss(sampler, variant, observer, ocean, volume_weighted=False):
    """Constraint loss of `variant` ('none', 'c7' or 'zero_mean') for `sampler`: None, the loss of get_loss
    (LOSS_NAMES) for the standard sampler, for the centred one zero_mean on the field with land and padding at 0."""
    if variant == 'none':
        return None
    if sampler == 'centred' and variant == 'zero_mean':
        return OceanOnly(ZeroMean(observer=observer), ocean)
    return get_loss(LOSS_NAMES[variant], observer, volume_weighted=volume_weighted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sampler', choices=list(RUN_SETS), default='centred')
    parser.add_argument('--season', choices=list(SOURCES), required=True)
    parser.add_argument('--variant', choices=['none', *LOSS_NAMES], required=True)
    parser.add_argument('--eta', type=float, required=True)
    parser.add_argument('--eps', type=float, required=True)
    parser.add_argument('--lbd0', type=float, default=1.0)
    parser.add_argument('--mode', choices=['noisy', 'clean'], default='noisy', help='SST overwrite')
    parser.add_argument('--constraint_mode', choices=['raw', 'sqrt'], default='raw', help='standard sampler only')
    parser.add_argument('--volume_weighted', action='store_true', help='standard sampler only')
    parser.add_argument('--projection', choices=['border_zero', 'none'], default='border_zero')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None,
                        help='keep only the last `steps` timesteps (smoke test), all of them if not given')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--out', type=str, default=None, help='output folder, RUN_SETS[sampler] if not given')
    args = parser.parse_args()
    args.out = args.out or RUN_SETS[args.sampler]

    device = Accelerator().device
    pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL).to(device)
    pipeline.scheduler.set_timesteps(pipeline.scheduler.config.num_train_timesteps)
    if args.steps is not None:
        pipeline.scheduler.timesteps = pipeline.scheduler.timesteps[-args.steps:]
    obs = Observer('data', TrainingConfig(), device=device)
    ocean = ocean_mask(obs).to(device)
    source = int(SOURCES[args.season])
    sst0 = normalised_state(load_states(), source, obs)[0].to(device)
    loss_c = make_loss(args.sampler, args.variant, obs, ocean, args.volume_weighted)
    projection = BorderZero(obs) if args.projection == 'border_zero' else None

    t0 = time.time()
    common = dict(mode=args.mode, loss_c=loss_c, eta=args.eta, eps=args.eps, projection=projection, seed=args.seed,
                  lbd0=args.lbd0)
    if args.sampler == 'centred':
        x, lbd, trace = sample_sst(pipeline, args.batch, sst0, ocean, **common)
    else:
        x, lbd, trace = sample_sst_standard(pipeline, args.batch, sst0, ocean, constraint_mode=args.constraint_mode,
                                            **common)
    out = f'{args.out}/{args.name}'
    os.makedirs(out, exist_ok=True)
    np.save(f'{out}/generated.npy', x.cpu().numpy())
    if lbd is not None:
        np.save(f'{out}/lbd.npy', lbd.cpu().numpy())
        np.save(f'{out}/trace.npy', trace)
    err = (x[:, 0] - sst0)[:, ocean[0] > 0].abs().max().item()
    land = x[ocean[None].expand_as(x) == 0].abs().max().item()
    with open(f'{out}/args.json', 'w') as f:
        json.dump(dict(vars(args), steps=len(pipeline.scheduler.timesteps), source_index=source, model=MODEL,
                       centred=args.sampler == 'centred', loss=LOSS_NAMES.get(args.variant),
                       sst_max_abs_err_normalised=err, land_max_abs_normalised=land,
                       minutes=(time.time() - t0) / 60), f, indent=2)
    print(f'Saved in {out} ({(time.time() - t0) / 60:.1f} min), max |x0 - sst0| on ocean {err:.3g}, '
          f'max |x| on land and padding {land:.3g}')
