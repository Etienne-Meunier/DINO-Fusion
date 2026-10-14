"""
Generate states with the constraints of `pipelines/new_constraint.py`.

Methods :
    none        : plain DDPM sampling
    gradient    : gradient guidance on a loss (`--loss`), adaptive beta     (`sample_gradient`)
    projection  : augmented Lagrangian with a projection of the density   (`sample_projection`)

Outputs in `--output` (default `<model_path>/inference/<method>_<option>_<time>/`) :
    generated.npy        normalised states (b, 37, 208, 64)
    metrics.csv          dual variables along sampling
    args.json            arguments of the run
    clean/{toce,soce,ssh}.npy  un-normalised states, input of Results/a_priori
"""
from __init__ import PRP; import sys
sys.path.append(PRP)

import argparse, json, time, os
from functools import partial
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.new_constraint import (Beta, GradientConstraint, LossAD, BorderZero, IsotonicDensity, MeanDensityProfileProjection,
                                      MeanDensityProfile, MeanGradDensityProfile, NegativeGradDensityProfile, ZeroMean, DistancetoIsotonic)
from pipelines.constrained_sampling import sample_gradient, sample_projection
from post_processing import save_clean


def get_loss(name, observer) :
    """Loss of the gradient method, computed on the normalised state."""
    if name == 'mean_density' :
        return MeanDensityProfile(observer=observer, result='strided',
                                  profile=np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train_strided.txt'))
    elif name == 'mean_grad_density' :
        return MeanGradDensityProfile(observer=observer)
    elif name == 'negative_grad_density' :
        return NegativeGradDensityProfile(observer=observer)
    elif name == 'zero_mean' :
        return ZeroMean()
    elif name == 'dist_isoton' :
        return DistancetoIsotonic(observer=observer)
    raise ValueError(f'Unknown loss {name}')


def get_projection(name) :
    """Projection of the density (b, k, i, j) on the constraint set."""
    if name == 'isotonic' :
        return IsotonicDensity()
    elif name == 'mean_profile' :
        return MeanDensityProfileProjection()
    raise ValueError(f'Unknown projection {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate states with constraints")
    parser.add_argument("--model_path", type=str, help="path of the trained diffuser model")
    parser.add_argument("--output", type=str, default=None, help="output folder")
    parser.add_argument("--batch", type=int, default=8, help="Number of generated states")
    parser.add_argument("--inf_steps", type=int, default=1000, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=['none', 'gradient', 'projection'], default='none')
    # gradient
    parser.add_argument("--tweedie", action='store_true', help="gradient : evaluate the loss on the Tweedie estimate x0_hat instead of x_t")
    parser.add_argument("--loss", choices=['mean_density', 'mean_grad_density', 'negative_grad_density', 'zero_mean', 'dist_isoton'], default='mean_density')
    parser.add_argument("--beta", type=float, default=1e-5, help="initial beta (gradient)")
    parser.add_argument("--beta_type", choices=['constant', 'linear_up', 'linear_down', 'eq', 'ineq'], default='eq')
    parser.add_argument("--eps", type=float, default=0.0, help="gradient, ineq : tolerance of the constraint C <= eps")
    # projection
    parser.add_argument("--projection", choices=['isotonic', 'mean_profile'], default='isotonic')
    parser.add_argument("--tau", type=float, default=5e-3, help="step size of the gradient step on the augmented Lagrangian (projection)")
    parser.add_argument("--eta_rho", type=float, default=3e-3, help="step size of rho (projection)")
    parser.add_argument("--rho_min", type=float, default=0.01, help="lower bound of rho (projection)")
    # common
    parser.add_argument("--eta", type=float, default=None, help="step size of the dual variable (default 1e-5 gradient, 7e-3 projection)")
    args = parser.parse_args()
    if args.eta is None :
        args.eta = 1e-5 if args.method == 'gradient' else 7e-3

    device = Accelerator().device
    pipeline = DDPMPipeline_Tensor.from_pretrained(args.model_path).to(device)
    pipeline.scheduler.set_timesteps(args.inf_steps)
    observer = Observer('data', TrainingConfig(), device=device)

    generator = torch.Generator(device).manual_seed(args.seed)
    shape = (args.batch, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    x = torch.randn(shape, generator=generator, device=device)

    if args.method == 'none' :
        option = ''
        with torch.no_grad() :
            for t in pipeline.progress_bar(pipeline.scheduler.timesteps) :
                x = pipeline.scheduler.step(pipeline.unet(x, t).sample, t, x, generator=generator).prev_sample
        metrics = None
    elif args.method == 'gradient' :
        option = args.loss
        constraint = GradientConstraint(Beta(beta_init=args.beta, beta_type=args.beta_type, params_beta={'eta' : args.eta, 'eps' : args.eps,
                                                                                           'max_iter' : pipeline.scheduler.config.num_train_timesteps}),
                                        LossAD(get_loss(args.loss, observer)))
        x, metrics = sample_gradient(pipeline, x, constraint, projection=BorderZero(observer), tweedie=args.tweedie)
    elif args.method == 'projection' :
        option = args.projection
        x, metrics = sample_projection(pipeline, x, partial(observer.density, result='strided'), get_projection(args.projection),
                                       data_projection=BorderZero(observer), tau=args.tau, eta=args.eta, eta_rho=args.eta_rho, rho_min=args.rho_min)

    output = args.output or f'{args.model_path}/inference/{args.method}_{option}_{time.strftime("%Y%m%d-%H%M%S")}'
    Path(output).mkdir(exist_ok=True, parents=True)
    np.save(f'{output}/generated.npy', x.cpu().numpy())
    if metrics is not None :
        metrics.to_csv(f'{output}/metrics.csv', index=False)
    json.dump(vars(args), open(f'{output}/args.json', 'w'), indent=2)
    save_clean(x, f'{output}/clean/', observer.normalizer)
    print(f'Saved in {output}')
