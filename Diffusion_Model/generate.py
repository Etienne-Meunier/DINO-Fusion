"""
Generate states with `pipelines/sampling.sample` : DDPM sampling, optionally constrained by a per-sample loss of
`pipelines/losses.py` (`--loss`, unconstrained sampling if not given).

Outputs in `<model_path>/inference/<prefix>/<loss or none>[_border0]/<time>/` (`_border0` with the BorderZero projection) :
    generated.npy        normalised states (b, 37, 208, 64)
    lbd.npy              final dual variables (b,), constrained sampling only
    args.json            arguments of the run
    clean/{toce,soce,ssh}.npy  un-normalised states, input of Results/a_priori
"""
from __init__ import PRP; import sys
sys.path.append(PRP)

import argparse, json, os, time
from pathlib import Path
import numpy as np
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.losses import BorderZero, LOSSES, eps_data, get_loss
from pipelines.sampling import sample
from post_processing import save_clean

PROJECTIONS = {'border_zero' : BorderZero, 'none' : None}


def generate(pipeline, observer, out_dir, loss=None, volume_weighted=False, eps='default', batch=8, eta=1e-3, mode='raw',
             projection='border_zero', seed=0, lbd0=0.0, info=None) :
    """
    Sample `batch` states, constrained by `loss` if given, and save them in `<out_dir>/<loss or none>[_border0]/<time>/`.
        pipeline (DDPMPipeline_Tensor) : diffusion model, scheduler timesteps already set
        observer (Observer) : for the loss and the projection
        loss : name in LOSSES, None for unconstrained sampling
        volume_weighted, eta, mode, seed, lbd0 : see `get_loss` and `sample`
        projection : name in PROJECTIONS, applied after each sampling step, 'none' for no projection
        eps : tolerance of the loss, 'default' takes eps_data[volume_weighted][loss]
        info (dict) : extra entries written in args.json
    Returns :
        x (b, c, i, j), lbd (b,) or None, output directory
    """
    loss_c = None if loss is None else get_loss(loss, observer, volume_weighted=volume_weighted)
    if loss is not None and eps == 'default' :
        eps = eps_data[volume_weighted][loss]
        if eps is None :
            raise ValueError(f'No default eps for {loss} with volume_weighted={volume_weighted}, give eps')

    proj = None if PROJECTIONS[projection] is None else PROJECTIONS[projection](observer)
    x, lbd = sample(pipeline, batch, loss_c=loss_c, eta=eta, eps=eps, mode=mode, projection=proj, seed=seed, lbd0=lbd0)

    name = (loss or 'none') + ('_border0' if projection == 'border_zero' else '')
    output = f'{out_dir}/{name}/{time.strftime("%d%m%y-%H%M%S")}'
    Path(output).mkdir(exist_ok=True, parents=True)
    np.save(f'{output}/generated.npy', x.cpu().numpy())
    if lbd is not None :
        np.save(f'{output}/lbd.npy', lbd.cpu().numpy())
    args = dict(loss=loss, volume_weighted=volume_weighted, eps=eps, batch=batch, eta=eta, mode=mode, projection=projection,
                seed=seed, lbd0=lbd0, inf_steps=len(pipeline.scheduler.timesteps), **(info or {}))
    json.dump(args, open(f'{output}/args.json', 'w'), indent=2)
    save_clean(x, f'{output}/clean/', observer.normalizer)
    print(f'Saved in {output}')
    return x, lbd, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate states, optionally constrained by a per-sample loss")
    parser.add_argument("--model_path", type=str, default=f"{os.environ.get('OCEANDATA', '')}/models/dino-fusion/tav0h83b", help="path of the trained diffuser model")
    parser.add_argument("--prefix", type=str, default=time.strftime("%d%m%y"), help="name of the inference folder, default today ddmmyy")
    parser.add_argument("--batch", type=int, default=8, help="Number of generated states")
    parser.add_argument("--inf_steps", type=int, default=1000, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", choices=LOSSES, default=None, help="constraint loss, unconstrained sampling if not given")
    parser.add_argument("--volume_weighted", action='store_true', help="weight the loss by the ocean volume")
    parser.add_argument("--eta", type=float, default=1e-3, help="step size of the dual variable")
    parser.add_argument("--eps", type=lambda s : s if s == 'default' else float(s), default='default',
                        help="tolerance of the constraint loss <= eps, 'default' takes eps_data of the loss")
    parser.add_argument("--lbd0", type=float, default=0.0, help="initial value of the dual variable")
    parser.add_argument("--mode", choices=['raw', 'sqrt'], default='raw', help="constraint function : loss - eps or sqrt(loss) - sqrt(eps)")
    parser.add_argument("--projection", choices=list(PROJECTIONS), default='border_zero', help="projection applied after each sampling step")
    args = parser.parse_args()

    device = Accelerator().device
    print(f"Device : {device}, model : {args.model_path}")
    pipeline = DDPMPipeline_Tensor.from_pretrained(args.model_path).to(device)
    pipeline.scheduler.set_timesteps(args.inf_steps)
    observer = Observer('data', TrainingConfig(), device=device)
    generate(pipeline, observer, f'{args.model_path}/inference/{args.prefix}', loss=args.loss,
             volume_weighted=args.volume_weighted, eps=args.eps, batch=args.batch, eta=args.eta, mode=args.mode,
             projection=args.projection, seed=args.seed, lbd0=args.lbd0, info=dict(model_path=args.model_path, prefix=args.prefix))
