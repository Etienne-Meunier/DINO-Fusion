"""
Transfer check of the density-only losses to guided sampling, with UNet forward/backward passes only (no sampling).
States x (8 unconstrained, 8 data) are noised to x_t at t in TS, the Tweedie estimate x0_hat(x_t) is formed as in
`sampling.sample`, and for each loss L :
    E(x0_hat), L(x0_hat) / L(x),
    cos = cosine between the guidance gradient grad_{x_t} L(x0_hat(x_t)) and the clean gradient grad_x L(x),
    active = fraction of ocean cells where the guidance gradient is nonzero.
Also L on the clean data states (scale of eps). Output : `transfer.csv`.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

import os
import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pd_explore import ocean_mask
from frontier import get_candidate
from step1_bins import R, HERE, physical, split_E

LOSSES = ['C0', 'C1', 'C2_1e-4', 'C6_1e-4', 'C7_1e-4']
TS = [100, 300, 600]
MODEL = f"{os.environ.get('OCEANDATA', '/Volumes/LoCe/oceandata/')}/models/dino-fusion/tav0h83b"


def E_of(x, observer) :
    T, S = physical(x.detach(), observer)
    return split_E(T, S, observer.volume, torch.float64)[0]


def cosine(a, b, mask) :
    a, b = a[:, mask], b[:, mask]
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-30)


if __name__ == '__main__' :
    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL)
    sch = pipeline.scheduler
    losses = {n : get_candidate(n, observer) for n in LOSSES}

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    sets = {'DATA' : states[np.random.default_rng(1).choice(len(states), 8, replace=False)],
            'NONE' : torch.tensor(np.load(f'{R}/none/generated.npy'))}
    del states

    rows = []
    for sname, x in sets.items() :
        clean = {}
        for n, l in losses.items() :
            xc = x.clone().requires_grad_(True)
            v = l(xc)
            g, = torch.autograd.grad(v.sum(), xc)
            clean[n] = (v.detach(), g)
            rows.append({'set' : sname, 't' : 0, 'loss' : n, 'L' : v.mean().item(), 'L_ratio' : 1.0, 'cos' : 1.0,
                         'active' : (g[:, mask] != 0).float().mean().item(), 'E' : E_of(x, observer).mean().item()})
        gen = torch.Generator().manual_seed(0)
        for t in TS :
            abar = sch.alphas_cumprod[t]
            xt = (abar.sqrt() * x + (1 - abar).sqrt() * torch.randn(x.shape, generator=gen)).requires_grad_(True)
            noise_pred = pipeline.unet(xt, t).sample
            x0_hat = ((xt - (1 - abar).sqrt() * noise_pred) / abar.sqrt()).clamp(-sch.config.clip_sample_range, sch.config.clip_sample_range)
            E_hat = E_of(x0_hat, observer).mean().item()
            for n, l in losses.items() :
                v = l(x0_hat)
                g, = torch.autograd.grad(v.sum(), xt, retain_graph=True)
                rows.append({'set' : sname, 't' : t, 'loss' : n, 'L' : v.mean().item(),
                             'L_ratio' : (v.detach() / clean[n][0].clamp(min=1e-30)).mean().item(),
                             'cos' : cosine(g, clean[n][1], mask).mean().item(),
                             'active' : (g[:, mask] != 0).float().mean().item(), 'E' : E_hat})
            print(sname, t, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'transfer.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda v : f'{v:.3g}'))
