"""
Transfer check v2 (after review) : how useful the guidance gradient grad_{x_t} L(x0_hat(x_t)) is for the clean state.
For states x (8 unconstrained) noised to x_t at t in TS (3 noise draws) :
    step test : dE = E(x - s * u) - E(x), u = gradient normalised to unit RMS over ocean cells, s = STEP, for the
                guidance gradient and for the clean gradient grad_x L(x) (reference) ; ratio = dE_guided / dE_clean,
    cos       : cosine between the two gradients,
    clipped   : fraction of ocean cells of x0_hat at the clip bound.
For data states : quantiles q50 / q90 of L(x0_hat(x_t)) per t, the scale of an eps(t) calibrated on data.
Output : `transfer2.csv` (per state and draw), `transfer2_eps.csv`.
"""
from pathlib import Path
import sys
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

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
from transfer import MODEL, E_of, cosine
from step1_bins import R, HERE

LOSSES = ['C0', 'C1', 'C6_1e-4', 'C7_1e-4']
TS = [0, 50, 100, 200, 300, 450, 600]
DRAWS = 3
STEP = 0.005


def unit(g, mask) :
    """Gradient scaled to unit RMS over the ocean cells, zero elsewhere."""
    rms = g[:, mask].pow(2).mean(dim=1).sqrt().clamp(min=1e-30)
    return g * mask / rms[:, None, None, None]


def x0_hat_of(pipeline, x, t, gen) :
    sch = pipeline.scheduler
    abar = sch.alphas_cumprod[t]
    xt = (abar.sqrt() * x + (1 - abar).sqrt() * torch.randn(x.shape, generator=gen)).requires_grad_(True)
    noise_pred = pipeline.unet(xt, t).sample
    raw = (xt - (1 - abar).sqrt() * noise_pred) / abar.sqrt()
    r = sch.config.clip_sample_range
    return xt, raw.clamp(-r, r), (raw.abs() >= r)


if __name__ == '__main__' :
    config = TrainingConfig()
    observer = Observer('data', config)
    mask = ocean_mask(observer)
    pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL)
    losses = {n : get_candidate(n, observer) for n in LOSSES}
    gen = torch.Generator().manual_seed(0)

    x = torch.tensor(np.load(f'{R}/none/generated.npy'))
    E0 = E_of(x, observer)
    rows = []
    clean = {}
    for n, l in losses.items() :
        xc = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(l(xc).sum(), xc)
        clean[n] = g
        dE = E_of(x - STEP * unit(g, mask), observer) - E0
        rows += [{'t' : 0, 'draw' : 0, 'loss' : n, 'sample' : b, 'dE_clean' : dE[b].item()} for b in range(len(x))]
    clean_dE = pd.DataFrame(rows).set_index(['loss', 'sample']).dE_clean
    rows = []
    for t in [t for t in TS if t > 0] :
        for d in range(DRAWS) :
            xt, x0_hat, clipped = x0_hat_of(pipeline, x, t, gen)
            clip_frac = clipped[:, mask].float().mean(dim=1)
            for n, l in losses.items() :
                g, = torch.autograd.grad(l(x0_hat).sum(), xt, retain_graph=True)
                dE = E_of(x - STEP * unit(g, mask), observer) - E0
                cos = cosine(g, clean[n], mask)
                rows += [{'t' : t, 'draw' : d, 'loss' : n, 'sample' : b, 'dE_guided' : dE[b].item(),
                          'dE_clean' : clean_dE[(n, b)], 'cos' : cos[b].item(), 'clipped' : clip_frac[b].item()} for b in range(len(x))]
            print('NONE', t, d, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'transfer2.csv', index=False)

    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation)
    states = torch.cat([b for b in get_dataloader(config.data_file, transform, batch_size=32, shuffle=False)])
    xd = states[np.random.default_rng(2).choice(len(states), 16, replace=False)]
    del states
    eps_rows = []
    for t in TS :
        with torch.no_grad() :
            if t == 0 :
                x0s = [xd]
            else :
                x0s = [x0_hat_of(pipeline, xd, t, gen)[1].detach() for _ in range(2)]
            for n, l in losses.items() :
                v = torch.cat([l(x0) for x0 in x0s])
                eps_rows.append({'t' : t, 'loss' : n, 'q50' : v.quantile(0.5).item(), 'q90' : v.quantile(0.9).item(),
                                 'E_x0hat' : torch.cat([E_of(x0, observer) for x0 in x0s]).mean().item()})
        print('DATA', t, flush=True)
    eps = pd.DataFrame(eps_rows)
    eps.to_csv(HERE / 'transfer2_eps.csv', index=False)

    pd.set_option('display.width', 200)
    s = df.groupby(['loss', 't']).agg(dE_guided=('dE_guided', 'median'), dE_clean=('dE_clean', 'median'),
                                     cos=('cos', 'median'), cos_std=('cos', 'std'), clipped=('clipped', 'mean'))
    s['ratio'] = s.dE_guided / s.dE_clean
    print(s.to_string(float_format=lambda v : f'{v:.3g}'))
    print(eps.to_string(index=False, float_format=lambda v : f'{v:.3g}'))
