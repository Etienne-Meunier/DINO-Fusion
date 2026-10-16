"""
Exploration variants of the primal-dual constrained sampler `pipelines.sampling.sample`, with a per-step trace, and the
metrics used to compare generations (constraint satisfaction, distance to the unconstrained state of the same seed,
diversity over the batch).

Variants (arguments of `sample_pd`) :
    mode : constraint function given to the dual variable
        'raw' : g = loss - eps                   (pipelines.sampling.sample)
        'rel' : g = loss / eps - 1
        'log' : g = log(loss) - log(eps)         (gradient lbd * grad(loss) / loss, independent of the loss scale)
        'sqrt' : g = sqrt(loss) - sqrt(eps)      (losses are squared distances : bounded gradient near the constraint)
    t_start : guidance only for t <= t_start (None : all steps)
    lbd_max : cap on the dual variable (None : no cap)
    final_opt : number of LBFGS steps of `OptimProjection` applied on the final state (0 : none)
    through_unet : gradient of the loss of the Tweedie estimate through the UNet (DPS), else the noise prediction is
                   detached and the gradient is grad_x0(loss) / sqrt(abar)
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import numpy as np
import torch
from pipelines.losses import OptimProjection

LOG_FLOOR = 1e-12


class Stacked :
    """Several losses evaluated together, (b, c, i, j) -> (b, k), one dual variable per sample and per loss."""
    def __init__(self, losses) :
        self.losses = losses

    def __call__(self, field) :
        return torch.stack([l(field) for l in self.losses], dim=-1)


def constraint(loss, eps, mode) :
    """Constraint function g(loss) <= 0 given to the dual variable, loss (b,) or (b, k), eps float or (k,)."""
    eps = torch.as_tensor(eps, dtype=loss.dtype, device=loss.device)
    if mode == 'raw' :
        return loss - eps
    elif mode == 'rel' :
        return loss / eps - 1
    elif mode == 'log' :
        return torch.log(loss + LOG_FLOOR) - torch.log(eps + LOG_FLOOR)
    elif mode == 'sqrt' :
        return (loss + LOG_FLOOR).sqrt() - eps.sqrt()
    raise ValueError(f'Unknown mode {mode}')


def sample_pd(pipeline, batch_size, loss_c=None, eta=0.0, eps=0.0, projection=None, seed=0, mode='raw', t_start=None,
              lbd_max=None, final_opt=0, through_unet=True) :
    """
        Same sampler as `pipelines.sampling.sample` (Tweedie estimate clipped to the scheduler range) with the variants of the module docstring.
    Returns :
        x (b, c, i, j) : generated normalised states
        lbd (b,) or (b, k) for a `Stacked` loss : final dual variables, None if loss_c is None
        trace : {'t', 'loss_x0', 'lbd', 'grad_norm', 'step_norm' : np.array (steps, b)} along the sampling, loss_x0 is
                the loss of the Tweedie estimate, step_norm the norm of the DDPM update x_t -> x_{t-1}
    """
    sch = pipeline.scheduler
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    x = torch.randn(shape, generator=generator, device=pipeline.device)
    lbd = None
    trace = {k : [] for k in ['t', 'loss_x0', 'lbd', 'grad_norm', 'step_norm']}

    for t in pipeline.progress_bar(sch.timesteps) :
        active = loss_c is not None and (t_start is None or bool(t <= t_start))
        x = x.detach().requires_grad_(active)
        with torch.set_grad_enabled(active and through_unet) :
            noise_pred = pipeline.unet(x, t).sample
        with torch.set_grad_enabled(active) :
            if active :
                abar = sch.alphas_cumprod[t].to(x.device)
                x0_hat = (x - (1 - abar).sqrt() * noise_pred) / abar.sqrt()
                x0_hat = x0_hat.clamp(-sch.config.clip_sample_range, sch.config.clip_sample_range)
                loss = loss_c(x0_hat)
                g = constraint(loss, eps, mode)
                if lbd is None :
                    lbd = torch.zeros_like(loss.detach())
                grad, = torch.autograd.grad((lbd * g).sum(), x)
        with torch.no_grad() :
            x_prev = sch.step(noise_pred, t, x, generator=generator).prev_sample
            step_norm = (x_prev - x).flatten(1).norm(dim=1)
            x = x_prev
            if active :
                x = x - grad
                lbd = (lbd + torch.as_tensor(eta, dtype=g.dtype, device=g.device) * g.detach()).clamp(min=0)
                if lbd_max is not None :
                    lbd = lbd.clamp(max=lbd_max)
            if projection is not None :
                x = projection(x)
        trace['t'].append(np.full(batch_size, int(t)))
        trace['step_norm'].append(step_norm.cpu().numpy())
        trace['loss_x0'].append(loss.detach().cpu().numpy() if active else None)
        trace['lbd'].append(lbd.cpu().numpy() if lbd is not None else None)
        trace['grad_norm'].append(grad.flatten(1).norm(dim=1).cpu().numpy() if active else np.zeros(batch_size))

    x = x.detach()
    if final_opt > 0 and loss_c is not None :
        x = OptimProjection(loss_c, eps=torch.as_tensor(eps, device=x.device), max_opt_steps=final_opt)(x)
        if projection is not None :
            x = projection(x)
    for k in ['loss_x0', 'lbd'] :
        shape = next((v.shape for v in trace[k] if v is not None), (batch_size,))
        trace[k] = [np.full(shape, np.nan) if v is None else v for v in trace[k]]
    return x, lbd, {k : np.stack(v) for k, v in trace.items()}


def ocean_mask(observer) :
    """Boolean (c, i, j) of the ocean cells of the padded normalised state."""
    land = observer.normalizer.masker.mask
    return ~observer.normalizer.padder(land.float()[None])[0].bool()


def diversity(x, mask) :
    """
        Mean over ocean cells of the standard deviation over the batch, by field group, x (b, c, i, j) normalised.
    Returns :
        {'toce', 'soce', 'ssh', 'all' : float}
    """
    std = x.std(dim=0)
    groups = {'toce' : slice(0, 18), 'soce' : slice(18, 36), 'ssh' : slice(36, 37), 'all' : slice(0, 37)}
    return {k : std[s][mask[s]].mean().item() for k, s in groups.items()}


def distance(x, x_ref, mask) :
    """RMS over ocean cells of x - x_ref, per sample (b,)."""
    d = (x - x_ref)[:, mask]
    return d.pow(2).mean(dim=1).sqrt()


def variance_split(x, mask) :
    """
        Split the variance over the batch of the ocean cells, x (b, c, i, j) normalised, into the part carried by the
        spatial mean of each channel (one number per level and sample, what the horizontal-mean constraints act on) and
        the part of the spatial anomalies around it.
    Returns :
        {'total', 'level_mean', 'anomaly' : float}, mean over ocean cells of the variance over the batch,
        total = level_mean + anomaly
    """
    n = mask.sum(dim=(1, 2)).clamp(min=1)
    level_mean = (x * mask).sum(dim=(2, 3)) / n                                  # (b, c)
    anomaly = (x - level_mean[..., None, None]) * mask
    w = n / n.sum()
    v_mean = (level_mean.var(dim=0) * w).sum().item()
    v_anom = ((anomaly.var(dim=0) * mask).sum(dim=(1, 2)) / n * w).sum().item()
    return {'total' : v_mean + v_anom, 'level_mean' : v_mean, 'anomaly' : v_anom}


def inversion_fraction(x, observer, tol=0.0) :
    """
        Local static stability : fraction of the vertical pairs of ocean cells (strided levels) where the density
        decreases with depth by more than `tol` (kg/m3), per sample, x (b, c, i, j) normalised -> (b,)
    """
    density = observer.density(x, result='strided')
    inv = density[:, :-1] - density[:, 1:] > tol
    valid = ~(density[:, :-1].isnan() | density[:, 1:].isnan())
    return (inv & valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3))


def roughness(x, mask) :
    """
        Horizontal small-scale content : mean over pairs of neighbouring ocean cells (along i and j) of the squared
        difference, by field group, x (b, c, i, j) normalised -> {'toce', 'soce', 'ssh', 'all' : float}
    """
    groups = {'toce' : slice(0, 18), 'soce' : slice(18, 36), 'ssh' : slice(36, 37), 'all' : slice(0, 37)}
    out = {}
    for k, s in groups.items() :
        xs, ms = x[:, s], mask[s]
        di, mi = (xs[..., 1:, :] - xs[..., :-1, :]) ** 2, ms[..., 1:, :] & ms[..., :-1, :]
        dj, mj = (xs[..., :, 1:] - xs[..., :, :-1]) ** 2, ms[..., :, 1:] & ms[..., :, :-1]
        out[k] = ((di * mi).sum() + (dj * mj).sum()).item() / (x.shape[0] * (mi.sum() + mj.sum()).item())
    return out
