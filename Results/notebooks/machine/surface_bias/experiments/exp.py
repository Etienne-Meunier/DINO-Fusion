"""
Experimental constraints against the surface lightening of C7 (negative_grad_density_huber), and a copy of
`pipelines.sampling.sample` that accepts several constraints at once, each with its own dual variable. Nothing of the
repository is modified, the losses of `pipelines/losses.py` are imported.

    profile   : one-sided Huber hinge on the sample's own volume-weighted mean density profile
    topfree   : C7 with the surface density detached (no push on level 0)
    c7_surf   : C7 and, as a second constraint, the volume-weighted mean T and S of level 0 close to the data mean
    c7_anchor : C7 and, as a second constraint, the level-0 mean T and S (or density) of the sample kept at those of the
                unconstrained sample of the same seed, no surface data statistics
"""
from pathlib import Path
import os, sys
PRP = f'{Path(__file__).resolve().parents[5]}/'
os.environ.setdefault('OCEANDATA', '/Volumes/LoCe/oceandata/')
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import numpy as np
import torch
from pipelines.losses import Loss, NegativeGradDensityHuber

HERE = Path(__file__).parent
MODEL = f"{os.environ['OCEANDATA']}/models/dino-fusion/tav0h83b"


def hinge(g, tau=1e-4) :
    """One-sided Huber of NegativeGradDensityHuber : 0 for g >= 0, g^2 / (2 tau) on (-tau, 0), -g - tau / 2 below."""
    h = torch.where(g < -tau, -g - tau / 2, g**2 / (2 * tau))
    return torch.where(g < 0, h, torch.zeros_like(g))


class ProfileHinge(Loss) :
    """
        Hinge on the steps of the sample's own volume-weighted horizontal-mean density profile rbar (b, 36),
        sum_k pw_k h(rbar[k+1] - rbar[k]), pw = ocean volume of the lower level normalised to 1.
    """
    def __init__(self, tau=1e-4, **kwargs) :
        super().__init__(**kwargs)
        self.tau = tau
        self.lv = self.observer.level_volume()
        pw = self.lv[1:] * (self.lv[:-1] > 0) * (self.lv[1:] > 0)
        self.pw = pw / pw.sum()

    def profile(self, field) :
        rho = torch.nan_to_num(self.observer.density(field))
        return (rho * self.observer.volume).sum((-2, -1)) / self.lv.clamp(min=1)

    def __call__(self, field) :
        r = self.profile(field)
        return (self.pw.to(r) * hinge(r[:, 1:] - r[:, :-1], self.tau)).sum(1)


class HuberTopFree(NegativeGradDensityHuber) :
    """C7 with rho[0] detached in the top pair : inversions of the pair (0, 1) only densify level 1."""
    def __call__(self, field) :
        rho = self.observer.density(field)
        up = torch.cat([rho[:, :1].detach(), rho[:, 1:-1]], 1)
        g = torch.where(self.valid, rho[:, 1:] - up, torch.zeros_like(up))
        h = torch.where(g < -self.tau, -g - self.tau / 2, g**2 / (2 * self.tau))
        return (self.w.to(g) * torch.where(g < 0, h, torch.zeros_like(g))).sum(dim=(1, 2, 3))


class SurfaceMean(Loss) :
    """
        Squared distance, in units of the data spread, of the volume-weighted mean T and S of level 0 to their data
        mean : ((T0 - mT) / sT)^2 + ((S0 - mS) / sS)^2, (mT, sT, mS, sS) from `calib.py` (all training states).
    """
    def __init__(self, stats=None, **kwargs) :
        super().__init__(**kwargs)
        s = stats or dict(np.load(HERE / 'calib.npz'))
        self.mT, self.sT, self.mS, self.sS = (float(s[k]) for k in ['T0_mean', 'T0_std', 'S0_mean', 'S0_std'])
        self.v0 = self.observer.volume[0] / self.observer.volume[0].sum()

    def means(self, field) :
        """Volume-weighted mean T and S of level 0, (b,), (b,)."""
        f = self.observer.unnormalized(field)
        v = self.v0.to(f['toce'])
        return (torch.nan_to_num(f['toce'][:, 0]) * v).sum((-2, -1)), (torch.nan_to_num(f['soce'][:, 0]) * v).sum((-2, -1))

    def __call__(self, field) :
        T0, S0 = self.means(field)
        return ((T0 - self.mT) / self.sT)**2 + ((S0 - self.mS) / self.sS)**2


class SurfaceZeroMean(Loss) :
    """
        The TS constraint (`ZeroMean`, unweighted) restricted to the normalised level-0 T and S channels :
        sum over c in `channels` of nanmean(field[c])^2, same units and scale as the zero_mean runs.
    """
    def __init__(self, channels=(0, 18), **kwargs) :
        super().__init__(**kwargs)
        self.channels = list(channels)

    def __call__(self, field) :
        return (torch.nanmean(field[:, self.channels], axis=(-2, -1))**2).sum(1)


class Reference :
    """
        Holder of the reference x0 (b, c, i, j) of the anchor losses : the x0 estimate of the unconstrained trajectory
        of the same seed at the current step (filled by `sample`), or a fixed batch (`x0` given).
    """
    def __init__(self, x0=None) :
        self.x0 = x0
        self.fixed = x0 is not None


class SurfaceAnchor(SurfaceMean) :
    """
        Squared distance of the level-0 volume-weighted mean T and S of the sample to those of the reference,
        ((T0 - T0r) / dT)^2 + ((S0 - S0r) / dS)^2, with fixed units dT (C) and dS (psu).
    """
    def __init__(self, ref, dT=0.1, dS=0.01, **kwargs) :
        Loss.__init__(self, **kwargs)
        self.ref, self.dT, self.dS = ref, dT, dS
        self.v0 = self.observer.volume[0] / self.observer.volume[0].sum()

    def __call__(self, field) :
        T0, S0 = self.means(field)
        with torch.no_grad() :
            T0r, S0r = self.means(self.ref.x0.to(field))
        return ((T0 - T0r) / self.dT)**2 + ((S0 - S0r) / self.dS)**2


class SurfaceAnchorRho(SurfaceAnchor) :
    """
        One-sided : the level-0 volume-weighted mean density of the sample may not be lighter than that of the
        reference, (max(0, rho0r - rho0) / drho)^2, drho in kg/m3.
    """
    def __init__(self, ref, drho=0.1, **kwargs) :
        super().__init__(ref, **kwargs)
        self.drho = drho

    def rho0(self, field) :
        rho = torch.nan_to_num(self.observer.density(field)[:, 0])
        return (rho * self.v0.to(rho)).sum((-2, -1))

    def __call__(self, field) :
        with torch.no_grad() :
            r = self.rho0(self.ref.x0.to(field))
        return ((r - self.rho0(field)).clamp(min=0) / self.drho)**2


class HuberSkipTop(NegativeGradDensityHuber) :
    """C7 without the top pair (0, 1) : no push on level 0, and none on level 1 from a surface inversion."""
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.w = self.w.clone()
        self.w[0] = 0


class ProfileAnchor(SurfaceAnchor) :
    """
        SurfaceAnchor on the K top levels : sum_k p_k [((T_k - T_kr) / dT)^2 + ((S_k - S_kr) / dS)^2], T_k, S_k the
        volume-weighted level means and p_k the volume of level k normalised to 1 over the K levels.
    """
    def __init__(self, ref, K=12, **kwargs) :
        super().__init__(ref, **kwargs)
        lv = self.observer.level_volume()[:K]
        self.K, self.p = K, lv / lv.sum()
        self.vk = self.observer.volume[:K] / lv.clamp(min=1)[:, None, None]

    def level_means(self, field) :
        """Volume-weighted level means of T and S, (b, K), (b, K)."""
        f = self.observer.unnormalized(field)
        v = self.vk.to(f['toce'])
        return tuple((torch.nan_to_num(f[k][:, :self.K]) * v).sum((-2, -1)) for k in ['toce', 'soce'])

    def __call__(self, field) :
        T, S = self.level_means(field)
        with torch.no_grad() :
            Tr, Sr = self.level_means(self.ref.x0.to(field))
        return (self.p.to(T) * (((T - Tr) / self.dT)**2 + ((S - Sr) / self.dS)**2)).sum(1)


class Detach(Loss) :
    """
        `loss` evaluated with the normalised channels `channels` detached (S levels 18-35 by default) : the guidance
        then acts only through the other channels (temperature).
    """
    def __init__(self, loss, channels=slice(18, 36)) :
        self.loss, self.channels = loss, channels

    def __call__(self, field) :
        mask = torch.zeros(field.shape[1], dtype=torch.bool, device=field.device)
        mask[self.channels] = True
        return self.loss(torch.where(mask[None, :, None, None], field.detach(), field))


class Stack(Loss) :
    """Several constraints evaluated together, (b, c, i, j) -> (b, m)."""
    def __init__(self, losses) :
        self.losses = losses

    def __call__(self, field) :
        return torch.stack([l(field) for l in self.losses], dim=1)


def ddpm_step(sch, eps, t, x, generator, ocean, centre_noise=True) :
    """
        `DDPMScheduler.step` (1000 inference steps, fixed_small variance, x0 clipped to the clip range) where the added
        noise has its per-(member, channel) spatial mean over the ocean cells `ocean` (c, i, j) removed (only if
        `centre_noise`), and the land and padding cells of the result are replaced by fresh noise of the next noise level
        (0 after the last step).
    """
    a_t = sch.alphas_cumprod[t].to(x.device)
    a_prev = sch.alphas_cumprod[t - 1].to(x.device) if t > 0 else torch.tensor(1., device=x.device)
    beta_t = 1 - a_t / a_prev
    x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-sch.config.clip_sample_range, sch.config.clip_sample_range)
    x = a_prev.sqrt() * beta_t / (1 - a_t) * x0 + (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t) * x
    if t > 0 :
        z = torch.randn(x.shape, generator=generator, device=x.device)
        if centre_noise :
            z = z - (z * ocean).sum((-2, -1), keepdim=True) / ocean.sum((-2, -1), keepdim=True)
        x = (x + ((1 - a_prev) / (1 - a_t) * beta_t).clamp(min=1e-20).sqrt() * z) * ocean \
            + (1 - a_prev).sqrt() * torch.randn(x.shape, generator=generator, device=x.device) * (1 - ocean)
    return x


def remove_level_mean(g, w) :
    """g (b, c, i, j) minus its w-weighted spatial mean per (member, channel), zero where w (c, i, j) is 0."""
    return (g - (g * w).sum((-2, -1), keepdim=True) / w.sum((-2, -1), keepdim=True)) * (w > 0)


class OceanOnly(Loss) :
    """`loss` evaluated on the field with its land and padding cells (ocean == 0) set to 0."""
    def __init__(self, loss, ocean) :
        self.loss, self.ocean = loss, ocean

    def __call__(self, field) :
        return self.loss(field * self.ocean)


def sample(pipeline, batch_size, loss_c=None, eta=0.0, eps=0.0, projection=None, seed=0, lbd0=0.0, mode='raw', t_on=None,
           reference=None, ocean=None, grad_weights=None, centre_noise=True) :
    """
    `pipelines.sampling.sample`, with loss_c returning (b,) or (b, m) for m constraints; eta, eps, lbd0 and mode
    ('raw' g = loss - eps, 'sqrt' g = sqrt(loss) - sqrt(eps)) are scalars or sequences of length m, one dual variable
    per sample and constraint. t_on (None or sequence of length m) : a constraint is active (gradient and dual update)
    only at the timesteps t < t_on.
    reference (`Reference`, not fixed) : the unconstrained trajectory of the same seed is sampled alongside (own
    generator, same noise) and its clamped x0 estimate is put in reference.x0 before each loss evaluation.
    Returns x (b, c, i, j), lbd (b,) or (b, m), the trace of the mean lbd and loss over the steps (T, m), and the final
    unconstrained sample (or None).
    ocean (None or (c, i, j) float mask of the ocean cells, padded) : noise of the initial state and of every step
    centred over the ocean cells, land and padding cells kept at the noise level of the step (`ddpm_step`), the
    guidance step masked to the ocean and `projection` applied after the last step only. centre_noise=False keeps the
    noisy land cells but leaves the noise uncentred.
    grad_weights (None or (c, i, j)) : the guidance step has its grad_weights-weighted spatial mean per channel removed
    (`remove_level_mean`), so that it leaves the weighted mean of every level unchanged.
    """
    sch = pipeline.scheduler
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    x = torch.randn(shape, generator=generator, device=pipeline.device)
    centre = lambda z : z if ocean is None or not centre_noise else z - (z * ocean).sum((-2, -1), keepdim=True) / ocean.sum((-2, -1), keepdim=True) * ocean
    step = lambda e, t, y, g : (sch.step(e, t, y, generator=g).prev_sample if ocean is None else ddpm_step(sch, e, t, y, g, ocean, centre_noise))
    x = centre(x)
    eta = torch.as_tensor(eta, dtype=torch.float32, device=pipeline.device)
    eps = torch.as_tensor(eps, dtype=torch.float32, device=pipeline.device)
    sqrt = torch.as_tensor(np.atleast_1d(mode) == 'sqrt', device=pipeline.device)
    lbd, trace = None, []
    parallel = reference is not None and not reference.fixed
    if parallel :
        gen_ref = torch.Generator(pipeline.device).manual_seed(seed)
        x_ref = centre(torch.randn(shape, generator=gen_ref, device=pipeline.device))

    for t in pipeline.progress_bar(sch.timesteps) :
        x = x.detach().requires_grad_(loss_c is not None)
        with torch.set_grad_enabled(loss_c is not None) :
            noise_pred = pipeline.unet(x, t).sample
        if loss_c is not None :
            abar = sch.alphas_cumprod[t].to(x.device)
            if parallel :
                with torch.no_grad() :
                    noise_ref = pipeline.unet(x_ref, t).sample
                    reference.x0 = ((x_ref - (1 - abar).sqrt() * noise_ref) / abar.sqrt()).clamp(
                        -sch.config.clip_sample_range, sch.config.clip_sample_range)
            x0_hat = (x - (1 - abar).sqrt() * noise_pred) / abar.sqrt()
            x0_hat = x0_hat.clamp(-sch.config.clip_sample_range, sch.config.clip_sample_range)
            loss = loss_c(x0_hat)
            g = torch.where(sqrt, (loss + 1e-12).sqrt() - eps.sqrt(), loss - eps)
            if t_on is not None :
                g = g * torch.as_tensor([float(t < t1) for t1 in t_on], device=g.device)
            if lbd is None :
                lbd = (torch.zeros_like(g) + torch.as_tensor(lbd0).to(g)).detach()
            grad, = torch.autograd.grad((lbd * g).sum(), x)
        with torch.no_grad() :
            x = step(noise_pred, t, x, generator)
            if loss_c is not None :
                if grad_weights is not None :
                    grad = remove_level_mean(grad, grad_weights)
                x = x - (grad if ocean is None else grad * ocean)
                lbd = (lbd + eta * g).clamp(min=0)
                trace.append(torch.stack([lbd.reshape(batch_size, -1).mean(0), loss.reshape(batch_size, -1).mean(0)]).cpu())
            project = projection is not None and (ocean is None or t == 0)
            if project :
                x = projection(x)
            if parallel :
                x_ref = step(noise_ref, t, x_ref, gen_ref)
                if project :
                    x_ref = projection(x_ref)
    return x.detach(), lbd, (torch.stack(trace).numpy() if trace else None), (x_ref if parallel else None)
