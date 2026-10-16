import torch


def sample(pipeline, batch_size, loss_c=None, eta=0.0, eps=0.0, mode='raw', projection=None, seed=0, lbd0=0.0):
    """
    DDPM sampling with an optional per-sample primal-dual constraint (PD-LMC, Chamon et al. 2025) evaluated on the
    Tweedie estimate clipped to the scheduler range, with gradient through the UNet (DPS, Chung et al. 2023) :
        pipeline (DDPMPipeline_Tensor) : diffusion model and scheduler (timesteps already set)
        batch_size (int) : number of generated states
        loss_c (callable) : normalised state (b, c, i, j) -> (b,), None for unconstrained sampling
        eta : step size of the dual variable
        eps : tolerance of the constraint loss_c <= eps
        mode : constraint function, 'raw' g = loss_c - eps or 'sqrt' g = sqrt(loss_c) - sqrt(eps)
        projection (Projection) : applied on x after each step (ex : BorderZero)
        seed (int) : seed of the initial noise and of the DDPM noise
        lbd0 : initial value of the dual variables
    Returns :
        x (b, c, i, j) : generated normalised states
        lbd (b,) : final dual variables, None if loss_c is None
    """
    sch = pipeline.scheduler
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    x = torch.randn(shape, generator=generator, device=pipeline.device)
    lbd = None if loss_c is None else torch.full((batch_size,), float(lbd0), device=pipeline.device)
    
    for t in pipeline.progress_bar(sch.timesteps):
        x = x.detach().requires_grad_(loss_c is not None)
        with torch.set_grad_enabled(loss_c is not None):
            noise_pred = pipeline.unet(x, t).sample
        if loss_c is not None:
            abar = sch.alphas_cumprod[t].to(x.device)
            x0_hat = (x - (1 - abar).sqrt() * noise_pred) / abar.sqrt() # Tweedy
            x0_hat = x0_hat.clamp(-sch.config.clip_sample_range, sch.config.clip_sample_range)
            loss = loss_c(x0_hat)
            # sqrt : bounded gradient near the constraint, the large early loss does not inflate lbd
            g = loss - eps if mode == 'raw' else (loss + 1e-12).sqrt() - eps**0.5
            grad, = torch.autograd.grad((lbd * g).sum(), x)
        with torch.no_grad():
            x = sch.step(noise_pred, t, x, generator=generator).prev_sample
            if loss_c is not None:
                x = x - grad
                lbd = (lbd + eta * g).clamp(min=0)
            if projection is not None:
                x = projection(x)
    return x.detach(), lbd
