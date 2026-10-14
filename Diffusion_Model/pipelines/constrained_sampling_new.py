import torch


def sample(pipeline, batch_size, loss_c=None, eta=0.0, eps=0.0, projection=None, seed=0):
    """
    DDPM sampling with an optional per-sample primal-dual constraint (PD-LMC, Chamon et al. 2025) evaluated on the
    Tweedie estimate with gradient through the UNet (DPS, Chung et al. 2023) :
        pipeline (DDPMPipeline_Tensor) : diffusion model and scheduler (timesteps already set)
        batch_size (int) : number of generated states
        loss_c (callable) : normalised state (b, c, i, j) -> (b,), None for unconstrained sampling
        eta : step size of the dual variable
        eps : tolerance of the constraint loss_c <= eps
        projection (Projection) : applied on x after each step (ex : BorderZero)
        seed (int) : seed of the initial noise and of the DDPM noise
    Returns :
        x (b, c, i, j) : generated normalised states
        lbd (b,) : final dual variables, None if loss_c is None
    """
    sch = pipeline.scheduler
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    shape = (batch_size, pipeline.unet.config.in_channels, *pipeline.unet.config.sample_size)
    x = torch.randn(shape, generator=generator, device=pipeline.device)
    lbd = None if loss_c is None else torch.zeros(batch_size, device=pipeline.device)

    for t in pipeline.progress_bar(sch.timesteps):
        x = x.detach().requires_grad_(loss_c is not None)
        with torch.set_grad_enabled(loss_c is not None):
            noise_pred = pipeline.unet(x, t).sample
        if loss_c is not None:
            abar = sch.alphas_cumprod[t].to(x.device)
            x0_hat = (x - (1 - abar).sqrt() * noise_pred) / abar.sqrt()
            g = loss_c(x0_hat) - eps
            grad, = torch.autograd.grad((lbd * g).sum(), x)
        with torch.no_grad():
            x = sch.step(noise_pred, t, x, generator=generator).prev_sample
            if loss_c is not None:
                x = x - grad
                lbd = (lbd + eta * g).clamp(min=0)
            if projection is not None:
                x = projection(x)
    return x.detach(), lbd
