"""
Constrained sampling loops built on the objects of `new_constraint.py`.

- `sample_gradient` : gradient guidance, x_{t-1} = DDPM(x_t) - beta_t * grad C(x_t)  (adaptive beta, see `GradientConstraint`)
      with `tweedie=True` the loss is evaluated on the clipped Tweedie estimate x0_hat(x_t) returned by the scheduler :
      x_{t-1} = DDPM(x_t) - beta_t * grad_x0 C(x0_hat)  (no backpropagation through the UNet)
- `sample_projection` : augmented Lagrangian with a projection on the constraint set in observation space
      x'        = DDPM(x_t)
      z         = P(g(x') + lambda / (2 rho))
      L(x)      = rho E[||g(x) - z||^2] + lambda . E[g(x) - z]
      x_{t-1}   = x' - tau * grad L(x')
      lambda    <- lambda + eta * dL/dlambda,  rho <- max(rho + eta_rho * dL/drho, rho_min)
"""
import torch
import einops
import pandas as pd


def sample_gradient(pipeline, x, constraint, projection=None, tweedie=False):
    """
        pipeline (DDPMPipeline_Tensor) : diffusion model and scheduler (timesteps already set)
        x (b, c, i, j) : initial noise
        constraint (GradientConstraint) : returns beta * grad C(x)
        projection (Projection) : applied on x after each step (ex : BorderZero)
        tweedie (bool) : evaluate the constraint on the Tweedie estimate x0_hat instead of x_t
    Returns :
        x (b, c, i, j) : generated normalised states
        metrics (pd.DataFrame) : beta along sampling, stops at the first non finite state
    """
    metrics = []
    with torch.no_grad() :
        pbar = pipeline.progress_bar(pipeline.scheduler.timesteps)
        for t in pbar:
            model_output = pipeline.unet(x, t).sample
            sch_out = pipeline.scheduler.step(model_output, t, x)
            increment_model = sch_out.prev_sample

            constraint_dt = constraint(sch_out.pred_original_sample if tweedie else x, t)
            x = increment_model - constraint_dt

            if projection is not None :
                x = projection(x)
            metric = {'t' : int(t), 'mean(beta)' : float(torch.as_tensor(constraint.beta.value).mean())}
            metrics.append(metric)
            pbar.set_postfix(metric)
            if not torch.isfinite(x).all() :
                break
    return x.detach(), pd.DataFrame(metrics)


def sample_projection(pipeline, x, g, projection, data_projection=None, tau=5e-3, eta=7e-3, eta_rho=3e-3, rho_min=0.01):
    """
        pipeline (DDPMPipeline_Tensor) : diffusion model and scheduler (timesteps already set)
        x (b, c, i, j) : initial noise
        g (callable) : observation operator, normalised state -> (b, k) or (b, k, i, j)
        projection (Projection) : projection on the constraint set in observation space
        data_projection (Projection) : applied on x after each step (ex : BorderZero)
        tau : step size of the gradient step on L
        eta, eta_rho : step sizes of lambda and rho
        rho_min : lower bound of rho
    Returns :
        x (b, c, i, j) : generated normalised states
        metrics (pd.DataFrame) : rho, lambda (mean and norm) and E[||g(x) - z||^2] along sampling
    """
    def L_r(x, z, lbd, rho) :
        diff = (g(x) - z).nan_to_num(0.0)
        E_d_sq = (diff**2).sum(axis=1).mean()
        E_d = einops.reduce(diff, 'n k ... -> k', 'mean')
        return rho*E_d_sq + lbd@E_d, {'E_d_sq' : E_d_sq.detach()}

    rho = torch.tensor(rho_min, requires_grad=True, device=x.device)
    with torch.no_grad() :
        lbd = torch.zeros((g(x).shape[1]), device=x.device)
    lbd.requires_grad_(True)

    metrics = []
    pbar = pipeline.progress_bar(pipeline.scheduler.timesteps)
    for t in pbar:
        with torch.no_grad():
            x = pipeline.scheduler.step(pipeline.unet(x, t).sample, t, x).prev_sample
            gx = g(x)
            z = projection(gx + lbd[None, :, *((gx.ndim-2)*[None])] / (2*rho))

        x.requires_grad_(True)
        l, m = L_r(x, z, lbd, rho)
        l.backward()

        with torch.no_grad():
            x = x - tau * x.grad
            if data_projection is not None :
                x = data_projection(x)
            lbd += eta*lbd.grad
            rho += eta_rho*rho.grad
            rho.clamp_(min=rho_min)
        lbd.grad, rho.grad = None, None
        x = x.detach()
        metric = {'t' : int(t), 'rho' : rho.item(), 'mean(lbd)' : lbd.mean().item(), 'norm(lbd)' : lbd.norm().item(), 'E[||g(x) - z||^2]' : m['E_d_sq'].item()}
        metrics.append(metric)
        pbar.set_postfix(metric)
    return x.detach(), pd.DataFrame(metrics)
