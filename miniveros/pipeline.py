"""Conditional DDPM sampler (fork of DINO-Fusion pipelines/pipeline_tensor.py) with the constraints hook."""
from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor


class LandFill:
    """Re-impose the fill value (the normalised level mean, 0 in the std modes) on land and padding cells after
    every denoising step (DINO's BorderZeroConstraint)."""

    def __init__(self, fill_mask: torch.Tensor, fill: torch.Tensor):
        self.mask = fill_mask                                  # (C, H, W) bool
        self.fill = fill                                       # (C, 1, 1)

    def __call__(self, x: torch.Tensor, t) -> torch.Tensor:
        return torch.where(self.mask, self.fill.to(x), x)

    def __str__(self):
        return f"LandFill({int(self.mask.sum())} cells)"


@torch.no_grad()
def sample(model, scheduler, cond: torch.Tensor, num_inference_steps: int,
           generator: torch.Generator | None = None, constraints=(), guidance_scale: float = 1.0,
           progress: bool = False) -> torch.Tensor:
    """Draw one state per row of ``cond`` (n, cond_dim). Returns normalised, padded tensors (n, C, Yp, Xp)."""
    n = cond.shape[0]
    shape = (n, model.in_channels, *model.sample_size)
    x = randn_tensor(shape, generator=generator, device=cond.device)
    scheduler.set_timesteps(num_inference_steps)
    steps = scheduler.timesteps
    if progress:
        from tqdm.auto import tqdm
        steps = tqdm(steps, leave=False)
    all_drop = torch.ones(n, dtype=torch.bool, device=cond.device)
    for t in steps:
        eps = model(x, t, cond)
        if guidance_scale != 1.0:
            eps_u = model(x, t, cond, all_drop)
            eps = eps_u + guidance_scale * (eps - eps_u)
        x = scheduler.step(eps, t, x, generator=generator).prev_sample
        for c in constraints:
            x = c(x, t)
    return x
