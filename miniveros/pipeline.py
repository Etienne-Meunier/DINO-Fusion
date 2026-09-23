"""Conditional DDPM sampler (fork of DINO-Fusion pipelines/pipeline_tensor.py) with the constraints hook."""
from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor


class LandZero:
    """Re-impose the land and padding cells after every denoising step. ``mode="noised"`` (default): zeros at the
    noise level of the state just produced, sqrt(1 - abar_prev) z, as those cells looked in training, and exact
    zeros at the last step. ``mode="clean"``: exact zeros after every step (DINO's BorderZeroConstraint), which the
    network never saw at high noise levels and which cost 0.03-0.09 K of hold-out RMSE."""

    def __init__(self, zero_mask: torch.Tensor, mode: str = "noised", scheduler=None,
                 generator: torch.Generator | None = None):
        self.mask = zero_mask                                  # (C, H, W) bool
        self.mode, self.scheduler, self.generator = mode, scheduler, generator
        if mode not in ("clean", "noised"):
            raise ValueError(f"fill mode must be 'clean' or 'noised', got {mode!r}")
        if mode == "noised" and scheduler is None:
            raise ValueError("the noised mode needs the scheduler")

    def __call__(self, x: torch.Tensor, t) -> torch.Tensor:
        if self.mode == "clean":
            return x.masked_fill(self.mask, 0.0)
        s = self.scheduler
        prev_t = int(t) - s.config.num_train_timesteps // s.num_inference_steps   # x is x_{prev_t}
        if prev_t < 0:
            return x.masked_fill(self.mask, 0.0)
        ab = s.alphas_cumprod[prev_t].to(x)
        z = randn_tensor(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return torch.where(self.mask, (1 - ab).sqrt() * z, x)

    def __str__(self):
        return f"LandZero({int(self.mask.sum())} cells, {self.mode})"


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
