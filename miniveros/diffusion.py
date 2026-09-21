"""DDPM training objective (fork of DINO-Fusion DiffusionModel.training_step, with conditioning) and the sampler's
scheduler, with an optional per-channel clip of the predicted clean state."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


class PerLevelClipDDPMScheduler(DDPMScheduler):
    """DDPMScheduler whose clipping of the predicted clean state is per channel: [lo_c, hi_c] instead of a scalar.
    Uses the scheduler's thresholding hook (``thresholding=True`` routes the clean-state estimate through
    ``_threshold_sample`` instead of the scalar clamp)."""

    def set_clip_bounds(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        self._lo = lo.reshape(1, -1, 1, 1)
        self._hi = hi.reshape(1, -1, 1, 1)

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(sample, self._hi.to(sample)), self._lo.to(sample))


class Diffusion:
    def __init__(self, cfg, clip_bounds=None):
        """``clip_bounds``: (lo, hi) per-channel tensors from the transform; needed only for sampling in per_level mode."""
        self.cfg = cfg
        if cfg.clip_mode == "per_level" and clip_bounds is not None:
            self.scheduler = PerLevelClipDDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                       beta_schedule=cfg.beta_schedule, clip_sample=False, thresholding=True)
            self.scheduler.set_clip_bounds(*clip_bounds)
        else:
            self.scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                           beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                                           clip_sample_range=cfg.clip_sample_range)

    def training_loss(self, model, x0: torch.Tensor, cond: torch.Tensor,
                      zero_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Epsilon-prediction MSE. ``zero_mask`` (C, H, W) excludes land+padding cells when cfg.mask_loss."""
        bs = x0.shape[0]
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=x0.device, dtype=torch.long)
        xt = self.scheduler.add_noise(x0, noise, t)
        drop = None
        if self.cfg.cond_drop_prob > 0:
            drop = torch.rand(bs, device=x0.device) < self.cfg.cond_drop_prob
        pred = model(xt, t, cond, drop)
        if zero_mask is None or not self.cfg.mask_loss:
            return F.mse_loss(pred, noise)
        w = (~zero_mask).to(pred.dtype)
        return ((pred - noise) ** 2 * w).sum() / (w.sum() * bs)
