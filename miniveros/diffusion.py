"""DDPM training objective (fork of DINO-Fusion DiffusionModel.training_step, with conditioning)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


class Diffusion:
    def __init__(self, cfg):
        self.cfg = cfg
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
