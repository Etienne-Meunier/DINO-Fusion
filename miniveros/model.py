"""Conditional UNet: DINO-Fusion's ``get_simple_unet`` plus a parameter-conditioning path.

The two standardised log-parameters go through a small MLP whose output has the width of the
UNet time embedding; diffusers adds it to the timestep embedding (``class_embed_type="identity"``).
A learned null embedding is used when a condition is dropped (classifier-free guidance).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int, sample_size: tuple[int, int], cond_dim: int,
                 block_out_channels=(64, 64, 128, 128), layers_per_block: int = 2, cond_hidden: int = 256):
        super().__init__()
        self.kwargs = dict(in_channels=in_channels, sample_size=tuple(int(s) for s in sample_size), cond_dim=cond_dim,
                           block_out_channels=tuple(int(c) for c in block_out_channels),
                           layers_per_block=layers_per_block, cond_hidden=cond_hidden)
        n = len(block_out_channels)
        self.unet = UNet2DModel(
            sample_size=self.kwargs["sample_size"],
            in_channels=in_channels, out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=self.kwargs["block_out_channels"],
            down_block_types=("DownBlock2D",) * n,
            up_block_types=("UpBlock2D",) * n,
            class_embed_type="identity",
        )
        temb = self.unet.time_embedding.linear_2.out_features
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, temb))
        self.null_cond = nn.Parameter(torch.zeros(temb))

    @property
    def in_channels(self) -> int:
        return self.kwargs["in_channels"]

    @property
    def sample_size(self) -> tuple[int, int]:
        return self.kwargs["sample_size"]

    def embed(self, cond: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.cond_mlp(cond)
        if drop_mask is not None:
            emb = torch.where(drop_mask[:, None], self.null_cond[None].expand_as(emb), emb)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.unet(x, t, class_labels=self.embed(cond, drop_mask), return_dict=False)[0]

    # ---- persistence (plain torch files; avoids depending on diffusers' pipeline save format)
    def save(self, path) -> None:
        torch.save({"kwargs": self.kwargs, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "ConditionalUNet":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(**ck["kwargs"])
        m.load_state_dict(ck["state_dict"])
        return m

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
