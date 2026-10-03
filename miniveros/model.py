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
                 block_out_channels=(64, 64, 128, 128), layers_per_block: int = 2, cond_hidden: int = 256,
                 mask_input: bool = False, act_penalty: bool = False, in_mask: torch.Tensor | None = None):
        """``mask_input``: a fixed (H, W) mask (``in_mask``, True on land and padding) is appended to the input as a
        channel; it is stored as a buffer, so ``load`` restores it. ``act_penalty``: the decoder resnet outputs are
        recorded at every forward, for :meth:`activation_ratio`."""
        super().__init__()
        self.kwargs = dict(in_channels=in_channels, sample_size=tuple(int(s) for s in sample_size), cond_dim=cond_dim,
                           block_out_channels=tuple(int(c) for c in block_out_channels),
                           layers_per_block=layers_per_block, cond_hidden=cond_hidden,
                           mask_input=bool(mask_input), act_penalty=bool(act_penalty))
        n = len(block_out_channels)
        self.unet = UNet2DModel(
            sample_size=self.kwargs["sample_size"],
            in_channels=in_channels + int(mask_input), out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=self.kwargs["block_out_channels"],
            down_block_types=("DownBlock2D",) * n,
            up_block_types=("UpBlock2D",) * n,
            class_embed_type="identity",
        )
        temb = self.unet.time_embedding.linear_2.out_features
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, cond_hidden), nn.SiLU(), nn.Linear(cond_hidden, temb))
        self.null_cond = nn.Parameter(torch.zeros(temb))
        if mask_input:
            m = torch.zeros(1, 1, *self.kwargs["sample_size"]) if in_mask is None else in_mask.reshape(1, 1, *in_mask.shape[-2:]).float()
            self.register_buffer("in_mask", m)
        self._act_maps: list[torch.Tensor] = []
        if act_penalty:
            for b in self.unet.up_blocks:
                for rn in b.resnets:
                    rn.register_forward_hook(lambda mod, inp, out: self._act_maps.append(out.float().pow(2).mean(1).sqrt()))

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
        self._act_maps = []
        if self.kwargs["mask_input"]:
            x = torch.cat([x, self.in_mask.expand(x.shape[0], 1, -1, -1).to(x.dtype)], dim=1)
        return self.unet(x, t, class_labels=self.embed(cond, drop_mask), return_dict=False)[0]

    def activation_ratio(self) -> torch.Tensor:
        """Mean over the decoder resnets and the batch of log(max / mean over positions) of the per-position RMS
        activation, from the last forward (1 for a flat map; the spike of a defective decoder gives 2--3)."""
        if not self._act_maps:
            return torch.zeros((), device=self.null_cond.device)
        r = [(m.flatten(1).amax(1) / (m.flatten(1).mean(1) + 1e-6)).log().mean() for m in self._act_maps]
        return torch.stack(r).mean()

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
