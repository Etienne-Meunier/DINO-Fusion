"""Field <-> network-tensor transforms (fork of DINO-Fusion TransformationPipeline, without tensordict).

Forward  (``normalise``):   {field: (..., Z, Y, X)}  ->  concat channels -> normalise -> zero land -> pad
Backward (``denormalise``): (..., C, Yp, Xp) -> unpad -> land to NaN -> un-normalise -> split fields

Everything is batch-agnostic: leading dimensions are preserved.

Normalisation (``"<k>-std"``): per vertical level, ``(x - mean_z) / (k * std_z)``, as in DINO-Fusion.
The std is floored (``std_floor``) so constant fields (salinity == 35) map to exactly 0.
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F


class Concatener:
    """Concatenate a dict of fields along a new channel axis (Z of each field becomes channels)."""

    def __init__(self, field_levels: dict[str, int]):
        self.field_levels = dict(field_levels)          # ordered {field: n_levels}

    def __call__(self, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name, nz in self.field_levels.items():
            f = fields[name]
            assert f.shape[-3] == nz, f"{name}: expected {nz} levels, got {f.shape[-3]}"
            parts.append(f)
        return torch.cat(parts, dim=-3)

    def uncall(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        chunks = torch.split(x, list(self.field_levels.values()), dim=-3)
        return {name: c for name, c in zip(self.field_levels, chunks)}


class Normaliser:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, k: float, std_floor: float):
        """mean/std: (C, 1, 1), one value per channel (= per field and level)."""
        self.mean = mean
        self.std = std.clamp_min(std_floor)
        self.k = k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.k * self.std)

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.k * self.std) + self.mean


class Masker:
    """Set land cells to ``val_mask`` (forward) and back to ``val_unmask`` (backward)."""

    def __init__(self, mask: torch.Tensor, val_mask: float = 0.0, val_unmask: float = float("nan")):
        self.mask = mask                                 # (C, Y, X) bool, True on land
        self.val_mask, self.val_unmask = val_mask, val_unmask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(self.mask, self.val_mask)

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(self.mask, self.val_unmask)


class Padder:
    """Zero-pad the last two axes so the UNet can halve the grid several times."""

    def __init__(self, paddings: tuple[int, int, int, int] = (1, 1, 3, 3), value: float = 0.0):
        self.paddings = tuple(int(p) for p in paddings)   # (x_left, x_right, y_low, y_high)
        self.value = value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.paddings, mode="constant", value=self.value)

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        xl, xr, yl, yh = self.paddings
        return x[..., yl:x.shape[-2] - yh, xl:x.shape[-1] - xr]


class FieldTransform:
    """The full pipeline. Build it with :func:`FieldTransform.from_dataset`."""

    def __init__(self, field_levels: dict[str, int], mean: torch.Tensor, std: torch.Tensor,
                 land_mask: torch.Tensor, k: float, std_floor: float,
                 paddings: tuple[int, int, int, int], device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.concatener = Concatener(field_levels)
        self.normaliser = Normaliser(mean.to(self.device), std.to(self.device), k, std_floor)
        self.masker = Masker(land_mask.to(self.device))
        self.padder = Padder(paddings)
        self.n_channels = int(sum(field_levels.values()))
        self.clip_lo = None   # (C,) normalised per-channel clip bounds, set by from_dataset
        self.clip_hi = None
        self.field_shape = tuple(land_mask.shape[-2:])                          # (Y, X)
        probe = self.padder(torch.zeros(self.n_channels, *self.field_shape))
        self.padded_shape = tuple(probe.shape[-2:])                              # (Yp, Xp)
        # cells that must be exactly zero in network space: land + padding
        self.zero_mask = self.padder(self.masker.mask.float().cpu()).bool().to(self.device) | \
                         (self.padder(torch.ones(self.n_channels, *self.field_shape)) == 0).to(self.device)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dataset(cls, ds: dict | np.lib.npyio.NpzFile, fields: tuple[str, ...], norm_mode: str,
                     std_floor: float, paddings: tuple[int, int, int, int],
                     device: str | torch.device = "cpu", clip_margin: float = 0.05,
                     clip_min_halfwidth: float = 0.02) -> "FieldTransform":
        """``ds`` is the converted-dataset npz (see extract_data.py); stats were computed on the train split."""
        stats_fields = [str(f) for f in ds["stats_fields"]]
        idx = [stats_fields.index(f) for f in fields]
        land = torch.as_tensor(np.asarray(ds["mask_land"]))                     # (Z, Y, X)
        nz = land.shape[0]
        field_levels = {f: nz for f in fields}
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-std", norm_mode)
        if not m:
            raise ValueError(f"norm_mode must be '<k>-std' (e.g. '3-std'), got {norm_mode!r}")
        mean = torch.as_tensor(np.asarray(ds["lvl_mean"])[idx]).reshape(-1, 1, 1)                   # (C, 1, 1)
        std = torch.as_tensor(np.asarray(ds["lvl_std"])[idx]).reshape(-1, 1, 1)
        k = float(m.group(1))
        land_c = land.repeat(len(fields), 1, 1)                                                         # (C, Y, X)
        tr = cls(field_levels, mean.float(), std.float(), land_c, k, std_floor, paddings, device)
        tr.set_clip_bounds(ds, fields, land, clip_margin, clip_min_halfwidth)
        return tr

    def set_clip_bounds(self, ds, fields, land, margin: float, min_halfwidth: float) -> None:
        """Per-channel clip bounds in normalised units from the observed data range of each field and level
        (``lvl_min``/``lvl_max`` in the dataset if present, else computed over water cells), widened by
        ``margin`` x range and at least ``min_halfwidth`` wide on each side of the range's centre."""
        keys = set(getattr(ds, "files", ds.keys()))
        stats_fields = [str(f) for f in ds["stats_fields"]]
        water = ~np.asarray(land)
        mins, maxs = [], []
        for f in fields:
            if "lvl_min" in keys and "lvl_max" in keys:
                i = stats_fields.index(f); mn, mx = np.asarray(ds["lvl_min"])[i], np.asarray(ds["lvl_max"])[i]
            else:
                arr = np.asarray(ds[f])                                                         # (N, Z, Y, X)
                mn = np.array([arr[:, z][:, water[z]].min() for z in range(arr.shape[1])])
                mx = np.array([arr[:, z][:, water[z]].max() for z in range(arr.shape[1])])
            mins.append(mn); maxs.append(mx)
        mn = torch.as_tensor(np.concatenate(mins), dtype=torch.float32).to(self.device)
        mx = torch.as_tensor(np.concatenate(maxs), dtype=torch.float32).to(self.device)
        mean, std = self.normaliser.mean.reshape(-1), self.normaliser.std.reshape(-1)
        lo = (mn - mean) / (self.normaliser.k * std); hi = (mx - mean) / (self.normaliser.k * std)
        rng = hi - lo; c = 0.5 * (hi + lo); half = torch.maximum(0.5 * rng * (1 + 2 * margin), torch.full_like(rng, min_halfwidth))
        self.clip_lo, self.clip_hi = c - half, c + half

    # ------------------------------------------------------------------ forward / backward
    def normalise(self, fields: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.concatener({k: v.to(self.device) for k, v in fields.items()})
        x = self.normaliser(x)
        x = self.masker(x)
        return self.padder(x)

    def denormalise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.padder.uncall(x.to(self.device))
        x = self.masker.uncall(x)
        x = self.normaliser.uncall(x)
        return self.concatener.uncall(x)

    __call__ = normalise
