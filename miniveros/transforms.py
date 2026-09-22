"""Field <-> network-tensor transforms (fork of DINO-Fusion TransformationPipeline, without tensordict).

Forward  (``normalise``):   {field: (..., Z, Y, X)}  ->  concat channels -> normalise -> fill land -> pad
Backward (``denormalise``): (..., C, Yp, Xp) -> unpad -> land to NaN -> un-normalise -> split fields

Everything is batch-agnostic: leading dimensions are preserved.

Normalisation is an affine map per vertical level (= per channel), ``x' = (x - offset_z) / scale_z``:
  ``"<k>-std"``: offset = mean_z, scale = k * max(std_z, std_floor)                       (as in DINO-Fusion)
  ``"minmax"``:  offset = (min_z + max_z) / 2, scale = max((max_z - min_z) / 2, std_floor): the level's data range -> [-1, 1]
The floor makes constant fields (salinity == 35) map to exactly 0.
Land and padding cells carry the normalised value of the level mean, the ``fill``: 0 in the std modes,
(mean_z - offset_z) / scale_z in minmax, so an inactive cell always reads as "the mean of that level".
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
    def __init__(self, offset: torch.Tensor, scale: torch.Tensor):
        """offset/scale: (C, 1, 1), one value per channel (= per field and level)."""
        self.offset, self.scale = offset, scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset


class Masker:
    """Set land cells to the per-channel ``fill`` (forward) and to NaN (backward)."""

    def __init__(self, mask: torch.Tensor, fill: torch.Tensor, val_unmask: float = float("nan")):
        self.mask = mask                                 # (C, Y, X) bool, True on land
        self.fill = fill                                 # (C, 1, 1)
        self.val_unmask = val_unmask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.mask, self.fill.to(x.dtype), x)

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(self.mask, self.val_unmask)


class Padder:
    """Pad the last two axes with the per-channel ``fill`` so the UNet can halve the grid several times."""

    def __init__(self, paddings: tuple[int, int, int, int], fill: torch.Tensor):
        self.paddings = tuple(int(p) for p in paddings)   # (x_left, x_right, y_low, y_high)
        self.fill = fill                                  # (C, 1, 1)

    def pad_mask(self, field_shape: tuple[int, int]) -> torch.Tensor:
        """(Yp, Xp) bool, True on the padding cells."""
        return F.pad(torch.ones(*field_shape), self.paddings, mode="constant", value=0.0) == 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, self.paddings, mode="constant", value=0.0)
        return torch.where(self.pad_mask(x.shape[-2:]).to(x.device), self.fill.to(x.dtype), xp)

    def uncall(self, x: torch.Tensor) -> torch.Tensor:
        xl, xr, yl, yh = self.paddings
        return x[..., yl:x.shape[-2] - yh, xl:x.shape[-1] - xr]


def level_range(ds, fields: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Per-level (min, max) over water cells for each field, (C,) each. Read from the data file when stored
    (``lvl_min``/``lvl_max``, extract_data.py), computed from the fields otherwise (older files)."""
    stats_fields = [str(f) for f in ds["stats_fields"]]
    keys = ds.files if hasattr(ds, "files") else list(ds.keys())
    if "lvl_min" in keys and "lvl_max" in keys:
        idx = [stats_fields.index(f) for f in fields]
        return np.asarray(ds["lvl_min"])[idx].reshape(-1), np.asarray(ds["lvl_max"])[idx].reshape(-1)
    water = ~np.asarray(ds["mask_land"]).astype(bool)                       # (Z, Y, X)
    sel = None                                                               # same samples as the stored mean/std
    if "meta" in keys and "train_runs" in keys:
        import json
        if json.loads(str(ds["meta"])).get("stats_on") == "train":
            sel = np.isin(np.asarray(ds["run_id"]), np.asarray(ds["train_runs"]))
    lo, hi = [], []
    for f in fields:
        arr = ds[f]                                                          # an npz member loads whole; chunking bounds the temporaries only
        l = np.full(water.shape[0], np.inf); h = np.full(water.shape[0], -np.inf)
        for c in range(0, arr.shape[0], 2000):
            blk = np.asarray(arr[c:c + 2000])
            if sel is not None:
                blk = blk[sel[c:c + 2000]]
                if blk.shape[0] == 0:
                    continue
            bmin, bmax = blk.min(0), blk.max(0)
            for z in range(water.shape[0]):
                if water[z].any():
                    l[z] = min(l[z], bmin[z][water[z]].min()); h[z] = max(h[z], bmax[z][water[z]].max())
        lo.append(l); hi.append(h)
    return np.concatenate(lo), np.concatenate(hi)


class FieldTransform:
    """The full pipeline. Build it with :func:`FieldTransform.from_dataset`."""

    def __init__(self, field_levels: dict[str, int], offset: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor,
                 std: torch.Tensor, land_mask: torch.Tensor, paddings: tuple[int, int, int, int],
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        offset, scale, mean = (t.to(self.device).float() for t in (offset, scale, mean))
        self.concatener = Concatener(field_levels)
        self.normaliser = Normaliser(offset, scale)
        self.mean, self.std = mean, std.to(self.device).float()                  # (C, 1, 1) physical units, std floored
        self.fill = (mean - offset) / scale                                      # (C, 1, 1): the level mean, normalised
        self.masker = Masker(land_mask.to(self.device), self.fill)
        self.padder = Padder(paddings, self.fill)
        self.n_channels = int(sum(field_levels.values()))
        self.field_shape = tuple(land_mask.shape[-2:])                          # (Y, X)
        pad = self.padder.pad_mask(self.field_shape)                             # (Yp, Xp)
        self.padded_shape = tuple(pad.shape)
        # inactive cells in network space (land + padding): they hold ``fill``
        land_p = F.pad(self.masker.mask.float().cpu(), self.padder.paddings, mode="constant", value=0.0).bool()
        self.fill_mask = (land_p | pad).to(self.device)                           # (C, Yp, Xp)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dataset(cls, ds, fields: tuple[str, ...], norm_mode: str, std_floor: float,
                     paddings: tuple[int, int, int, int], device: str | torch.device = "cpu") -> "FieldTransform":
        """``ds`` is the converted-dataset npz (see extract_data.py) with the per-level statistics."""
        stats_fields = [str(f) for f in ds["stats_fields"]]
        idx = [stats_fields.index(f) for f in fields]
        land = torch.as_tensor(np.asarray(ds["mask_land"]))                     # (Z, Y, X)
        nz = land.shape[0]
        field_levels = {f: nz for f in fields}
        mean = torch.as_tensor(np.asarray(ds["lvl_mean"])[idx]).reshape(-1, 1, 1).double()        # (C, 1, 1)
        std = torch.as_tensor(np.asarray(ds["lvl_std"])[idx]).reshape(-1, 1, 1).double().clamp_min(std_floor)
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-std", norm_mode)
        if m:
            offset, scale = mean, float(m.group(1)) * std
        elif norm_mode == "minmax":
            lo, hi = level_range(ds, fields)
            lo = torch.as_tensor(lo).reshape(-1, 1, 1).double(); hi = torch.as_tensor(hi).reshape(-1, 1, 1).double()
            offset, scale = (lo + hi) / 2, ((hi - lo) / 2).clamp_min(std_floor)
        else:
            raise ValueError(f"norm_mode must be '<k>-std' (e.g. '3-std') or 'minmax', got {norm_mode!r}")
        land_c = land.repeat(len(fields), 1, 1)                                                         # (C, Y, X)
        return cls(field_levels, offset, scale, mean, std, land_c, paddings, device)

    def clip_bounds(self, clip_ref: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-channel (lo, hi) in normalised units for ``clip_ref`` '<k>-std' (mean_z +- k std_z in physical units);
        None for 'norm' (the scheduler's scalar clip). Under '<k>-std' normalisation with the same k this is exactly +-1."""
        if clip_ref == "norm":
            return None
        m = re.fullmatch(r"(\d+(?:\.\d+)?)-std", clip_ref)
        if not m:
            raise ValueError(f"clip_ref must be 'norm' or '<k>-std', got {clip_ref!r}")
        k = float(m.group(1))
        lo = self.normaliser(self.mean - k * self.std).reshape(-1); hi = self.normaliser(self.mean + k * self.std).reshape(-1)
        return lo, hi

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
