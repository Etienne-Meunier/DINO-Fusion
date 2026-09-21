"""Torch dataset over the converted Veros ACC file (analogue of DINO's DataLoader.py, in-memory instead of webdataset)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from extract_data import choose_holdout
from transforms import FieldTransform


def resolve_split(cfg, ds):
    """(train_runs, holdout_runs) from the config; "file" uses the split stored in the dataset."""
    run_ck, run_eps = np.asarray(ds["run_ck"]), np.asarray(ds["run_eps"])
    if cfg.split_mode == "file":
        hold = np.asarray(ds["holdout_runs"], dtype=np.int64)
    else:
        hold = choose_holdout(run_ck, run_eps, cfg.split_mode, cfg.n_holdout, cfg.split_seed, list(cfg.split_rows))
    return np.setdiff1d(np.arange(len(run_ck)), hold), np.asarray(hold, dtype=np.int64)


class CondEncoder:
    """(ck, eps) -> standardised conditioning vector, using the statistics stored in the dataset file."""

    def __init__(self, ds):
        self.keys = [str(k) for k in ds["cond_keys"]]
        self.mean = torch.as_tensor(np.asarray(ds["cond_mean"]), dtype=torch.float32)
        self.std = torch.as_tensor(np.asarray(ds["cond_std"]), dtype=torch.float32)

    def __call__(self, ck, eps) -> torch.Tensor:
        ck = torch.as_tensor(np.asarray(ck, dtype=np.float64)).reshape(-1)
        eps = torch.as_tensor(np.asarray(eps, dtype=np.float64)).reshape(-1)
        raw = {"log_ck": torch.log(ck), "log_eps": torch.log(eps)}
        v = torch.stack([raw[k] for k in self.keys], dim=1).float()
        return (v - self.mean) / self.std

    @property
    def dim(self) -> int:
        return len(self.keys)


class VerosTSDataset(Dataset):
    """Returns ``(x, cond)``: x is the normalised, land-zeroed, padded (C, Yp, Xp) tensor; cond is (n_cond,)."""

    def __init__(self, data_file: str, split: str, transform: FieldTransform, fields=("temp", "salt"),
                 snapshot_stride: int = 1, holdout_runs=None):
        """``holdout_runs``: run ids held out (from :func:`resolve_split`); None uses the dataset's stored split."""
        assert split in ("train", "holdout", "all")
        ds = np.load(data_file, allow_pickle=False)
        run_id = np.asarray(ds["run_id"])
        holdout = np.asarray(ds["holdout_runs"]) if holdout_runs is None else np.asarray(holdout_runs)
        if split == "all":
            idx = np.arange(len(run_id))
        else:
            in_hold = np.isin(run_id, holdout)
            idx = np.where(in_hold if split == "holdout" else ~in_hold)[0]
        if snapshot_stride > 1 and len(idx):
            idx = np.concatenate([idx[run_id[idx] == r][::snapshot_stride] for r in np.unique(run_id[idx])])
        self.idx = idx
        self.fields = {f: torch.from_numpy(np.ascontiguousarray(ds[f][idx])) for f in fields}   # float32, in memory
        self.run_id = run_id[idx]
        self.ck = np.asarray(ds["ck"])[idx]
        self.eps = np.asarray(ds["eps"])[idx]
        self.year = np.asarray(ds["year"])[idx]
        self.run_names = [str(r) for r in ds["run_names"]]
        self.encoder = CondEncoder(ds)
        self.cond = self.encoder(self.ck, self.eps)
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        x = self.transform.normalise({f: v[i] for f, v in self.fields.items()})
        return x.cpu(), self.cond[i]

    def runs(self) -> list[int]:
        return sorted(set(self.run_id.tolist()))

    def run_condition(self, r: int) -> tuple[float, float]:
        j = np.where(self.run_id == r)[0][0]
        return float(self.ck[j]), float(self.eps[j])

    def run_fields(self, r: int) -> dict[str, torch.Tensor]:
        """All snapshots of run r as raw (un-normalised) fields: {field: (n, Z, Y, X)}."""
        sel = torch.from_numpy(np.where(self.run_id == r)[0])
        return {f: v[sel] for f, v in self.fields.items()}


def build_transform(data_file: str, cfg, device="cpu") -> FieldTransform:
    ds = np.load(data_file, allow_pickle=False)
    return FieldTransform.from_dataset(ds, tuple(cfg.fields), cfg.norm_mode, cfg.std_floor,
                                       tuple(cfg.paddings), device=device)
