"""Configuration for the miniveros pipeline.

Nothing here reads environment variables at import time (unlike the DINO config).
Paths and hyper-parameters come from the dataclass defaults, a preset, or
``--set key=value`` overrides on the command line.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_origin


@dataclass
class Config:
    # ---- paths (relative to the current working dir unless absolute)
    data_file: str = "data/veros_acc_TS.npz"     # converted dataset written by extract_data.py
    run_dir: str = "runs/dev"                     # checkpoints, logs, samples, figures

    # ---- fields and grid
    fields: tuple[str, ...] = ("temp", "salt")    # channel order: all temp levels, then all salt levels
    paddings: tuple[int, int, int, int] = (1, 1, 3, 3)  # zeros added (x_left, x_right, y_low, y_high): 42x30 -> 48x32

    # ---- normalisation
    norm_mode: str = "3-std"     # "<k>-std": per vertical level, (x - mean_z) / (k * std_z), as in DINO-Fusion
                                 # "minmax": per level, the data range [min_z, max_z] -> [-1, 1]; land and padding hold the
                                 # normalised level mean in both modes (0 for "<k>-std")
    std_floor: float = 0.05      # std is floored before dividing: constant fields (salinity) would otherwise divide by ~0

    # ---- conditioning
    cond_keys: tuple[str, ...] = ("log_ck", "log_eps")
    cond_drop_prob: float = 0.0  # > 0 trains a null-condition embedding (classifier-free guidance)

    # ---- model (same UNet family as DINO's get_simple_unet)
    block_out_channels: tuple[int, ...] = (64, 64, 128, 128)
    layers_per_block: int = 2
    cond_hidden: int = 256

    # ---- diffusion
    num_train_timesteps: int = 1000
    num_inference_steps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    clip_sample_range: float = 1.0   # sampling only. Clipping the predicted clean state at 3 sigma regularises the chain:
                                     # range 3 doubled the hold-out RMSE (0.15 -> 0.30 K) although the data barely exceed 1
    mask_loss: bool = False      # True: land and padding cells are excluded from the MSE

    # ---- hold-out split (statistics are fixed on all runs, so the split is a training-time choice)
    split_mode: str = "interior_random"   # "interior_random" | "rows" | "row_ck_max" | "none" | "file" (use the dataset's stored split)
    split_rows: tuple[float, ...] = ()    # with "rows": c_k values of the held-out rows
    split_seed: int = 0
    n_holdout: int = 10                   # with "interior_random"

    # ---- training
    batch_size: int = 32
    lr: float = 1e-4
    lr_warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    ckpt_every: int = 500
    log_every: int = 20
    snapshot_stride: int = 1     # use every k-th snapshot of each training run
    num_workers: int = 4
    seed: int = 0

    # ---- sampling / evaluation
    n_samples: int = 32          # samples per condition (evaluation uses their ensemble mean; 32 halves the Monte-Carlo noise of 8)
    guidance_scale: float = 1.0  # 1.0 = plain conditional sampling

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load a saved config; keys that no longer exist in Config (older runs) are dropped with a note."""
        return cls(**_coerce_all(json.loads(Path(path).read_text()), strict=False))


PRESETS: dict[str, dict[str, Any]] = {
    "dev": dict(run_dir="runs/dev", max_steps=300, ckpt_every=100, log_every=10, batch_size=16,
                num_train_timesteps=100, num_inference_steps=100, n_samples=4, num_workers=2),
    "full": dict(run_dir="runs/full", max_steps=20000),
}


def _coerce(value: Any, ftype: Any) -> Any:
    """Coerce a CLI/JSON value to the dataclass field type (handles tuple[...] and bool)."""
    origin = get_origin(ftype)
    if origin is tuple:
        if isinstance(value, str):
            value = [v for v in value.replace("(", "").replace(")", "").split(",") if v.strip()]
        inner = get_args(ftype)
        elem_t = inner[0] if inner else str
        return tuple(_coerce(v, elem_t) for v in value)
    if ftype is bool:
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y")
        return bool(value)
    if ftype in (int, float, str):
        return ftype(value)
    return value


def _coerce_all(d: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    """strict=True (CLI overrides): unknown keys are an error. strict=False (saved configs): they are dropped."""
    types = {f.name: f.type for f in fields(Config)}
    # dataclass field types may be strings under `from __future__ import annotations`
    resolved = {}
    for k, v in d.items():
        if k not in types:
            if strict:
                raise KeyError(f"unknown config key: {k}. Known keys: {sorted(types)}")
            print(f"config: ignoring legacy key {k!r}")
            continue
        t = types[k]
        if isinstance(t, str):
            t = eval(t, {"tuple": tuple, "int": int, "float": float, "str": str, "bool": bool})  # noqa: S307
        resolved[k] = _coerce(v, t)
    return resolved


def parse_cli(argv: list[str] | None = None, description: str = "") -> tuple[Config, argparse.Namespace]:
    """Build a Config from ``--preset`` and ``--set key=value`` overrides.

    Returns (config, extra_namespace) where extra holds ``--config`` if a saved JSON was given.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--config", default=None, help="path to a saved config.json (e.g. from a run_dir)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    args, _ = p.parse_known_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.preset:
        cfg = Config(**{**asdict(cfg), **_coerce_all(PRESETS[args.preset])})
    overrides = {}
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k] = v
    if overrides:
        cfg = Config(**{**asdict(cfg), **_coerce_all(overrides)})
    return cfg, args
