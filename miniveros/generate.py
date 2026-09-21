"""Sample T/S states for given (ck, eps) with a trained model (fork of DINO-Fusion generate_images.py).

    python generate.py --run-dir runs/full --holdout            # the hold-out runs of the dataset
    python generate.py --run-dir runs/full --grid               # every run in the dataset (the 10x10 map)
    python generate.py --run-dir runs/full --ck 0.3 0.3 --eps 0.5 2.0   # explicit pairs (never seen)
Writes an npz with un-normalised fields (land = NaN): temp/salt (n_cond, n_samples, Z, Y, X), ck, eps, run_id.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import Config, _coerce_all
from dataset import CondEncoder, build_transform, resolve_split
from diffusion import Diffusion
from model import ConditionalUNet
from pipeline import LandZero, sample
from train import get_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--holdout", action="store_true"); g.add_argument("--grid", action="store_true")
    g.add_argument("--ck", nargs="+", type=float)
    p.add_argument("--eps", nargs="+", type=float)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--batch", type=int, default=64, help="samples per forward pass")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields (e.g. data_file=...)")
    a = p.parse_args(argv)

    run_dir = Path(a.run_dir)
    cfg = Config.load(run_dir / "config.json")
    if a.set:
        cfg = Config(**{**cfg.__dict__, **_coerce_all(dict(kv.split("=", 1) for kv in a.set))})
    n_samples = a.n_samples or cfg.n_samples
    steps = a.steps or cfg.num_inference_steps
    guidance = a.guidance if a.guidance is not None else cfg.guidance_scale
    device = get_device()

    ds = np.load(cfg.data_file, allow_pickle=False)
    enc = CondEncoder(ds)
    run_ck, run_eps, run_names = np.asarray(ds["run_ck"]), np.asarray(ds["run_eps"]), [str(r) for r in ds["run_names"]]
    if a.holdout:
        _, rid = resolve_split(cfg, ds); tag = "holdout"
    elif a.grid:
        rid = np.arange(len(run_ck)); tag = "grid"
    else:
        assert a.eps and len(a.eps) == len(a.ck), "--ck and --eps must have the same length"
        rid = -np.ones(len(a.ck), dtype=int); tag = "custom"
    cks = run_ck[rid] if tag != "custom" else np.array(a.ck)
    epss = run_eps[rid] if tag != "custom" else np.array(a.eps)
    n_cond = len(cks)

    weights = run_dir / ("model_ema.pt" if a.weights == "ema" and (run_dir / "model_ema.pt").exists() else "model.pt")
    model = ConditionalUNet.load(weights, map_location=device).to(device).eval()
    scheduler = Diffusion(cfg).scheduler
    tr = build_transform(cfg.data_file, cfg, device=device)
    constraints = [LandZero(tr.zero_mask)]
    print(f"{tag}: {n_cond} conditions x {n_samples} samples, {steps} steps, guidance {guidance}, weights {weights.name}, "
          f"device {device}", flush=True)

    Z, Y, X = ds["mask_land"].shape
    out = {f: np.full((n_cond, n_samples, Z, Y, X), np.nan, np.float32) for f in cfg.fields}
    cond_all = enc(np.repeat(cks, n_samples), np.repeat(epss, n_samples)).to(device)       # (n_cond*n_samples, d)
    gen = torch.Generator(device).manual_seed(a.seed)
    t0 = time.time(); total = n_cond * n_samples
    for s in range(0, total, a.batch):
        c = cond_all[s:s + a.batch]
        x = sample(model, scheduler, c, steps, generator=gen, constraints=constraints, guidance_scale=guidance)
        fields = tr.denormalise(x)
        for f in cfg.fields:
            arr = fields[f].cpu().numpy()
            for j in range(arr.shape[0]):
                k = s + j
                out[f][k // n_samples, k % n_samples] = arr[j]
        print(f"  {min(s + a.batch, total)}/{total} samples  {time.time() - t0:.0f}s", flush=True)

    out_path = Path(a.out) if a.out else run_dir / "samples" / f"{tag}_n{n_samples}_s{steps}_{a.weights}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out, ck=cks.astype(np.float32), eps=epss.astype(np.float32), run_id=rid,
             run_names=np.array([run_names[r] if r >= 0 else f"ck{c:g}_eps{e:g}" for r, c, e in zip(rid, cks, epss)]),
             mask_land=ds["mask_land"], zt=ds["zt"], n_samples=n_samples, steps=steps, weights=str(weights.name),
             guidance=guidance, norm_mode=cfg.norm_mode)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
