"""Where does a sampling chain go wrong?  Two diagnostics for one condition of a trained run, saved to an npz:

  denoise: true snapshots of that run are noised to several t and denoised in one step; the network's own
           clean-state estimate x0_hat is stored per t (no chain involved).
  chain:   the usual sampler, with x0_hat and the state recorded at the same t (the chain's dynamics).

    python diag_spot.py --run-dir runs/x --run-name ck0.126_eps0.8819 --n 8 --out runs/x/diag_ck0.126_eps0.8819.npz
Everything is stored in normalised, padded network space (land + padding = 0); x0_true holds the snapshots used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from config import Config, _coerce_all
from dataset import CondEncoder, build_transform
from diffusion import Diffusion
from model import ConditionalUNet
from pipeline import LandZero
from train import get_device

T_DENOISE = (950, 800, 600, 400, 200, 100, 50, 20, 5)
T_CHAIN = (999, 950, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 20, 10, 5, 1, 0)


def x0_from_pred(scheduler, pred, x_t, t):
    ab = scheduler.alphas_cumprod.to(x_t)[t].view(-1, 1, 1, 1)
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return (x_t - (1 - ab).sqrt() * pred) / ab.sqrt()
    if kind == "v_prediction":
        return ab.sqrt() * x_t - (1 - ab).sqrt() * pred
    return pred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-name", required=True, help="dataset run, e.g. ck0.126_eps0.8819")
    p.add_argument("--n", type=int, default=8, help="snapshots / chains")
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--constraint", choices=["landzero", "none"], default="landzero")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    run_dir = Path(a.run_dir)
    cfg = Config.load(run_dir / "config.json")
    if a.set:
        cfg = Config(**{**cfg.__dict__, **_coerce_all(dict(kv.split("=", 1) for kv in a.set))})
    device = get_device()
    ds = np.load(cfg.data_file, allow_pickle=False)
    names = [str(r) for r in ds["run_names"]]
    r = names.index(a.run_name)
    sel = np.where(np.asarray(ds["run_id"]) == r)[0]
    pick = sel[np.linspace(0, len(sel) - 1, a.n).round().astype(int)]
    tr = build_transform(cfg.data_file, cfg, device=device)
    x0 = torch.stack([tr.normalise({f: torch.from_numpy(np.asarray(ds[f][i])) for f in cfg.fields}) for i in pick]).to(device)
    cond = CondEncoder(ds)(np.full(a.n, ds["run_ck"][r]), np.full(a.n, ds["run_eps"][r])).to(device)

    weights = run_dir / ("model_ema.pt" if a.weights == "ema" and (run_dir / "model_ema.pt").exists() else "model.pt")
    model = ConditionalUNet.load(weights, map_location=device).to(device).eval()
    scheduler = Diffusion(cfg).scheduler
    gen = torch.Generator(device).manual_seed(a.seed)
    steps = a.steps or cfg.num_inference_steps
    print(f"{a.run_name}: run {r}, {a.n} snapshots {pick.tolist()}, {weights.name}, {cfg.prediction_type}, "
          f"clip {'on' if cfg.clip_sample else 'off'}, constraint {a.constraint}, {steps} steps", flush=True)
    out = {"x0_true": x0.cpu().numpy(), "zero_mask": tr.zero_mask.cpu().numpy(), "t_denoise": np.array(T_DENOISE),
           "t_chain": np.array(T_CHAIN), "alphas_cumprod": scheduler.alphas_cumprod.numpy(), "run_name": a.run_name,
           "prediction_type": cfg.prediction_type, "weights": weights.name, "constraint": a.constraint}

    with torch.no_grad():
        # one-step denoising of true snapshots
        den = []
        for t in T_DENOISE:
            tt = torch.full((a.n,), t, device=device, dtype=torch.long)
            noise = randn_tensor(x0.shape, generator=gen, device=device)
            xt = scheduler.add_noise(x0, noise, tt)
            xt = torch.where(tr.zero_mask[None], (1 - scheduler.alphas_cumprod[t]).sqrt().to(xt) * noise, xt)  # as in training
            den.append(x0_from_pred(scheduler, model(xt, tt, cond), xt, tt).cpu().numpy())
        out["x0hat_denoise"] = np.stack(den)
        # the sampling chain, recording x0_hat and the state
        constraints = [LandZero(tr.zero_mask, scheduler, gen)] if a.constraint == "landzero" else []
        x = randn_tensor(x0.shape, generator=gen, device=device)
        scheduler.set_timesteps(steps)
        rec_x0, rec_x, rec_t = [], [], []
        for t in scheduler.timesteps:
            pred = model(x, t, cond)
            step = scheduler.step(pred, t, x, generator=gen)
            if int(t) in T_CHAIN:
                rec_t.append(int(t)); rec_x0.append(step.pred_original_sample.cpu().numpy()); rec_x.append(x.cpu().numpy())
            x = step.prev_sample
            for c in constraints:
                x = c(x, t)
        out["t_chain"] = np.array(rec_t); out["x0hat_chain"] = np.stack(rec_x0); out["x_chain"] = np.stack(rec_x)
        out["final"] = x.cpu().numpy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
