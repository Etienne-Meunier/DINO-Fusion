"""Train the conditional DDPM. Fork of DINO-Fusion train.py + DiffusionModel: step-based, resumable, CSV logging.

Usage examples (from the package directory):
    python train.py --preset dev  --set data_file=/path/veros_acc_TS.npz run_dir=/path/runs/dev
    python train.py --preset full --set data_file=... run_dir=... norm_mode=3-std
Rerunning with the same run_dir resumes from run_dir/ckpt.pt (needed for the 2 h dev-QoS cap).
"""
from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from config import parse_cli
from dataset import VerosTSDataset, build_transform, resolve_split
from diffusion import Diffusion
from model import ConditionalUNet
from pipeline import LandZero, sample


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_hash() -> str:
    """Commit hash of the code: from git if available, else from MV_GIT_HASH (set by jobs/submit.sh on the
    login node, because compute nodes may not have git)."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                           cwd=Path(__file__).parent).stdout.strip()
        if h:
            return h
    except Exception:
        pass
    return os.environ.get("MV_GIT_HASH", "unknown")


def save_weights(model, ema, run_dir: Path) -> None:
    model.save(run_dir / "model.pt")
    if ema is not None:
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        model.save(run_dir / "model_ema.pt")
        ema.restore(model.parameters())


@torch.no_grad()
def final_samples(model, ema, diffusion, tr, cfg, device, run_dir: Path) -> None:
    """Small sanity figure at the end of training: surface temperature, samples vs truth, a few conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, hold = resolve_split(cfg, np.load(cfg.data_file, allow_pickle=False))
    ds_h = VerosTSDataset(cfg.data_file, "holdout", tr, cfg.fields, snapshot_stride=12, holdout_runs=hold)
    ds = ds_h if len(ds_h) else VerosTSDataset(cfg.data_file, "train", tr, cfg.fields, snapshot_stride=12, holdout_runs=hold)
    runs = ds.runs()[:4]
    if ema is not None:
        ema.store(model.parameters()); ema.copy_to(model.parameters())
    model.eval()
    fig, axs = plt.subplots(len(runs), 3, figsize=(11, 3.2 * len(runs)), squeeze=False)
    out = {}
    for i, r in enumerate(runs):
        ck, eps = ds.run_condition(r)
        cond = ds.encoder([ck] * 2, [eps] * 2).to(device)
        g = torch.Generator(device).manual_seed(cfg.seed)
        x = sample(model, diffusion.scheduler, cond, cfg.num_inference_steps, generator=g,
                   constraints=[LandZero(tr.zero_mask.to(device), mode=cfg.fill_mode, scheduler=diffusion.scheduler, generator=g)])
        gen = tr.denormalise(x)["temp"].cpu()                          # (2, Z, Y, X)
        truth = ds.run_fields(r)["temp"].mean(0)                       # (Z, Y, X) time mean
        truth = truth.masked_fill(torch.as_tensor(tr.masker.mask[: truth.shape[0]]), float("nan"))
        out[f"{ds.run_names[r]}_gen"] = gen.numpy(); out[f"{ds.run_names[r]}_truth"] = truth.numpy()
        vmin, vmax = np.nanmin(truth[-1]), np.nanmax(truth[-1])
        for j, (arr, title) in enumerate([(truth[-1], "truth (time mean)"), (gen[0, -1], "sample 1"), (gen[1, -1], "sample 2")]):
            im = axs[i, j].imshow(arr.T, origin="lower", vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
            axs[i, j].set_title(f"{ds.run_names[r]}  {title}", fontsize=8); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
        plt.colorbar(im, ax=axs[i, :], fraction=0.02)
    fig.suptitle(f"surface temperature, {cfg.norm_mode}, step {cfg.max_steps}")
    fig.savefig(run_dir / "samples_final.png", dpi=110, bbox_inches="tight")
    np.savez(run_dir / "samples_final.npz", **out)
    if ema is not None:
        ema.restore(model.parameters())
    model.train()


def main(argv=None):
    cfg, _ = parse_cli(argv, description="Train the miniveros conditional diffusion model")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    run_dir = Path(cfg.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n")

    tr = build_transform(cfg.data_file, cfg, device="cpu")
    train_runs, hold = resolve_split(cfg, np.load(cfg.data_file, allow_pickle=False))
    print(f"split {cfg.split_mode}: {len(train_runs)} training runs, {len(hold)} held out", flush=True)
    ds = VerosTSDataset(cfg.data_file, "train", tr, cfg.fields, cfg.snapshot_stride, holdout_runs=hold)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers,
                    pin_memory=(device.type == "cuda"), persistent_workers=cfg.num_workers > 0)
    model = ConditionalUNet(tr.n_channels, tr.padded_shape, ds.encoder.dim, cfg.block_out_channels,
                            cfg.layers_per_block, cfg.cond_hidden).to(device)
    diffusion = Diffusion(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = get_cosine_schedule_with_warmup(opt, cfg.lr_warmup_steps, cfg.max_steps)
    ema = EMAModel(model.parameters(), decay=cfg.ema_decay) if cfg.use_ema else None
    zero_mask = tr.zero_mask.to(device)
    print(f"device={device} | train samples={len(ds)} from {len(ds.runs())} runs | batch={cfg.batch_size} "
          f"| x shape=({tr.n_channels},{tr.padded_shape[0]},{tr.padded_shape[1]}) | params={model.n_params() / 1e6:.2f}M "
          f"| norm={cfg.norm_mode} | steps={cfg.max_steps}", flush=True)

    step = 0
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        step = ck["step"]
        print(f"resumed from {ckpt_path} at step {step}", flush=True)

    log_path = run_dir / "train_log.csv"
    new_log = not log_path.exists()
    logf = open(log_path, "a", newline=""); logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "loss", "lr", "elapsed_s"])

    def checkpoint():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "ema": ema.state_dict() if ema is not None else None, "step": step}, ckpt_path)
        save_weights(model, ema, run_dir)

    model.train()
    t0 = time.time(); running = []; it = iter(dl)
    while step < cfg.max_steps:
        try:
            x, c = next(it)
        except StopIteration:
            it = iter(dl); x, c = next(it)
        x = x.to(device, non_blocking=True); c = c.to(device, non_blocking=True)
        loss = diffusion.training_loss(model, x, c, zero_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        if ema is not None:
            ema.step(model.parameters())
        step += 1; running.append(loss.item())
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            m = float(np.mean(running)); running = []
            logw.writerow([step, f"{m:.6f}", f"{sched.get_last_lr()[0]:.3e}", f"{time.time() - t0:.1f}"]); logf.flush()
            print(f"step {step:6d}/{cfg.max_steps} loss {m:.5f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            checkpoint()
    logf.close()
    print("training done; drawing final samples", flush=True)
    final_samples(model, ema, diffusion, tr, cfg, device, run_dir)
    print(f"done. outputs in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
