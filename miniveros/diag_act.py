"""Which layer makes a localised artefact?  Forward hooks on every UNet block: the per-position RMS over channels of
each block's output (mean over inputs), for true snapshots of one run noised to a few t.

    python diag_act.py --run-dir runs/x --run-name ck0.126_eps0.8819 --out runs/x/act.npz [--weights raw] [--t 600 50]
Saved: for each layer name, the map (h, w), plus layer shapes; analysed offline.
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
from train import get_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--t", type=int, nargs="+", default=(600, 50))
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
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

    u = model.unet
    layers = {"conv_in": u.conv_in}
    for i, b in enumerate(u.down_blocks):
        for j, rn in enumerate(b.resnets):
            layers[f"down{i}.res{j}"] = rn
        if b.downsamplers:
            layers[f"down{i}.downsample"] = b.downsamplers[0]
    layers["mid"] = u.mid_block
    for i, b in enumerate(u.up_blocks):
        for j, rn in enumerate(b.resnets):
            layers[f"up{i}.res{j}"] = rn
        if b.upsamplers:
            layers[f"up{i}.upsample"] = b.upsamplers[0]
    layers["conv_out"] = u.conv_out
    acts = {}
    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            acts[name] = o.detach().float()
        return f
    handles = [m.register_forward_hook(hook(n)) for n, m in layers.items()]
    out = {"layer_names": np.array(list(layers)), "t": np.array(a.t), "weights": weights.name, "prediction_type": cfg.prediction_type,
           "zero_mask": tr.zero_mask.cpu().numpy()}
    with torch.no_grad():
        for t in a.t:
            tt = torch.full((a.n,), t, device=device, dtype=torch.long)
            noise = randn_tensor(x0.shape, generator=gen, device=device)
            xt = scheduler.add_noise(x0, noise, tt)
            xt = torch.where(tr.zero_mask[None], (1 - scheduler.alphas_cumprod[t]).sqrt().to(xt) * noise, xt)
            acts.clear(); pred = model(xt, tt, cond)
            out[f"pred_t{t}"] = pred.cpu().numpy(); out[f"x0_true"] = x0.cpu().numpy()
            for n, o in acts.items():
                out[f"map_t{t}_{n}"] = o.pow(2).mean(1).sqrt().mean(0).cpu().numpy()          # (h, w) RMS over channels, mean over inputs
                out[f"chmax_t{t}_{n}"] = o.abs().amax(dim=(0, 2, 3)).cpu().numpy()             # per-channel max |activation|
                out[f"chmaxpos_t{t}_{n}"] = o.abs().mean(0).flatten(1).argmax(1).cpu().numpy()  # position of each channel's max
    for h in handles:
        h.remove()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(layers)} layers, t {list(a.t)}")


if __name__ == "__main__":
    main()
