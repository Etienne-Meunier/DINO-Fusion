"""Smoke tests for the miniveros pipeline. Run from the package dir:  python -m tests.test_roundtrip [data.npz]

1. transforms: normalise -> denormalise recovers water cells exactly, land -> NaN, padded shape, zero mask.
2. dataset + model forward + training loss + a 3-step sample, all on CPU with tiny tensors.
If a converted dataset path is given (e.g. the 3-run test file), tests run on it; otherwise a synthetic one is made.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config  # noqa: E402
from dataset import VerosTSDataset, build_transform  # noqa: E402
from diffusion import Diffusion, PerLevelClipDDPMScheduler  # noqa: E402
from model import ConditionalUNet  # noqa: E402
from pipeline import LandZero, sample  # noqa: E402
from transforms import FieldTransform  # noqa: E402


def synthetic_dataset(path: str, n_runs=4, n_snap=3, Z=15, Y=42, X=30, seed=0) -> str:
    rng = np.random.default_rng(seed)
    mask = np.zeros((Z, Y, X), bool); mask[:, 11:, 5:7] = True                      # a fake ridge
    N = n_runs * n_snap
    temp = (rng.normal(8, 3, (N, Z, Y, X)) + np.linspace(0, 6, Z)[None, :, None, None]).astype(np.float32)
    salt = np.full((N, Z, Y, X), 35.0, np.float32)
    temp[:, mask] = 0.0; salt[:, mask] = 0.0
    run_id = np.repeat(np.arange(n_runs), n_snap).astype(np.int16)
    run_ck = np.array([0.0125, 0.05, 0.2, 0.8]); run_eps = np.array([0.0875, 0.35, 1.4, 5.6])
    water = ~mask
    def lvl_stats(a):
        m = np.array([a[:, z][:, water[z]].mean() for z in range(Z)]); s = np.array([a[:, z][:, water[z]].std() for z in range(Z)])
        return m, s
    tm, ts = lvl_stats(temp); sm, ss = lvl_stats(salt)
    cond = np.stack([np.log(run_ck), np.log(run_eps)], 1)
    np.savez(path, temp=temp, salt=salt, run_id=run_id, time_s=np.arange(N) * 1.0, year=np.arange(N) / 12.0,
             ck=run_ck[run_id].astype(np.float32), eps=run_eps[run_id].astype(np.float32),
             run_names=np.array([f"ck{c}_eps{e}" for c, e in zip(run_ck, run_eps)]), run_ck=run_ck, run_eps=run_eps,
             mask_land=mask, zt=np.linspace(-1942, -14, Z), holdout_runs=np.array([3]), train_runs=np.array([0, 1, 2]),
             stats_fields=np.array(["temp", "salt"]), lvl_mean=np.stack([tm, sm]), lvl_std=np.stack([ts, ss]),
             cond_keys=np.array(["log_ck", "log_eps"]), cond_mean=cond.mean(0), cond_std=cond.std(0) + 1e-12,
             meta=json.dumps({"synthetic": True}))
    return path


def test_transforms(data_file: str):
    ds = np.load(data_file, allow_pickle=False)
    land = torch.as_tensor(ds["mask_land"])
    for mode in ("3-std", "6-std"):
        cfg = Config(norm_mode=mode)
        tr = FieldTransform.from_dataset(ds, ("temp", "salt"), mode, cfg.std_floor, cfg.paddings)
        temp = torch.as_tensor(ds["temp"][:5]); salt = torch.as_tensor(ds["salt"][:5])       # batch of 5
        x = tr.normalise({"temp": temp, "salt": salt})
        assert x.shape == (5, 30, 48, 32), x.shape
        assert x.shape[-2] % 16 == 0 and x.shape[-1] % 16 == 0, "padded grid must halve 4 times"
        assert torch.isfinite(x).all()
        assert (x.masked_select(tr.zero_mask.expand_as(x)) == 0).all(), "land/padding must be exactly 0"
        back = tr.denormalise(x)
        w = ~land
        for name, ref in (("temp", temp), ("salt", salt)):
            got = back[name]
            assert got.shape == ref.shape
            assert torch.allclose(got[:, w], ref[:, w], atol=1e-3, rtol=0), f"{mode}/{name}: roundtrip mismatch"
            assert torch.isnan(got[:, land]).all(), f"{mode}/{name}: land should be NaN after denormalise"
        salt_norm = x[:, 15:30][:, ~tr.zero_mask[15:30]]
        assert salt_norm.abs().max() < 1e-4, "constant salinity must normalise to 0"
        frac_out = (x[:, :15].abs() > 1).float().mean().item()
        print(f"  [{mode}] roundtrip OK | padded {tuple(x.shape[-2:])} | zero cells {int(tr.zero_mask.sum())} "
              f"| temp values outside [-1,1]: {100 * frac_out:.2f}%")
    # single sample (no batch dim)
    x1 = tr.normalise({"temp": temp[0], "salt": salt[0]}); assert x1.shape == (30, 48, 32)


def test_model_and_loss(data_file: str):
    cfg = Config(num_train_timesteps=20, num_inference_steps=3, block_out_channels=(32, 32, 64, 64), cond_drop_prob=0.2, split_mode="file")
    tr = build_transform(data_file, cfg)
    ds = VerosTSDataset(data_file, "train", tr, cfg.fields, snapshot_stride=1)
    x, c = ds[0]
    assert x.shape == (tr.n_channels, *tr.padded_shape) and c.shape == (2,)
    model = ConditionalUNet(tr.n_channels, tr.padded_shape, ds.encoder.dim, cfg.block_out_channels, cfg.layers_per_block, 64)
    xb = torch.stack([ds[i][0] for i in range(2)]); cb = torch.stack([ds[i][1] for i in range(2)])
    out = model(xb, torch.tensor([3, 7]), cb)
    assert out.shape == xb.shape and torch.isfinite(out).all()
    loss = Diffusion(cfg).training_loss(model, xb, cb, tr.zero_mask)
    assert torch.isfinite(loss), loss
    loss.backward()
    cfg.mask_loss = True
    assert torch.isfinite(Diffusion(cfg).training_loss(model, xb, cb, tr.zero_mask))
    s = sample(model.eval(), Diffusion(cfg).scheduler, cb, 3, generator=torch.Generator().manual_seed(0),
               constraints=[LandZero(tr.zero_mask)])
    # per-level clip: bounds exist, salt channels are narrow around 0, and the scheduler clamps per channel
    assert tr.clip_lo is not None and tr.clip_lo.shape == (tr.n_channels,) and (tr.clip_lo < tr.clip_hi).all()
    assert (tr.clip_hi[15:] <= 0.05).all() and (tr.clip_lo[15:] >= -0.05).all()
    sch = Diffusion(cfg, clip_bounds=(tr.clip_lo, tr.clip_hi)).scheduler
    assert isinstance(sch, PerLevelClipDDPMScheduler) and sch.config.thresholding
    probe = torch.full((2, tr.n_channels, 4, 4), 5.0); probe[:, 0] = -5.0
    out = sch._threshold_sample(probe)
    assert torch.allclose(out[:, 0], tr.clip_lo[0].expand(2, 4, 4)) and torch.allclose(out[:, 1], tr.clip_hi[1].expand(2, 4, 4))
    s2 = sample(model.eval(), sch, cb, 3, generator=torch.Generator().manual_seed(0), constraints=[LandZero(tr.zero_mask)])
    assert torch.isfinite(s2).all()
    assert s.shape == xb.shape and torch.isfinite(s).all()
    assert (s.masked_select(tr.zero_mask.expand_as(s)) == 0).all()
    fields = tr.denormalise(s)
    assert set(fields) == {"temp", "salt"} and fields["temp"].shape[-3:] == (15, 42, 30)
    with tempfile.TemporaryDirectory() as d:
        model.save(f"{d}/m.pt"); m2 = ConditionalUNet.load(f"{d}/m.pt")
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), m2.state_dict().values()))
    print(f"  model {model.n_params() / 1e6:.2f}M params | forward/loss/sample/save-load OK | "
          f"train set {len(ds)} samples from runs {ds.runs()} | cond example {c.numpy().round(3)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]; print(f"using dataset {data_file}")
    else:
        data_file = synthetic_dataset(os.path.join(tempfile.gettempdir(), "miniveros_synth.npz")); print("using synthetic dataset")
    print("test_transforms"); test_transforms(data_file)
    print("test_model_and_loss"); test_model_and_loss(data_file)
    print("ALL TESTS PASSED")
