"""Convert the raw Veros ACC runs into one dataset file (analogue of DINO's scripts/xarray_numpy.py).

Input : ``<raw_dir>/ck<ck>_eps<eps>.npz`` files, one per run. Each holds arrays of shape
        (t, x, y, z) including 2 ghost cells on each horizontal side, plus ``time`` (s), ``zt`` (m).
Output: one uncompressed npz with, for every kept snapshot of every run,
        temp/salt as float32 (N, Z, Y, X) on the interior grid, the run parameters, the land mask,
        an optional stored split, and the per-level normalisation statistics (by default on ALL runs, so that
        every hold-out split shares one normalised space; ``--stats train`` restricts them to the training runs).

Only T and S are read (by seeking inside the zip, the other variables are never touched).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import time as clock
import zipfile

import numpy as np
from numpy.lib import format as npf

SECONDS_PER_YEAR = 360 * 86400.0
GHOST = 2


# ----------------------------------------------------------------------------- raw file access
def _member(zf: zipfile.ZipFile, name: str):
    info = zf.getinfo(name)
    assert info.compress_type == zipfile.ZIP_STORED, f"{name} is compressed; seeking needs ZIP_STORED"
    fh = zf.open(name)
    ver = npf.read_magic(fh)
    if ver == (1, 0):                       # public API only: the private _read_array_header vanished in numpy 2.4
        shape, _, dtype = npf.read_array_header_1_0(fh)
    elif ver == (2, 0):
        shape, _, dtype = npf.read_array_header_2_0(fh)
    else:
        raise ValueError(f"{name}: unsupported .npy format version {ver}")
    return fh, tuple(shape), np.dtype(dtype), fh.tell()


def read_time_block(zf: zipfile.ZipFile, name: str, a: int, b: int) -> np.ndarray:
    """Read snapshots a..b-1 of a (t, ...) array without touching the rest of the file."""
    fh, shape, dtype, hdr = _member(zf, name)
    frame = int(np.prod(shape[1:])) * dtype.itemsize
    fh.seek(hdr + a * frame)
    buf = fh.read((b - a) * frame)
    return np.frombuffer(buf, dtype).reshape((b - a,) + shape[1:])


def read_small(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(zf.read(name)))


def parse_run_name(path: str) -> tuple[float, float]:
    m = re.match(r"ck([\d.]+)_eps([\d.]+)\.npz$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse (ck, eps) from {path}")
    return float(m.group(1)), float(m.group(2))


def to_zyx_interior(block: np.ndarray) -> np.ndarray:
    """(t, x, y, z) with ghost cells -> (t, z, y, x) interior, float32."""
    return np.ascontiguousarray(block.transpose(0, 3, 2, 1)[:, :, GHOST:-GHOST, GHOST:-GHOST], dtype=np.float32)


# ----------------------------------------------------------------------------- split
def choose_holdout(run_ck: np.ndarray, run_eps: np.ndarray, mode: str, n: int, seed: int,
                   rows: list[float] | None = None) -> np.ndarray:
    cks, epss = np.unique(run_ck), np.unique(run_eps)
    ick = np.searchsorted(cks, run_ck)
    ieps = np.searchsorted(epss, run_eps)
    if mode == "none":
        return np.zeros(0, dtype=np.int64)
    if mode == "rows":                       # whole rows of constant c_k (a horizontal band of the grid)
        assert rows, "--holdout-ck is required with --holdout-mode rows"
        sel = np.zeros(len(run_ck), dtype=bool)
        for c in rows:
            j = int(np.argmin(np.abs(np.log(cks) - np.log(c))))
            assert abs(np.log(cks[j]) - np.log(c)) < 0.02, f"c_k={c} is not on the grid {cks}"
            sel |= ick == j
        return np.where(sel)[0]
    if mode == "interior_random":
        cand = np.where((ick >= 1) & (ick <= len(cks) - 2) & (ieps >= 1) & (ieps <= len(epss) - 2))[0]
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False))
    if mode == "row_ck_max":
        return np.where(ick == ick.max())[0]
    raise ValueError(f"unknown holdout mode {mode!r}")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="folder with ck*_eps*.npz")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--fields", default="temp,salt")
    p.add_argument("--last-years", type=float, default=20.0, help="keep snapshots in the last N years of each run")
    p.add_argument("--stride", type=int, default=1, help="keep every k-th of those snapshots")
    p.add_argument("--n-holdout", type=int, default=10)
    p.add_argument("--holdout-mode", default="none", choices=["interior_random", "row_ck_max", "rows", "none"],
                   help="optional split stored in the file (normally the split is chosen in the training config)")
    p.add_argument("--stats", default="all", choices=["all", "train"], help="runs used for the normalisation statistics")
    p.add_argument("--holdout-ck", default="", help="with --holdout-mode rows: comma-separated c_k values of the rows to hold out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-runs", type=int, default=0, help="debug: only the first k runs (sorted by name)")
    args = p.parse_args(argv)

    fields = [f for f in args.fields.split(",") if f]
    files = sorted(glob.glob(os.path.join(args.raw_dir, "ck*_eps*.npz")))
    if args.limit_runs:
        files = files[: args.limit_runs]
    if not files:
        raise SystemExit(f"no run files in {args.raw_dir}")
    t0 = clock.time()

    # ---- time axis and grid from the first run
    with zipfile.ZipFile(files[0]) as zf:
        tvec = read_small(zf, "time.npy")
        zt = read_small(zf, "zt.npy")
        _, tshape, _, _ = _member(zf, f"{fields[0]}.npy")
    keep = np.where(tvec > tvec[-1] - args.last_years * SECONDS_PER_YEAR - 1.0)[0][:: args.stride]
    a, b = int(keep[0]), int(keep[-1]) + 1
    nz, ny, nx = tshape[3], tshape[2] - 2 * GHOST, tshape[1] - 2 * GHOST
    n_run, n_keep = len(files), len(keep)
    N = n_run * n_keep
    print(f"{n_run} runs | keeping {n_keep} snapshots/run in the last {args.last_years:g} y (stride {args.stride}) "
          f"| interior grid (Z,Y,X)=({nz},{ny},{nx}) | N={N} samples")

    data = {f: np.empty((N, nz, ny, nx), dtype=np.float32) for f in fields}
    run_id = np.empty(N, dtype=np.int16)
    time_s = np.empty(N, dtype=np.float64)
    run_names, run_ck, run_eps = [], [], []
    land_mask = None

    for r, f in enumerate(files):
        ck, eps = parse_run_name(f)
        run_names.append(os.path.basename(f)[:-4]); run_ck.append(ck); run_eps.append(eps)
        sl = slice(r * n_keep, (r + 1) * n_keep)
        with zipfile.ZipFile(f) as zf:
            t_run = read_small(zf, "time.npy")
            assert t_run.shape == tvec.shape and np.allclose(t_run, tvec), f"{f}: time axis differs"
            for name in fields:
                blk = to_zyx_interior(read_time_block(zf, f"{name}.npy", a, b)[keep - a])
                assert not np.isnan(blk).any(), f"{f}:{name} has NaNs"
                data[name][sl] = blk
        run_id[sl] = r
        time_s[sl] = tvec[keep]
        # land mask: exactly-zero cells, identical across fields, snapshots and runs
        m = np.all([np.all(data[name][sl] == 0.0, axis=0) for name in fields], axis=0)
        m_any = np.any([np.any(data[name][sl] == 0.0, axis=0) for name in fields], axis=0)
        assert np.array_equal(m, m_any), f"{f}: zero cells differ between fields/snapshots"
        if land_mask is None:
            land_mask = m
        assert np.array_equal(land_mask, m), f"{f}: land mask differs from the first run"
        if (r + 1) % 10 == 0 or r == n_run - 1:
            print(f"  read {r + 1}/{n_run} runs  ({clock.time() - t0:.0f}s)")

    run_ck, run_eps = np.array(run_ck), np.array(run_eps)
    water = ~land_mask
    rows = [float(c) for c in args.holdout_ck.split(",") if c.strip()]
    holdout = choose_holdout(run_ck, run_eps, args.holdout_mode, args.n_holdout, args.seed, rows)
    train_runs = np.setdiff1d(np.arange(n_run), holdout)
    is_train = np.isin(run_id, train_runs)
    print(f"split: {len(train_runs)} train runs, {len(holdout)} hold-out runs -> {[run_names[i] for i in holdout]}")

    # ---- normalisation statistics (float64 accumulation, chunked) on all samples, or on the training samples
    F = len(fields)
    lvl_mean, lvl_std = np.zeros((F, nz)), np.zeros((F, nz))
    lvl_min, lvl_max = np.full((F, nz), np.inf), np.full((F, nz), -np.inf)      # over water cells, for norm_mode=minmax
    tr_idx = np.where(is_train)[0] if args.stats == "train" else np.arange(N)
    print(f"normalisation statistics on {len(tr_idx)} samples ({args.stats})")
    for fi, name in enumerate(fields):
        s1 = np.zeros((nz, ny, nx)); s2 = np.zeros((nz, ny, nx))
        for c in range(0, len(tr_idx), 2000):
            blk = data[name][tr_idx[c:c + 2000]].astype(np.float64)
            s1 += blk.sum(0); s2 += (blk ** 2).sum(0)
            bmin, bmax = blk.min(0), blk.max(0)
            for z in range(nz):
                if water[z].any():
                    lvl_min[fi, z] = min(lvl_min[fi, z], bmin[z][water[z]].min())
                    lvl_max[fi, z] = max(lvl_max[fi, z], bmax[z][water[z]].max())
        n = len(tr_idx)
        for z in range(nz):
            w = water[z]
            if w.any():
                lvl_mean[fi, z] = (s1[z][w].sum()) / (n * w.sum())
                lvl_std[fi, z] = np.sqrt(max(s2[z][w].sum() / (n * w.sum()) - lvl_mean[fi, z] ** 2, 0.0))

    # ---- conditioning statistics over the parameter grid
    cond_keys = np.array(["log_ck", "log_eps"])
    cond_vals = np.stack([np.log(run_ck), np.log(run_eps)], 1)
    cond_mean, cond_std = cond_vals.mean(0), cond_vals.std(0) + 1e-12

    meta = dict(raw_dir=os.path.abspath(args.raw_dir), fields=fields, last_years=args.last_years, stride=args.stride,
                n_runs=n_run, n_keep_per_run=n_keep, holdout_mode=args.holdout_mode, holdout_ck=rows, seed=args.seed,
                stats_on=args.stats,
                dims="(N, Z, Y, X); Z index 0 = bottom (zt ascending to the surface); Y meridional; X zonal (periodic)",
                created=clock.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **data, run_id=run_id, time_s=time_s, year=(time_s / SECONDS_PER_YEAR).astype(np.float32),
             ck=run_ck[run_id].astype(np.float32), eps=run_eps[run_id].astype(np.float32),
             run_names=np.array(run_names), run_ck=run_ck, run_eps=run_eps,
             mask_land=land_mask, zt=zt, holdout_runs=holdout, train_runs=train_runs,
             stats_fields=np.array(fields), lvl_mean=lvl_mean, lvl_std=lvl_std, lvl_min=lvl_min, lvl_max=lvl_max,
             cond_keys=cond_keys, cond_mean=cond_mean, cond_std=cond_std, meta=json.dumps(meta))
    os.replace(tmp, args.out)

    # ---- report
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e9:.2f} GB, {clock.time() - t0:.0f}s)")
    print(f"land cells: {int(land_mask.sum())}/{land_mask.size}")
    for fi, name in enumerate(fields):
        x = data[name][tr_idx[::max(1, len(tr_idx) // 2000)]]                    # subsample for the range check
        lvl = np.abs(x - lvl_mean[fi][:, None, None]) / (3 * np.maximum(lvl_std[fi], 0.05)[:, None, None])
        print(f"{name}: per-level mean range [{lvl_mean[fi].min():.3f}, {lvl_mean[fi].max():.3f}], "
              f"std range [{lvl_std[fi].min():.4f}, {lvl_std[fi].max():.4f}] | "
              f"fraction of water values outside [-1,1] with 3-std: {float(lvl[:, water].__gt__(1).mean()):.4f}")
        half = np.maximum((lvl_max[fi] - lvl_min[fi]) / 2, 0.05)
        print(f"{name}: per-level range [{lvl_min[fi].min():.3f}, {lvl_max[fi].max():.3f}] | minmax: level mean at "
              f"x' in [{((lvl_mean[fi] - (lvl_min[fi] + lvl_max[fi]) / 2) / half).min():+.2f}, "
              f"{((lvl_mean[fi] - (lvl_min[fi] + lvl_max[fi]) / 2) / half).max():+.2f}]")


if __name__ == "__main__":
    main()
