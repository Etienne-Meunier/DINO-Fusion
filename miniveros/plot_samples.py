"""Visual check: random generated T and S states at three depth levels, next to the true time-mean state.

    python plot_samples.py --samples runs/full_3std/samples/holdout_n8_s1000_ema.npz \
        (--data-file data/veros_acc_TS.npz | --raw-dir <raw runs>) [--run ck0.126_eps0.5556] [--n-samples 3] --out fig.png
Rows: temperature then salinity at the chosen levels (default surface, ~650 m, bottom); columns: truth, then samples.
"""
from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np

SECONDS_PER_YEAR = 360 * 86400.0


def truth_from_raw(raw_dir: str, run: str, last_years: float = 20.0) -> dict[str, np.ndarray]:
    from extract_data import read_small, read_time_block, to_zyx_interior
    with zipfile.ZipFile(os.path.join(raw_dir, run + ".npz")) as zf:
        t = read_small(zf, "time.npy")
        keep = np.where(t > t[-1] - last_years * SECONDS_PER_YEAR - 1.0)[0]
        a, b = int(keep[0]), int(keep[-1]) + 1
        return {f: to_zyx_interior(read_time_block(zf, f"{f}.npy", a, b)).mean(0) for f in ("temp", "salt")}


def truth_from_dataset(data_file: str, run_id: int) -> dict[str, np.ndarray]:
    ds = np.load(data_file, allow_pickle=False)
    sel = np.asarray(ds["run_id"]) == run_id
    return {f: ds[f][sel].mean(0) for f in ("temp", "salt")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True)
    p.add_argument("--data-file", default=None)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--run", default=None, help="run name; default: a random condition of the samples file")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--levels", default="14,6,0", help="z indices (0 = bottom)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    gs = np.load(a.samples, allow_pickle=False)
    names = [str(r) for r in gs["run_names"]]
    rng = np.random.default_rng(a.seed)
    k = names.index(a.run) if a.run else int(rng.integers(len(names)))
    run = names[k]
    n_avail = gs["temp"].shape[1]
    picks = rng.choice(n_avail, size=min(a.n_samples, n_avail), replace=False)
    land = np.asarray(gs["mask_land"]); zt = np.asarray(gs["zt"]); levels = [int(z) for z in a.levels.split(",")]

    truth = None
    if a.data_file:
        truth = truth_from_dataset(a.data_file, int(gs["run_id"][k]))
    elif a.raw_dir:
        truth = truth_from_raw(a.raw_dir, run)
    if truth is not None:
        truth = {f: np.where(land, np.nan, v) for f, v in truth.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = (1 if truth is not None else 0) + len(picks)
    rows = 2 * len(levels)
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols + 1.2, 2.9 * rows), squeeze=False)
    fig.subplots_adjust(top=0.955, bottom=0.02, left=0.07, right=0.9, hspace=0.18, wspace=0.12)
    for fi, field in enumerate(("temp", "salt")):
        for li, z in enumerate(levels):
            r = fi * len(levels) + li
            panels = ([("truth (20-y mean)", truth[field][z])] if truth is not None else []) + \
                     [(f"sample {int(s) + 1}", gs[field][k, s, z]) for s in picks]
            ref = truth[field][z] if truth is not None else gs[field][k, picks[0], z]
            if field == "temp":
                vmin, vmax = np.nanmin(ref), np.nanmax(ref); cmap = "RdYlBu_r"
            else:
                dev = max(np.nanmax(np.abs(np.stack([pp[1] for pp in panels]) - 35.0)), 0.01)
                vmin, vmax = 35.0 - dev, 35.0 + dev; cmap = "PuOr_r"
            for c, (title, arr) in enumerate(panels):
                ax = axs[r, c]
                cm = matplotlib.colormaps[cmap].with_extremes(bad="0.55")     # land in grey
                im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{'T (°C)' if field == 'temp' else 'S (psu)'}\n{zt[z]:.0f} m", fontsize=9)
            plt.colorbar(im, ax=axs[r, :].tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(f"hold-out condition {run}: x zonal (periodic), y meridional; grey = land", fontsize=10, y=0.985)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}  (run {run}, samples {sorted(int(s) + 1 for s in picks)})")


if __name__ == "__main__":
    main()
