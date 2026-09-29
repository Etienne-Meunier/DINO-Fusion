"""Per-run distributions of T and S at one depth, read from the raw run files (last 20 years, water cells), and
the origin of the narrow peaks: the zonally uniform southern rows of the grid (the restoring zone).

  python level_density.py --raw-dir ../full_state/mnk965ig --depth 150 --out results/data/level_density_150m.png
"""
from __future__ import annotations

import argparse, glob, os, zipfile
import numpy as np
from extract_data import SECONDS_PER_YEAR, parse_run_name, read_small, read_time_block, to_zyx_interior


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True); p.add_argument("--depth", type=float, default=150.0)
    p.add_argument("--last-years", type=float, default=20.0); p.add_argument("--out", required=True)
    p.add_argument("--map-run", default="ck0.2_eps2.222"); p.add_argument("--bins", type=int, default=600)
    p.add_argument("--rows", type=int, default=8, help="southern rows to characterise")
    a = p.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.raw_dir, "ck*_eps*.npz")))
    with zipfile.ZipFile(files[0]) as zf:
        tvec = read_small(zf, "time.npy"); zt = read_small(zf, "zt.npy")
    keep = np.where(tvec > tvec[-1] - a.last_years * SECONDS_PER_YEAR - 1.0)[0]
    t0, t1 = int(keep[0]), int(keep[-1]) + 1
    iz = int(np.argmin(np.abs(np.abs(zt) - a.depth))); depth = abs(zt[iz])
    print(f"level {iz}: {depth:.0f} m (nearest to {a.depth:g} m), {t1 - t0} snapshots per run")
    vals, cks, names, tmean, row_mean, row_zstd = {"temp": [], "salt": []}, [], [], {}, [], []
    for f in files:
        ck, _ = parse_run_name(f); cks.append(ck); names.append(os.path.basename(f)[:-4])
        with zipfile.ZipFile(f) as zf:
            for name in vals:
                blk = to_zyx_interior(read_time_block(zf, f"{name}.npy", t0, t1))[:, iz]      # (t, y, x)
                if name == "temp":
                    land = np.all(blk == 0.0, axis=0); tm = blk.mean(0)
                    rows = [np.where(~land[y], tm[y], np.nan) for y in range(a.rows)]
                    row_mean.append([np.nanmean(r) for r in rows]); row_zstd.append([np.nanstd(r) for r in rows])
                    if names[-1] == a.map_run:
                        tmean["map"] = np.where(~land, tm, np.nan)
                vals[name].append(blk[:, ~land].ravel())
    cks = np.array(cks); row_mean = np.array(row_mean); row_zstd = np.array(row_zstd)          # (runs, rows)
    print(f"southern rows at {depth:.0f} m (time-mean T): row | mean over runs | std across runs | mean zonal std")
    for y in range(a.rows):
        print(f"  y={y}: {row_mean[:, y].mean():7.3f} °C | {row_mean[:, y].std():.4f} K | {row_zstd[:, y].mean():.4f} K")
    fixed = [y for y in range(a.rows) if row_zstd[:, y].mean() < 0.05]          # zonally uniform rows (the restoring zone)
    print("zonally uniform southern rows:", fixed)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.35, 0.8, 1]})
    cmap = plt.cm.viridis; norm = matplotlib.colors.LogNorm(cks.min(), cks.max())
    for ax, name, unit in ((axes[0], "temp", "°C"), (axes[1], "salt", "psu")):
        allv = np.concatenate(vals[name]); lo, hi = allv.min(), allv.max()
        if hi - lo < 1e-3:
            lo, hi = lo - 0.05, hi + 0.05
        edges = np.linspace(lo, hi, a.bins + 1); c = 0.5 * (edges[1:] + edges[:-1]); w = edges[1] - edges[0]
        for v, ck in zip(vals[name], cks):
            h, _ = np.histogram(v, bins=edges); ax.plot(c, h / (h.sum() * w), color=cmap(norm(ck)), lw=0.5, alpha=0.6)
        H, _ = np.histogram(allv, bins=edges); D = H / (H.sum() * w); ax.plot(c, D, color="k", lw=1.2)
        ax.set_yscale("log"); ax.set_ylim(bottom=max(1e-4, 1e-4 * D.max()))
        ax.set_xlabel(f"{'T' if name == 'temp' else 'S'} ({unit})"); ax.set_ylabel("density")
        if name == "temp":
            rowcol = plt.cm.tab10(np.arange(a.rows))
            for y in fixed:
                ax.axvline(row_mean[:, y].mean(), color=rowcol[y], lw=1.0, ls="--",
                           label=f"row y={y}: {row_mean[:, y].mean():.2f} ± {row_mean[:, y].std():.2f} °C across runs")
            ax.legend(fontsize=7, loc="upper left", title="zonally uniform southern rows", title_fontsize=7)
            ax.set_title(f"T at {depth:.0f} m: one density per run (colour: $c_k$), all runs in black", fontsize=9)
        else:
            ax.set_title(f"S at {depth:.0f} m: all values in [{allv.min():.4f}, {allv.max():.4f}] psu", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); fig.colorbar(sm, ax=axes[0], label="$c_k$", fraction=0.04, pad=0.02)
    ax = axes[2]; m = tmean["map"]
    im = ax.imshow(m, origin="lower", cmap="RdYlBu_r", aspect="auto"); fig.colorbar(im, ax=ax, label="T (°C)", fraction=0.04, pad=0.02)
    for y in fixed:
        ax.axhline(y, color=plt.cm.tab10(y), lw=1.5, ls="--")
    ax.set_title(f"time-mean T at {depth:.0f} m, run {a.map_run}; dashed: the zonally uniform southern rows", fontsize=9)
    ax.set_xlabel("x (zonal)"); ax.set_ylabel("y (meridional)")
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)


if __name__ == "__main__":
    main()
