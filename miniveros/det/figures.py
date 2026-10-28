"""Figures of the deterministic-emulator evaluation, from the files written by det.run.

    python -m det.figures [--data data/veros_acc_lastyear.npz] [--results det/results]

data.png: the target over the parameter grid and against the collapsed parameter s; splits.png: mean hold-out
RMSE per split; profiles.png: RMSE by depth per split; examples.png: true and predicted anomaly sections of two
runs, each held out in several splits.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402

INK, INK2, MUTED, GRIDLINE, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {                     # method -> label, colour, marker, line style (the network and the two best baselines)
    "mlp": ("network (MLP)", "#2a78d6", "o", "-"),
    "gp_s": ("GP on s", "#eb6834", "s", "-"),
    "rbf": ("thin-plate RBF", "#1baf7a", "^", "-"),
    "nearest": ("nearest run", MUTED, "D", "--"),
}
ORDER = ["loo", "scattered", "band", "centre5", "centre7", "top", "outer51", "outer75", "four"]
LABEL = {"loo": "leave-one-out", "scattered": "scattered", "band": "band of 3 rows", "centre5": "centre 5×5",
         "centre7": "centre 7×7", "top": "top row", "outer51": "outer 51", "outer75": "outer 75", "four": "four runs"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a52a2a"])
for cmap in (BLUES, DIVERGING):
    cmap.set_bad("#d9d8d2")                                       # land
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": False, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False})


def load(results: Path):
    rmse = defaultdict(dict)          # (split, method) -> {run: rmse}
    with open(results / "metrics.csv") as f:
        for r in csv.DictReader(f):
            rmse[(r["split"], r["method"])][r["run"]] = float(r["rmse_K"])
    levels = defaultdict(list)        # (split, method) -> [(z, rmse)]
    with open(results / "levels.csv") as f:
        for r in csv.DictReader(f):
            levels[(r["split"], r["method"])].append((float(r["z_m"]), float(r["rmse_K"])))
    return rmse, levels


def legend_on_top(fig):
    h = [plt.Line2D([], [], color=c, marker=mk, ls=ls, lw=1.4, ms=6) for _, c, mk, ls in STYLE.values()]
    fig.legend(h, [v[0] for v in STYLE.values()], loc="upper center", ncol=len(STYLE), bbox_to_anchor=(0.5, 1.0))


def fig_data(data, out):
    """Domain-mean last-year T (water cells) over the grid, and against s = log c_k - b log c_eps, the direction
    along which the first PCA coefficient of all states varies linearly (as in the GP-on-s baseline)."""
    from det.methods import PCA
    water = ~data["mask_land"]
    Y = data["temp"][:, water].astype(np.float64)
    run_ck, run_eps = data["run_ck"], data["run_eps"]
    logp = np.c_[np.log(run_ck), np.log(run_eps)]
    a1 = PCA(1).fit(Y).coefficients(Y)[:, 0]
    w = np.linalg.lstsq(np.c_[logp, np.ones(len(Y))], a1, rcond=None)[0][:2]
    b = -w[1] / w[0]
    s = logp[:, 0] - b * logp[:, 1]
    tm = Y.mean(1)
    cks, epss = np.unique(run_ck), np.unique(run_eps)
    g = np.full((len(cks), len(epss)), np.nan)
    g[np.searchsorted(cks, run_ck), np.searchsorted(epss, run_eps)] = tm
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
    im = axs[0].imshow(g, origin="lower", cmap=BLUES)
    axs[0].set_xticks(range(0, len(epss), 3), [f"{e:.3g}" for e in epss[::3]])
    axs[0].set_yticks(range(0, len(cks), 3), [f"{c:.3g}" for c in cks[::3]])
    axs[0].set_xlabel("c_eps"); axs[0].set_ylabel("c_k"); axs[0].tick_params(length=0)
    axs[0].set_title("mean last-year T over water cells (K)")
    cb = fig.colorbar(im, ax=axs[0], shrink=0.85, pad=0.02); cb.outline.set_visible(False)
    axs[1].scatter(s, tm, s=22, color=STYLE["mlp"][1], edgecolor=SURFACE, linewidth=0.6, zorder=3)
    axs[1].set_xlabel(f"s = log c_k − {b:.2f} log c_eps"); axs[1].set_ylabel("mean last-year T (K)")
    axs[1].set_title("the two parameters act through s"); axs[1].grid(alpha=1); axs[1].set_axisbelow(True)
    axs[1].tick_params(length=0)
    for sp in ("top", "right"):
        axs[1].spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "data.png", dpi=150); plt.close(fig)
    print(f"data.png: s = log c_k - {b:.3f} log c_eps; domain-mean T {tm.min():.3f} to {tm.max():.3f} K")


def fig_splits(rmse, out):
    splits = [s for s in ORDER if (s, "mlp") in rmse]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(splits) + 1.0))
    for j, (m, (_, c, mk, _)) in enumerate(STYLE.items()):
        x = [np.mean(list(rmse[(s, m)].values())) for s in splits]
        ax.scatter(x, np.arange(len(splits)) + 0.12 * (j - 1.5), color=c, marker=mk, s=30, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)
    ax.set_yticks(range(len(splits)), [f"{LABEL[s]} ({len(rmse[(s, 'mlp')])})" for s in splits])
    ax.set_xscale("log"); ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2], ["0.01", "0.02", "0.05", "0.1", "0.2"])
    ax.invert_yaxis(); ax.grid(axis="x"); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlabel("mean RMSE over the held-out runs (K)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    legend_on_top(fig); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "splits.png", dpi=150); plt.close(fig)


def fig_profiles(levels, out):
    splits = [s for s in ORDER if (s, "mlp") in levels]
    nc = 3; nr = int(np.ceil(len(splits) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(9, 2.6 * nr + 0.6), sharey=True, squeeze=False)
    for ax, s in zip(axs.flat, splits):
        for m, (_, c, mk, ls) in STYLE.items():
            z, v = np.array(levels[(s, m)]).T
            ax.plot(v, -z, color=c, ls=ls, lw=1.4, marker=mk, ms=3.5)
        ax.set_title(LABEL[s]); ax.grid(alpha=1); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.set_xlim(left=0)
    for ax in axs.flat[len(splits):]:
        ax.set_visible(False)
    axs[0, 0].invert_yaxis()
    for ax in axs[:, 0]:
        ax.set_ylabel("depth (m)")
    for ax in axs[-1]:
        ax.set_xlabel("RMSE (K)")
    legend_on_top(fig); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "profiles.png", dpi=150); plt.close(fig)


EXAMPLE_ROWS = [("centre7", "ck0.126_eps0.8819"), ("four", "ck0.126_eps0.8819"),     # the centre run and the warm
                ("top", "ck0.8_eps0.0875"), ("outer75", "ck0.8_eps0.0875"), ("four", "ck0.8_eps0.0875")]   # corner


def fig_examples(data, rmse, results, out):
    """Per example run, left: where it sits among the training runs; right: zonal-mean sections of the true and the
    predicted anomaly (relative to the mean of all runs) and of the error, on one colour scale per run."""
    ex = np.load(results / "examples.npz")
    rows = [(s, r) for s, r in EXAMPLE_ROWS if f"{s}/{r}/mlp" in ex.files and (s, "mlp") in rmse]
    if not rows:
        return
    water, zt = ~data["mask_land"], data["zt"]
    names = [str(n) for n in data["run_names"]]
    cks, epss = np.unique(data["run_ck"]), np.unique(data["run_eps"])
    ick, ieps = np.searchsorted(cks, data["run_ck"]), np.searchsorted(epss, data["run_eps"])
    ref = data["temp"].mean(0)
    zonal = lambda f: np.nanmean(np.where(water, f, np.nan), axis=2)              # (Z, Y, X) -> (Z, Y)
    sections, stats, lim = [], [], {}
    for s, r in rows:
        pred = np.zeros(water.shape); pred[water] = ex[f"{s}/{r}/mlp"]
        ta, pa = data["temp"][names.index(r)][water] - ref[water], pred[water] - ref[water]
        stats.append((np.corrcoef(ta, pa)[0, 1], ta @ pa / (ta @ ta)))           # pattern correlation, amplitude
        t, p = zonal(data["temp"][names.index(r)] - ref), zonal(pred - ref)
        sections.append((t, p, p - t))
        lim[r] = max(lim.get(r, 0.0), np.nanmax(np.abs(t)), np.nanmax(np.abs(p)))

    fig, axs = plt.subplots(len(rows), 5, figsize=(11.5, 2.7 * len(rows) + 0.3), squeeze=False,
                            gridspec_kw=dict(width_ratios=[0.6, 0.12, 1, 1, 1], hspace=0.5))
    for ax in axs[:, 1]:
        ax.set_visible(False)                                                      # spacer for the depth labels
    axs = axs[:, [0, 2, 3, 4]]
    levels = [14, 11, 8, 6, 4, 2, 0]                                               # depth labels, levels drawn equal
    for (s, r), (t, p, e), (corr, amp), a in zip(rows, sections, stats, axs):
        held = {names.index(n) for n in rmse[(s, "mlp")]}
        tr = np.array([j for j in range(len(names)) if j not in held]); ho = np.array(sorted(held))
        j = names.index(r)
        a[0].scatter(ieps[ho], ick[ho], s=12, facecolor="none", edgecolor=AXIS, linewidth=0.8)
        a[0].scatter(ieps[tr], ick[tr], s=12, color=MUTED)
        a[0].scatter(ieps[j], ick[j], s=90, marker="*", color=STYLE["mlp"][1], edgecolor=SURFACE, linewidth=0.5, zorder=3)
        a[0].set_aspect("equal"); a[0].set_xlim(-0.8, len(epss) - 0.2); a[0].set_ylim(-0.8, len(cks) - 0.2)
        a[0].set_xticks([0, 3, 6, 9], [f"{x:.3g}" for x in epss[[0, 3, 6, 9]]], fontsize=7)
        a[0].set_yticks([0, 3, 6, 9], [f"{x:.3g}" for x in cks[[0, 3, 6, 9]]], fontsize=7)
        a[0].set_xlabel("c_eps", fontsize=8); a[0].set_ylabel("c_k", fontsize=8); a[0].tick_params(length=0)
        a[0].set_title(f"split: {LABEL[s]}\n{len(tr)} training runs (grey)", fontsize=8.5)
        norm = TwoSlopeNorm(0.0, -lim[r], lim[r])
        titles = ("true anomaly\n(zonal mean)", f"network\ncorrelation {corr:.2f}, amplitude {amp:.2f}",
                  f"network − truth\nRMSE {rmse[(s, 'mlp')][r]:.3f} K")
        for ax, sec, title in zip(a[1:], (t, p, e), titles):
            im = ax.pcolormesh(np.arange(sec.shape[1] + 1) - 0.5, np.arange(len(zt) + 1) - 0.5, sec, cmap=DIVERGING, norm=norm)
            ax.set_title(title); ax.tick_params(length=0)
            ax.set_yticks(levels, [f"{-zt[k]:.0f}" for k in levels] if ax is a[1] else [])
        a[1].set_ylabel("depth (m)")
        cb = fig.colorbar(im, ax=list(a[1:]), shrink=0.9, pad=0.012); cb.set_label("K"); cb.outline.set_visible(False)
    for ax in axs[-1, 1:]:
        ax.set_xlabel("y index (south → north)")
    fig.savefig(out / "examples.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/veros_acc_lastyear.npz")
    p.add_argument("--results", default="det/results")
    a = p.parse_args(argv)
    results = Path(a.results)
    data = np.load(a.data)
    rmse, levels = load(results)
    fig_data(data, results)
    fig_splits(rmse, results)
    fig_profiles(levels, results)
    fig_examples(data, rmse, results, results)
    print(f"figures in {results}")


if __name__ == "__main__":
    main()
