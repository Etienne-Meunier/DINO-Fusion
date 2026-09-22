"""Per-level distribution of T over the whole extracted dataset (all runs, all snapshots, water cells).

  python tdist.py compute <data_file.npz> <stats.npz>        # on the cluster, ~90 s, ~2 GB of RAM
  python tdist.py plot    <stats.npz> <figure.png>           # ridge plot, raw and 3-std normalised side by side
"""
import sys, time
import numpy as np

QS = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
NBINS = 1000


def compute(src, out):
    t0 = time.time()
    d = np.load(src, allow_pickle=False)
    temp, land, zt, run_id = d["temp"], d["mask_land"], d["zt"], d["run_id"]      # temp (N, Z, Y, X), Z index 0 = bottom
    N, Z, Y, X = temp.shape
    water = ~(np.broadcast_to(land, (Z, Y, X)) if land.ndim == 2 else land).astype(bool)
    runs = np.unique(run_id); R = len(runs); T = N // R
    assert N == R * T and np.array_equal(run_id, np.repeat(runs, T)), "snapshots must be stored run by run"
    print(f"temp {temp.shape}, water cells per level {water.reshape(Z, -1).sum(1)}, load {time.time() - t0:.0f}s", flush=True)
    st = {k: np.zeros(Z) for k in ["mean", "std", "min", "max", "skew", "kurt", "frac_out3", "n", "sigma_runs", "sigma_time"]}
    quant, edges, counts = np.zeros((Z, len(QS))), np.zeros((Z, NBINS + 1)), np.zeros((Z, NBINS))
    run_level_mean = np.zeros((R, Z))
    for z in range(Z):
        v = temp[:, z][:, water[z]].astype(np.float64)                     # (N, n_water)
        x = v.ravel(); m, s = x.mean(), x.std()
        st["mean"][z], st["std"][z], st["min"][z], st["max"][z], st["n"][z] = m, s, x.min(), x.max(), x.size
        st["skew"][z] = ((x - m) ** 3).mean() / s ** 3
        st["kurt"][z] = ((x - m) ** 4).mean() / s ** 4 - 3
        st["frac_out3"][z] = (np.abs(x - m) > 3 * s).mean()
        quant[z] = np.quantile(x, QS)
        counts[z], edges[z] = np.histogram(x, bins=NBINS, range=(x.min(), x.max()))
        vr = v.reshape(R, T, -1); tm = vr.mean(1)                          # per-run time mean per cell
        run_level_mean[:, z] = tm.mean(1)
        st["sigma_runs"][z] = tm.std(0).mean()                              # run-to-run spread of the time mean, cell average
        st["sigma_time"][z] = vr.std(1).mean()                              # within-run temporal std, cell and run average
        print(f"z={zt[z]:8.1f} m  mean {m:6.3f} std {s:6.3f} [{x.min():.3f}, {x.max():.3f}] skew {st['skew'][z]:+5.2f} "
              f"out3 {100 * st['frac_out3'][z]:.2f}%  sigma_runs {st['sigma_runs'][z]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out, zt=zt, qs=QS, quant=quant, edges=edges, counts=counts, run_level_mean=run_level_mean, runs=runs,
                        **{f"stat_{k}": v for k, v in st.items()})
    print("wrote", out)


def plot(src, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(src); zt, edges, counts = d["zt"], d["edges"], d["counts"]
    mean, std, mn, mx, rlm = d["stat_mean"], d["stat_std"], d["stat_min"], d["stat_max"], d["run_level_mean"]
    Z = len(zt); order = np.argsort(-zt); gap = 1.0

    def ridge(ax, xs, tf, title, xlabel, label_right):
        for k, z in enumerate(order):
            c = edges[z][:-1] + np.diff(edges[z]) / 2; dens = counts[z] / (counts[z].sum() * np.diff(edges[z]))
            y = np.interp(xs, tf(c, z), dens, left=0, right=0); y = y / y.max() * 0.9 * gap; base = -k * gap
            ax.fill_between(xs, base, base + y, color=plt.cm.viridis(k / (Z - 1)), alpha=0.8, lw=0); ax.plot(xs, base + y, color="k", lw=0.5)
            ax.plot(tf(rlm[:, z], z), np.full(rlm.shape[0], base + 0.05), "|", color="tab:orange", ms=6, mew=0.7)
            m3 = tf(np.array([mean[z] - 3 * std[z], mean[z], mean[z] + 3 * std[z]]), z)
            ax.plot([m3[1], m3[1]], [base, base + 0.9 * gap], color="k", lw=0.8)
            for v in (m3[0], m3[2]): ax.plot([v, v], [base, base + 0.9 * gap], color="tab:red", lw=0.8, ls="--")
            ax.text(xs[0] + 0.01 * (xs[-1] - xs[0]), base + 0.35, f"{-zt[z]:.0f} m", fontsize=13, va="center")
            ax.text(xs[-1] - 0.01 * (xs[-1] - xs[0]), base + 0.35, label_right(z), fontsize=11, va="center", ha="right")
            ax.axhline(base, color="k", lw=0.4)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-(Z - 1) * gap - 0.1, 1.0); ax.set_yticks([]); ax.set_xlabel(xlabel, fontsize=14); ax.set_title(title, fontsize=15)

    plt.rcParams.update({"font.size": 13})   # legible when the figure is scaled to a page width
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 9))
    ridge(a1, np.linspace(0, 16, 1600), lambda c, z: c, "Before normalisation: T (°C)", "T (°C)",
          lambda z: f"μ {mean[z]:.2f}  σ {std[z]:.2f}  [{mn[z]:.2f}, {mx[z]:.2f}]")
    ridge(a2, np.linspace(-2.5, 3.5, 1200), lambda c, z: (c - mean[z]) / (3 * std[z]), "After normalisation: x' = (T − μ_z) / (3 σ_z)", "x'",
          lambda z: f"[{(mn[z] - mean[z]) / (3 * std[z]):+.2f}, {(mx[z] - mean[z]) / (3 * std[z]):+.2f}]")
    a2.axvline(-1, color="tab:red", ls="--", lw=1); a2.axvline(1, color="tab:red", ls="--", lw=1); a2.axvline(0, color="k", lw=0.8)
    fig.suptitle("Distribution of T per level, all 100 runs × 241 snapshots, water cells (surface at the top). "
                 "Black: μ_z; red dashed: μ_z ± 3σ_z; orange ticks: the 100 per-run level means.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=150); print("wrote", out)
    print(f"{'depth':>6} {'mean':>7} {'std':>6} {'min':>6} {'max':>6} {'skew':>6} {'out3s':>6} {'s_runs':>7} {'s_time':>7} {'s_runs/3s':>10} {'range/6s':>9}")
    for z in order:
        print(f"{-zt[z]:6.0f} {mean[z]:7.3f} {std[z]:6.3f} {mn[z]:6.2f} {mx[z]:6.2f} {d['stat_skew'][z]:+6.2f} {100 * d['stat_frac_out3'][z]:5.2f}% "
              f"{d['stat_sigma_runs'][z]:7.3f} {d['stat_sigma_time'][z]:7.3f} {100 * d['stat_sigma_runs'][z] / (3 * std[z]):9.1f}% {(mx[z] - mn[z]) / (6 * std[z]):9.2f}")


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]](sys.argv[2], sys.argv[3])
