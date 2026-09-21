"""Evaluate generated hold-out states against the truth and simple baselines.

    python evaluate.py --data-file data/veros_acc_TS.npz --samples runs/full/samples/holdout_n8_s1000_ema.npz \
                       [--grid-samples runs/full/samples/grid_n4_s1000_ema.npz] [--out-dir runs/full/eval]

Metrics are computed on water cells only. Baselines: the mean training state, the nearest training run in
standardised log-parameter space, the average of the axis-neighbour training runs on the parameter grid, and
linear interpolation in log c_k between the nearest training rows below and above in the same c_eps column
(one-sided, i.e. the nearest row, when training rows exist on one side only, as in an extrapolation split).
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from wmetrics import floor_w1, level_thickness, profile_means, w1_point, w1_samples, weighted_sum

SALT_REF = 35.0


def rmse(a, b, w):
    return float(np.sqrt(np.mean((a[..., w] - b[..., w]) ** 2)))


def temp_inversion_fraction(T, water, tol=1e-3):
    """Fraction of water interfaces where temperature decreases upward (Z index 0 = bottom).
    NOT an absolute stability test: the true states show inversions in the cold southern region (pressure
    effects in the equation of state), so compare the generated fraction with the truth's."""
    both = water[:-1] & water[1:]
    dT = T[..., 1:, :, :] - T[..., :-1, :, :]                 # upper minus lower
    viol = (dT < -tol) & both
    return float(viol.sum() / (both.sum() * (T.size / T.shape[-3:][0] / T.shape[-2] / T.shape[-1])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--grid-samples", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over all-NaN land columns is expected
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.samples).parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(a.data_file, allow_pickle=False)
    gs = np.load(a.samples, allow_pickle=False)
    water = ~np.asarray(ds["mask_land"]); zt = np.asarray(ds["zt"])
    run_id = np.asarray(ds["run_id"]); run_ck, run_eps = np.asarray(ds["run_ck"]), np.asarray(ds["run_eps"])
    run_names = [str(r) for r in ds["run_names"]]
    train_runs = set(np.asarray(ds["train_runs"]).tolist())
    cks, epss = np.unique(run_ck), np.unique(run_eps)
    ick, ieps = np.searchsorted(cks, run_ck), np.searchsorted(epss, run_eps)

    # truth time-mean / time-std per run (temperature), computed once
    temp_all = ds["temp"]
    tmean = np.stack([temp_all[run_id == r].mean(0) for r in range(len(run_ck))])         # (R, Z, Y, X)
    tstd = np.stack([temp_all[run_id == r].std(0) for r in range(len(run_ck))])
    train_mean_state = tmean[sorted(train_runs)].mean(0)
    # horizontal-mean temperature profiles of every snapshot, per run, time-ordered  -> {r: (n_t, Z)}
    time_s = np.asarray(ds["time_s"]); dz = level_thickness(zt)
    prof_truth = {}
    for r in range(len(run_ck)):
        sel = np.where(run_id == r)[0]; sel = sel[np.argsort(time_s[sel])]
        prof_truth[r] = profile_means(temp_all[sel], water)
    prof = lambda field: profile_means(field[None], water)[0]                      # (Z, Y, X) -> (Z,)
    rng = np.random.default_rng(0)
    def floor_n(r, n, draws=100):
        """Finite-sample floor: n random true snapshots against all snapshots, mean over draws -> (Z,)."""
        tp = prof_truth[r]
        return np.mean([w1_samples(tp[rng.choice(len(tp), n, replace=False)], tp) for _ in range(draws)], 0)
    lc, le = np.log(run_ck), np.log(run_eps)
    zc = np.stack([(lc - lc.mean()) / (lc.std() + 1e-12), (le - le.mean()) / (le.std() + 1e-12)], 1)

    def nearest_train(r):
        d = np.linalg.norm(zc - zc[r], axis=1); d[r] = np.inf
        d[[i for i in range(len(d)) if i not in train_runs]] = np.inf
        return int(np.argmin(d))

    def neighbour_avg(r):
        nb = [i for i in range(len(run_ck)) if i in train_runs and
              ((ick[i] == ick[r] and abs(ieps[i] - ieps[r]) == 1) or (ieps[i] == ieps[r] and abs(ick[i] - ick[r]) == 1))]
        return (tmean[nb].mean(0), len(nb)) if nb else (tmean[nearest_train(r)], 0)

    def interp_ck(r):
        """Linear interpolation in log c_k between the nearest training rows below and above, same c_eps column."""
        col = [i for i in range(len(run_ck)) if i in train_runs and ieps[i] == ieps[r]]
        below = [i for i in col if ick[i] < ick[r]]; above = [i for i in col if ick[i] > ick[r]]
        if below and above:
            b = max(below, key=lambda i: ick[i]); a_ = min(above, key=lambda i: ick[i])
            w = (lc[r] - lc[b]) / (lc[a_] - lc[b])
            return (1 - w) * tmean[b] + w * tmean[a_], 0
        i = max(below, key=lambda i: ick[i]) if below else min(above, key=lambda i: ick[i])
        return tmean[i], 1

    rows = []
    def add(run, method, metric, value):
        rows.append((run_names[run], method, metric, float(value)))

    gen_rid = np.asarray(gs["run_id"]); gT = gs["temp"]; gS = gs["salt"]                # (n_cond, n_s, Z, Y, X)
    lvl_rmse = {"diffusion_mean": [], "interp_ck": [], "neighbour_avg": [], "nearest_train": [], "train_mean": []}
    w1_levels = {}
    for k, r in enumerate(gen_rid):
        r = int(r)
        truth = tmean[r]; gen = np.nan_to_num(gT[k]); ens = gen.mean(0)
        nb_state, n_nb = neighbour_avg(r); nt_state = tmean[nearest_train(r)]; ip_state, one_sided = interp_ck(r)
        cands = {"diffusion_mean": ens, "interp_ck": ip_state, "neighbour_avg": nb_state, "nearest_train": nt_state,
                 "train_mean": train_mean_state}
        for m, st in cands.items():
            add(r, m, "rmse_K", rmse(st, truth, water))
            add(r, m, "bias_domain_mean_K", st[water].mean() - truth[water].mean())
            lvl_rmse[m].append([rmse(st[z], truth[z], water[z]) if water[z].any() else np.nan for z in range(len(zt))])
        add(r, "diffusion_sample", "rmse_K", np.mean([rmse(g, truth, water) for g in gen]))
        add(r, "diffusion", "spread_K", gen.std(0)[water].mean())
        add(r, "truth", "spread_K", tstd[r][water].mean())
        add(r, "diffusion", "salt_max_abs_err", np.nanmax(np.abs(gS[k][:, water] - SALT_REF)))
        add(r, "diffusion", "temp_inversion_frac", np.mean([temp_inversion_fraction(g, water) for g in gen]))
        add(r, "truth", "temp_inversion_frac", temp_inversion_fraction(truth, water))
        add(r, "neighbour_avg", "n_neighbours", n_nb)
        add(r, "interp_ck", "one_sided", one_sided)
        # Wasserstein-1 on horizontal-mean profiles, per level, thickness-weighted sum
        tp = prof_truth[r]; gp = profile_means(gen, water)
        W = {"diffusion": w1_samples(gp, tp), "interp_ck": w1_point(prof(ip_state), tp),
             "neighbour_avg": w1_point(prof(nb_state), tp), "nearest_train": w1_point(prof(nt_state), tp),
             "train_mean": w1_point(prof(train_mean_state), tp),
             "floor_n": floor_n(r, gp.shape[0]), "drift_halves": floor_w1(tp)}
        for m_, v in W.items():
            add(r, m_, "w1_profile_K", weighted_sum(v, dz)); w1_levels.setdefault(m_, []).append(v)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "method", "metric", "value"]); w.writerows(rows)

    # ---- summary
    def agg(method, metric):
        v = [x[3] for x in rows if x[1] == method and x[2] == metric]
        return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
    lines = [f"hold-out runs: {len(gen_rid)} | samples/run: {gT.shape[1]} | norm_mode={gs['norm_mode']} | weights={gs['weights']} | steps={int(gs['steps'])}",
             "", f"{'method':<18}{'RMSE(K) mean±std':>22}{'|bias|(K)':>12}"]
    for m in ["diffusion_mean", "diffusion_sample", "interp_ck", "neighbour_avg", "nearest_train", "train_mean"]:
        mu, sd = agg(m, "rmse_K"); bias = np.mean([abs(x[3]) for x in rows if x[1] == m and x[2] == "bias_domain_mean_K"]) if m != "diffusion_sample" else np.nan
        lines.append(f"{m:<18}{mu:>14.4f} ± {sd:<6.4f}{bias:>12.4f}")
    lines += [f"W1 on horizontal-mean T profiles, thickness-weighted (K): "
              + " | ".join(f"{m_} {agg(m_, 'w1_profile_K')[0]:.3f}" for m_ in ["diffusion", "interp_ck", "neighbour_avg", "nearest_train", "train_mean"])
              + f" | floor (n of 241 true) {agg('floor_n', 'w1_profile_K')[0]:.3f} | drift (half vs half) {agg('drift_halves', 'w1_profile_K')[0]:.3f}"]
    lines += [f"interp_ck one-sided (extrapolation) in {int(sum(x[3] for x in rows if x[1] == 'interp_ck' and x[2] == 'one_sided'))} of {len(gen_rid)} runs; "
              f"neighbour_avg used {agg('neighbour_avg', 'n_neighbours')[0]:.1f} training neighbours on average",
              "", f"ensemble spread (K): generated {agg('diffusion', 'spread_K')[0]:.4f} vs truth-in-window {agg('truth', 'spread_K')[0]:.4f}",
              f"interfaces with T decreasing upward: generated {100 * agg('diffusion', 'temp_inversion_frac')[0]:.3f}% vs truth {100 * agg('truth', 'temp_inversion_frac')[0]:.3f}% (truth has real inversions; compare, do not expect 0)",
              f"salinity max |S-35| over water: {agg('diffusion', 'salt_max_abs_err')[0]:.2e}"]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prof = {m: np.nanmean(np.array(v), 0) for m, v in lvl_rmse.items()}
    fig, ax = plt.subplots(figsize=(5, 6))
    for m, v in prof.items():
        ax.plot(v, zt, marker="o", ms=3, label=m)
    ax.set_xlabel("RMSE (K), mean over hold-out runs"); ax.set_ylabel("depth (m)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "rmse_profile.png", dpi=120); plt.close(fig)

    with open(out_dir / "w1_profile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "z_m", "w1_K_mean_over_runs"])
        for m_, v in w1_levels.items():
            for z, val in zip(zt, np.mean(v, 0)):
                w.writerow([m_, f"{z:.0f}", f"{val:.5f}"])
    fig, ax = plt.subplots(figsize=(5, 6))
    for m_, v in w1_levels.items():
        style = dict(ls="--", color="k") if m_ == "floor_n" else dict(ls=":", color="gray") if m_ == "drift_halves" else dict(marker="o", ms=3)
        ax.plot(np.mean(v, 0), zt, label=m_, **style)
    ax.set_xlabel("W1 of horizontal-mean T (K), mean over hold-out runs"); ax.set_ylabel("depth (m)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "w1_profile.png", dpi=120); plt.close(fig)

    n = len(gen_rid); fig, axs = plt.subplots(n, 4, figsize=(17, 2.6 * n), squeeze=False)
    yy = np.arange(water.shape[1])
    for k, r in enumerate(gen_rid):
        r = int(r); truth = np.where(water, tmean[r], np.nan); ens = np.nanmean(gT[k], 0)
        nb_state, _ = neighbour_avg(r); nb_state = np.where(water, nb_state, np.nan)
        zt_ = lambda A: np.nanmean(A, axis=-1)                                   # zonal mean -> (Z, Y)
        panels = [(zt_(truth), "truth", "RdYlBu_r", (-2, 15)), (zt_(ens), "diffusion mean", "RdYlBu_r", (-2, 15)),
                  (zt_(ens) - zt_(truth), "diffusion - truth", "RdBu_r", (-0.5, 0.5)),
                  (zt_(nb_state) - zt_(truth), "neighbour avg - truth", "RdBu_r", (-0.5, 0.5))]
        for j, (A, title, cmap, (vmin, vmax)) in enumerate(panels):
            im = axs[k, j].pcolormesh(yy, zt, A, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            axs[k, j].set_title(f"{run_names[r]}  {title}", fontsize=8); plt.colorbar(im, ax=axs[k, j], fraction=0.04)
    fig.tight_layout(); fig.savefig(out_dir / "sections_holdout.png", dpi=100); plt.close(fig)

    if a.grid_samples:
        gg = np.load(a.grid_samples, allow_pickle=False); grid_rid = np.asarray(gg["run_id"])
        dm_truth = np.full((len(cks), len(epss)), np.nan); dm_gen = dm_truth.copy()
        for k, r in enumerate(grid_rid):
            dm_truth[ick[r], ieps[r]] = tmean[r][water].mean()
            dm_gen[ick[r], ieps[r]] = np.nanmean(gg["temp"][k], 0)[water].mean()
        hr = np.asarray(ds["holdout_runs"]); tr_ = np.asarray(ds["train_runs"])
        ho = np.sqrt(np.nanmean(((dm_gen - dm_truth)[ick[hr], ieps[hr]]) ** 2)) if len(hr) else float("nan")
        al = np.sqrt(np.nanmean((dm_gen - dm_truth) ** 2))
        # numbers behind the figure, so it can be re-plotted without the samples
        with open(out_dir / "grid_domain_mean.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "ck", "eps", "ick", "ieps", "holdout", "dm_truth_K", "dm_gen_K"])
            for r in grid_rid:
                w.writerow([run_names[r], run_ck[r], run_eps[r], ick[r], ieps[r], int(r in set(hr.tolist())),
                            f"{dm_truth[ick[r], ieps[r]]:.5f}", f"{dm_gen[ick[r], ieps[r]]:.5f}"])
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, A, title, cmap, lim in [(axs[0], dm_truth, "truth", "viridis", None), (axs[1], dm_gen, "generated", "viridis", None),
                                        (axs[2], dm_gen - dm_truth, "generated - truth", "RdBu_r", 0.2)]:
            kw = dict(vmin=-lim, vmax=lim) if lim else dict(vmin=np.nanmin(dm_truth), vmax=np.nanmax(dm_truth))
            im = ax.imshow(A, origin="lower", cmap=cmap, aspect="auto", **kw); ax.set_title(f"domain-mean T (°C): {title}")
            ax.set_xticks(range(len(epss))); ax.set_xticklabels([f"{e:g}" for e in epss], rotation=60, fontsize=7)
            ax.set_yticks(range(len(cks))); ax.set_yticklabels([f"{c:g}" for c in cks], fontsize=7); ax.set_xlabel("c_eps"); ax.set_ylabel("c_k")
            ax.plot(ieps[tr_], ick[tr_], "o", ms=3, mfc="white", mec="black", mew=0.6, label="training run")
            if len(hr):
                ax.plot(ieps[hr], ick[hr], "x", ms=7, mew=1.8, color="red", label="hold-out run")
            plt.colorbar(im, ax=ax, fraction=0.046)
        axs[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
        fig.suptitle(f"{gs['norm_mode']} normalisation | RMSE of domain-mean T: hold-out {ho:.3f} K, all runs {al:.3f} K")
        fig.tight_layout(); fig.savefig(out_dir / "grid_domain_mean.png", dpi=120); plt.close(fig)

        # W1 map: thickness-weighted W1 of the profile distribution per grid point, its finite-sample floor, difference
        n_g = gg["temp"].shape[1]
        gw1 = np.full((len(cks), len(epss)), np.nan); gfl = gw1.copy()
        with open(out_dir / "grid_w1.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "ck", "eps", "holdout", "w1_K", "floor_K", "n_samples"])
            for k, r in enumerate(grid_rid):
                gp = profile_means(np.nan_to_num(gg["temp"][k]), water)
                gw1[ick[r], ieps[r]] = weighted_sum(w1_samples(gp, prof_truth[r]), dz)
                gfl[ick[r], ieps[r]] = weighted_sum(floor_n(r, n_g, draws=20), dz)
                w.writerow([run_names[r], run_ck[r], run_eps[r], int(r in set(hr.tolist())), f"{gw1[ick[r], ieps[r]]:.5f}", f"{gfl[ick[r], ieps[r]]:.5f}", n_g])
        w_ho = np.nanmean(gw1[ick[hr], ieps[hr]]) if len(hr) else float("nan"); w_all = np.nanmean(gw1)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.8)); vmax = np.nanmax(gw1)
        for ax, A, title, cmap, lim in [(axs[0], gw1, f"W1 diffusion vs truth ({n_g} samples)", "viridis", None),
                                        (axs[1], gfl, f"floor: {n_g} true snapshots vs all", "viridis", None),
                                        (axs[2], gw1 - gfl, "diffusion - floor", "RdBu_r", 0.2)]:
            kw = dict(vmin=-lim, vmax=lim) if lim else dict(vmin=0, vmax=vmax)
            im = ax.imshow(A, origin="lower", cmap=cmap, aspect="auto", **kw); ax.set_title(f"{title} (K)")
            ax.set_xticks(range(len(epss))); ax.set_xticklabels([f"{e:g}" for e in epss], rotation=60, fontsize=7)
            ax.set_yticks(range(len(cks))); ax.set_yticklabels([f"{c:g}" for c in cks], fontsize=7); ax.set_xlabel("c_eps"); ax.set_ylabel("c_k")
            ax.plot(ieps[tr_], ick[tr_], "o", ms=3, mfc="white", mec="black", mew=0.6)
            if len(hr):
                ax.plot(ieps[hr], ick[hr], "x", ms=7, mew=1.8, color="red")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{gs['norm_mode']} normalisation | thickness-weighted W1 of horizontal-mean T profiles: hold-out {w_ho:.3f} K, all runs {w_all:.3f} K")
        fig.tight_layout(); fig.savefig(out_dir / "grid_w1.png", dpi=120); plt.close(fig)
        print(f"grid W1: mean over all runs {w_all:.4f} K, hold-out only {w_ho:.4f} K")
        print(f"grid map: RMSE of domain-mean T over all runs {al:.4f} K, hold-out only {ho:.4f} K")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
