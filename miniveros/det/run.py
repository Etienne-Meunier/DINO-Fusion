"""Evaluate the emulator and the baselines on the hold-out splits and on leave-one-out over all runs.

    python -m det.run [--data data/veros_acc_lastyear.npz] [--out det/results] [--splits scattered,band,...,loo]

The network's hyper-parameters (activation, L2 weight) are chosen on each split by repeated 5-fold
cross-validation over its training runs only, with the one-standard-error rule (methods.select); for
leave-one-out they are chosen once, on all runs. The network is then fitted with SEEDS.size sets of initialisations
(5 members each) to measure its sensitivity to them; the baselines are deterministic.
Writes to --out: metrics.csv (one row per split, method, held-out run and, for the network, seed set), levels.csv
(RMSE by depth), selected.csv (the chosen hyper-parameters and the cross-validation scores), summary.md, and
examples.npz (the predictions of a few runs, first seed set, for the figures).
"""
from __future__ import annotations

import argparse
import csv
import os
import time as clock
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")        # lbfgs / GP convergence warnings, also in joblib workers

import numpy as np
from joblib import Parallel, delayed

from det.methods import GP, MLP, RBF, Nearest, TrainMean, select
from extract_data import choose_holdout

SPLITS = {                    # the eight hold-out splits, as choose_holdout arguments
    "scattered": dict(mode="interior_random", n=10, seed=0),
    "band": dict(mode="rows", rows=[0.126, 0.2, 0.3175]),
    "centre5": dict(mode="block", block=(3, 8, 3, 8)),
    "centre7": dict(mode="block", block=(2, 9, 2, 9)),
    "outer51": dict(mode="ring", block=(2, 9, 2, 9)),
    "outer75": dict(mode="ring", block=(3, 8, 3, 8)),
    "top": dict(mode="row_ck_max"),
    "four": dict(mode="points", points=(1, 1, 1, 8, 8, 1, 8, 8)),
}
GRID = [dict(activation=a, alpha=al) for a in ("relu", "tanh") for al in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
SEEDS = (0, 100, 200, 300)    # initialisation sets of the network (random_state = seed + member)
BASELINES = {"gp_s": lambda: GP(on="s"), "gp": GP, "rbf": RBF, "nearest": Nearest, "mean": TrainMean}
METHODS = ["mlp", *BASELINES]
EXAMPLES = {"centre7": ["ck0.126_eps0.8819"], "top": ["ck0.8_eps0.0875"], "outer75": ["ck0.8_eps0.0875"],
            "four": ["ck0.126_eps0.8819", "ck0.8_eps0.0875"]}                        # predictions kept for the figures


def fit_predict(tr, te, Z, Y, mlp_params):
    """Every method fitted on runs tr, predicting runs te: {method: (len(te), cells)} for the baselines and
    {("mlp", seed): ...} for the network, plus the network's spread over members per seed set."""
    pred = {k: make().fit(Z[tr], Y[tr]).predict(Z[te]) for k, make in BASELINES.items()}
    spread = {}
    for sd in SEEDS:
        pred[("mlp", sd)], spread[sd] = MLP(**mlp_params, seed=sd).fit(Z[tr], Y[tr]).predict(Z[te], spread=True)
    return pred, spread


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/veros_acc_lastyear.npz")
    p.add_argument("--out", default="det/results")
    p.add_argument("--splits", default=",".join([*SPLITS, "loo"]))
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.data)
    water = ~d["mask_land"]; zt = d["zt"]
    Y = d["temp"][:, water].astype(np.float64)                                       # (100, cells)
    level = np.broadcast_to(np.arange(len(zt))[:, None, None], water.shape)[water]   # level of every water cell
    run_ck, run_eps, names = d["run_ck"], d["run_eps"], [str(n) for n in d["run_names"]]
    logp = np.c_[np.log(run_ck), np.log(run_eps)]
    Z = (logp - logp.mean(0)) / logp.std(0)          # the parameter design is known: grid statistics, no leakage
    R = len(Y)

    rows, lvl_rows, sel_rows, examples = [], [], [], {}
    for name in a.splits.split(","):
        t0 = clock.time()
        if name == "loo":
            params, scores, se = select(MLP, GRID, Z, Y)
            res = Parallel(n_jobs=-1)(delayed(fit_predict)(np.setdiff1d(np.arange(R), [r]), [r], Z, Y, params)
                                      for r in range(R))
            ho = np.arange(R)
            pred = {k: np.concatenate([q[k] for q, _ in res]) for k in res[0][0]}
            spread = {sd: np.concatenate([s[sd] for _, s in res]) for sd in SEEDS}
        else:
            s = SPLITS[name]
            ho = np.asarray(choose_holdout(run_ck, run_eps, s["mode"], s.get("n", 10), s.get("seed", 0),
                                           s.get("rows"), s.get("block"), s.get("points", ())))
            tr = np.setdiff1d(np.arange(R), ho)
            params, scores, se = select(MLP, GRID, Z[tr], Y[tr])
            pred, spread = fit_predict(tr, ho, Z, Y, params)
        n_tr = R - 1 if name == "loo" else R - len(ho)
        sel_rows.append([name, n_tr, params["activation"], params["alpha"], round(se, 5), *np.round(scores, 5)])
        for key, P in pred.items():
            m, sd = key if isinstance(key, tuple) else (key, "")
            err = P - Y[ho]
            for i, r in enumerate(ho):
                rows.append([name, m, sd, names[r], run_ck[r], run_eps[r], np.sqrt((err[i] ** 2).mean()), err[i].mean(),
                             spread[sd][i] if m == "mlp" else np.nan])
        for m in METHODS:                         # RMSE by depth: per run, the network averaged over its seed sets
            Ps = [pred[("mlp", sd)] for sd in SEEDS] if m == "mlp" else [pred[m]]
            for z in range(len(zt)):
                lvl = np.mean([np.sqrt(((P - Y[ho])[:, level == z] ** 2).mean(1)) for P in Ps], axis=0)
                lvl_rows.append([name, m, zt[z], lvl.mean()])
        for r in EXAMPLES.get(name, []):
            i = list(ho).index(names.index(r))
            examples.update({f"{name}/{r}/{m}": pred[("mlp", SEEDS[0]) if m == "mlp" else m][i].astype(np.float32)
                             for m in METHODS})
        e = {m: np.mean([x[6] for x in rows if x[0] == name and x[1] == m]) for m in METHODS}
        print(f"{name:9s} {len(ho):3d} held out | mlp {params['activation']} alpha {params['alpha']:g} | "
              + " ".join(f"{m} {v:.3f}" for m, v in e.items()) + f" | {clock.time() - t0:.0f} s", flush=True)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "seed", "run", "ck", "eps", "rmse_K", "bias_K", "spread_K"]); w.writerows(rows)
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "method", "z_m", "rmse_K"]); w.writerows(lvl_rows)
    with open(out / "selected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_train", "activation", "alpha", "cv_se",
                    *[f"cv_{g['activation']}_{g['alpha']:g}" for g in GRID]])
        w.writerows(sel_rows)
    np.savez(out / "examples.npz", **examples)
    write_summary(out, rows, sel_rows, names[int(np.argmax(np.log(run_ck) - np.log(run_eps)))])


def per_run(rows, split, method):
    """{run: RMSE} of a method on a split; for the network, the mean over its seed sets."""
    acc = {}
    for r in rows:
        if r[0] == split and r[1] == method:
            acc.setdefault(r[3], []).append(r[6])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(a, b, n_boot=5000):
    """Mean of the per-run differences a - b, its 95 % bootstrap interval, the Wilcoxon signed-rank p-value and the
    number of runs where a < b."""
    from scipy.stats import wilcoxon
    k = sorted(a); d = np.array([a[r] - b[r] for r in k])
    rng = np.random.default_rng(0)
    boot = rng.choice(d, (n_boot, len(d))).mean(1)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), wilcoxon(d).pvalue, int((d < 0).sum()), len(d)


def write_summary(out, rows, sel_rows, corner):
    splits = list(dict.fromkeys(r[0] for r in rows))
    split_means = lambda s: [np.mean([r[6] for r in rows if r[0] == s and r[1] == "mlp" and r[2] == sd]) for sd in SEEDS]

    def table(title, stat):
        lines = [f"**{title}**", "", "| split | held out | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for s in splits:
            v = {m: stat(list(per_run(rows, s, m).values())) for m in METHODS}
            best = min(v, key=v.get)
            lines.append(f"| {s} | {len(per_run(rows, s, 'mlp'))} | "
                         + " | ".join((f"**{v[m]:.3f}**" if m == best else f"{v[m]:.3f}") for m in METHODS) + " |")
        return lines + [""]

    lines = ["# Results (last-year mean T, water cells, K)", "",
             f"Network: per-run RMSE averaged over {len(SEEDS)} initialisation sets of 5 members.", ""]
    lines += table("Mean RMSE over the held-out runs", np.mean) + table("Median RMSE over the held-out runs", np.median)
    lines += ["**Network against the GP on s** (per-run RMSE difference, network minus GP; negative = network better)", "",
              "| split | network mean (range over seed sets) | GP on s | difference | 95 % bootstrap interval | Wilcoxon p | network wins |",
              "|---|---|---|---|---|---|---|"]
    for s in splits:
        mlp, gps = per_run(rows, s, "mlp"), per_run(rows, s, "gp_s")
        sm = split_means(s); dm, lo, hi, pv, wins, n = paired(mlp, gps)
        lines.append(f"| {s} | {np.mean(sm):.3f} ({min(sm):.3f}-{max(sm):.3f}) | {np.mean(list(gps.values())):.3f} | "
                     f"{dm:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pv:.3f} | {wins}/{n} |")
    lines += ["", f"**Corner run {corner}** (when held out)", "", "| split | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for s in splits:
        if corner in per_run(rows, s, "mlp"):
            lines.append(f"| {s} | " + " | ".join(f"{per_run(rows, s, m)[corner]:.3f}" for m in METHODS) + " |")
    lines += ["", "**Network: chosen hyper-parameters and ensemble spread**", "",
              "| split | training runs | activation | alpha | CV RMSE (chosen) | best CV RMSE | standard error | spread |",
              "|---|---|---|---|---|---|---|---|"]
    keys = [(g["activation"], g["alpha"]) for g in GRID]
    for (s, n_tr, act, alpha, se, *scores) in sel_rows:
        spread = np.mean([r[8] for r in rows if r[0] == s and r[1] == "mlp"])
        lines.append(f"| {s} | {n_tr} | {act} | {alpha:g} | {scores[keys.index((act, alpha))]:.4f} | {min(scores):.4f} | "
                     f"{se:.4f} | {spread:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
