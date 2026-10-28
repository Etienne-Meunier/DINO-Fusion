"""Evaluate the emulator and the baselines on the hold-out splits and on leave-one-out over all runs.

    python -m det.run [--data data/veros_acc_lastyear.npz] [--out det/results] [--splits scattered,band,...,loo]

The network's hyper-parameters (activation, L2 weight) are chosen on each split by 5-fold cross-validation over
its training runs only; for leave-one-out they are chosen once, by 5-fold cross-validation over all runs.
Writes to --out: metrics.csv (one row per split, method and held-out run), levels.csv (RMSE by depth),
selected.csv (the chosen hyper-parameters and the cross-validation scores), summary.md, examples.npz (the
predictions of a few runs, for the figures).
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
BASELINES = {"gp_s": lambda: GP(on="s"), "gp": GP, "rbf": RBF, "nearest": Nearest, "mean": TrainMean}
METHODS = ["mlp", *BASELINES]
EXAMPLES = {"centre7": ["ck0.126_eps0.8819"], "outer75": ["ck0.8_eps0.0875"]}   # predictions kept for the figures


def fit_predict(tr, te, Z, Y, mlp_params):
    """Every method fitted on runs tr, predicting runs te: {method: (len(te), cells)}, and the network's spread."""
    pred = {k: make().fit(Z[tr], Y[tr]).predict(Z[te]) for k, make in BASELINES.items()}
    pred["mlp"], spread = MLP(**mlp_params).fit(Z[tr], Y[tr]).predict(Z[te], spread=True)
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
            params, scores = select(MLP, GRID, Z, Y)
            res = Parallel(n_jobs=-1)(delayed(fit_predict)(np.setdiff1d(np.arange(R), [r]), [r], Z, Y, params)
                                      for r in range(R))
            ho = np.arange(R)
            pred = {m: np.concatenate([q[m] for q, _ in res]) for m in METHODS}
            spread = np.concatenate([s for _, s in res])
        else:
            s = SPLITS[name]
            ho = np.asarray(choose_holdout(run_ck, run_eps, s["mode"], s.get("n", 10), s.get("seed", 0),
                                           s.get("rows"), s.get("block"), s.get("points", ())))
            tr = np.setdiff1d(np.arange(R), ho)
            params, scores = select(MLP, GRID, Z[tr], Y[tr])
            pred, spread = fit_predict(tr, ho, Z, Y, params)
        n_tr = R - 1 if name == "loo" else R - len(ho)
        sel_rows.append([name, n_tr, params["activation"], params["alpha"], *np.round(scores, 5)])
        for m in METHODS:
            err = pred[m] - Y[ho]
            for i, r in enumerate(ho):
                rows.append([name, m, names[r], run_ck[r], run_eps[r], np.sqrt((err[i] ** 2).mean()), err[i].mean(),
                             spread[i] if m == "mlp" else np.nan])
            for z in range(len(zt)):
                lvl_rows.append([name, m, zt[z], np.sqrt((err[:, level == z] ** 2).mean(1)).mean()])
        for r in EXAMPLES.get(name, []):
            i = list(ho).index(names.index(r))
            examples.update({f"{name}/{r}/{m}": pred[m][i].astype(np.float32) for m in METHODS})
        e = {m: np.mean([x[5] for x in rows if x[0] == name and x[1] == m]) for m in METHODS}
        print(f"{name:9s} {len(ho):3d} held out | mlp {params['activation']} alpha {params['alpha']:g} | "
              + " ".join(f"{m} {v:.3f}" for m, v in e.items()) + f" | {clock.time() - t0:.0f} s", flush=True)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "method", "run", "ck", "eps", "rmse_K", "bias_K", "spread_K"]); w.writerows(rows)
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "method", "z_m", "rmse_K"]); w.writerows(lvl_rows)
    with open(out / "selected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_train", "activation", "alpha", *[f"cv_{g['activation']}_{g['alpha']:g}" for g in GRID]])
        w.writerows(sel_rows)
    np.savez(out / "examples.npz", **examples)
    write_summary(out, rows, sel_rows, names[int(np.argmax(np.log(run_ck) - np.log(run_eps)))])


def write_summary(out, rows, sel_rows, corner):
    splits = list(dict.fromkeys(r[0] for r in rows))
    get = lambda s, m, col: np.array([r[col] for r in rows if r[0] == s and r[1] == m])

    def table(title, stat):
        lines = [f"**{title}**", "", "| split | held out | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for s in splits:
            v = {m: stat(get(s, m, 5)) for m in METHODS}
            best = min(v, key=v.get)
            lines.append(f"| {s} | {len(get(s, 'mlp', 5))} | "
                         + " | ".join((f"**{v[m]:.3f}**" if m == best else f"{v[m]:.3f}") for m in METHODS) + " |")
        return lines + [""]

    lines = ["# Results (last-year mean T, water cells, K)", ""]
    lines += table("Mean RMSE over the held-out runs", np.mean) + table("Median RMSE over the held-out runs", np.median)
    lines += [f"**Corner run {corner}** (when held out)", "", "| split | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for s in splits:
        runs = [r for r in rows if r[0] == s and r[2] == corner]
        if runs:
            lines.append(f"| {s} | " + " | ".join(f"{next(r[5] for r in runs if r[1] == m):.3f}" for m in METHODS) + " |")
    lines += ["", "**Network: chosen hyper-parameters, wins and ensemble spread**", "",
              "| split | training runs | activation | alpha | CV RMSE | wins vs nearest | wins vs gp_s | spread | mean RMSE |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (s, n_tr, act, alpha, *scores) in sel_rows:
        e, en, eg = get(s, "mlp", 5), get(s, "nearest", 5), get(s, "gp_s", 5)
        lines.append(f"| {s} | {n_tr} | {act} | {alpha:g} | {min(scores):.3f} | {(e < en).sum()}/{len(e)} | "
                     f"{(e < eg).sum()}/{len(e)} | {np.mean(get(s, 'mlp', 7)):.3f} | {e.mean():.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
