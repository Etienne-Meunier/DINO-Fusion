"""Evaluate the emulator and the baselines on the hold-out splits and on leave-one-out over all runs.

    python -m det.run [--data data/veros_acc_lastyear.npz] [--out det/results] [--splits scattered,band,...,loo]

The network's hyper-parameters (activation, L2 weight) are chosen on each split by repeated 5-fold
cross-validation over its training runs only (methods.select); for leave-one-out they are chosen once, on all runs.
Two rules read the same scores: the one-standard-error rule gives the reported network ("mlp"), and the minimum CV
error ("mlp_min") shows how much the conclusions depend on the rule. Each choice is fitted with len(SEEDS) sets of
initialisations (5 members each) to measure the sensitivity to them; the baselines are deterministic. Every fit
runs with one BLAS thread (see methods), so its result does not depend on the process it runs in.
Writes to --out: metrics.csv (one row per split, method, held-out run and, for the network, seed set), levels.csv
(RMSE by depth), selected.csv (both rules' choices and the cross-validation scores), summary.md, and examples.npz
(the predictions of a few runs, first seed set, for the figures).
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
MAIN = [s for s in SPLITS if s != "four"] + ["loo"]      # the main comparison; four training runs is a failure case
GRID = [dict(activation=a, alpha=al) for a in ("relu", "tanh") for al in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
SEEDS = (0, 100, 200, 300)    # initialisation sets of the network (random_state = seed + member)
RULES = ("mlp", "mlp_min")    # the network chosen by the one-standard-error rule (reported) and by the minimum CV error
BASELINES = {"gp_s": lambda: GP(on="s"), "gp": GP, "rbf": RBF, "nearest": Nearest, "mean": TrainMean}
METHODS = ["mlp", *BASELINES]
EXAMPLES = {"centre7": ["ck0.126_eps0.8819"], "top": ["ck0.8_eps0.0875"], "outer75": ["ck0.8_eps0.0875"],
            "four": ["ck0.126_eps0.8819", "ck0.8_eps0.0875"]}                        # predictions kept for the figures


def fit_predict(key, tr, te, Z, Y):
    """One method fitted on runs tr, predicting runs te: a baseline name, or (activation, alpha, seed set) for the
    network, which also returns its spread over members (std over members, mean over cells, per run)."""
    if key in BASELINES:
        return BASELINES[key]().fit(Z[tr], Y[tr]).predict(Z[te]), None
    act, alpha, sd = key
    return MLP(activation=act, alpha=alpha, seed=sd).fit(Z[tr], Y[tr]).predict(Z[te], spread=True)


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
            ho = np.arange(R)
            folds = [(np.setdiff1d(ho, [r]), ho[[r]]) for r in ho]
            cv = ho                                                   # the runs the hyper-parameters are chosen on
        else:
            s = SPLITS[name]
            ho = np.asarray(choose_holdout(run_ck, run_eps, s["mode"], s.get("n", 10), s.get("seed", 0),
                                           s.get("rows"), s.get("block"), s.get("points", ())))
            folds = [(np.setdiff1d(np.arange(R), ho), ho)]
            cv = folds[0][0]
        params, scores, se = select(MLP, GRID, Z[cv], Y[cv])
        nets = {"mlp": params, "mlp_min": GRID[int(np.argmin(scores))]}
        net = lambda m, sd: (nets[m]["activation"], nets[m]["alpha"], sd)
        keys = [*BASELINES, *dict.fromkeys(net(m, sd) for m in RULES for sd in SEEDS)]   # a shared choice is fitted once
        res = Parallel(n_jobs=-1)(delayed(fit_predict)(k, tr, te, Z, Y) for tr, te in folds for k in keys)
        got = lambda k, i: np.concatenate([res[f * len(keys) + keys.index(k)][i] for f in range(len(folds))])
        pred, spread = {m: got(m, 0) for m in BASELINES}, {}
        for m in RULES:
            for sd in SEEDS:
                pred[(m, sd)], spread[(m, sd)] = got(net(m, sd), 0), got(net(m, sd), 1)
        sel_rows.append([name, len(folds[0][0]), params["activation"], params["alpha"], nets["mlp_min"]["activation"],
                         nets["mlp_min"]["alpha"], se, *scores])                        # full precision
        for key, P in pred.items():
            m, sd = key if isinstance(key, tuple) else (key, "")
            err = P - Y[ho]
            for i, r in enumerate(ho):
                rows.append([name, m, sd, names[r], run_ck[r], run_eps[r], np.sqrt((err[i] ** 2).mean()), err[i].mean(),
                             spread[key][i] if m in RULES else np.nan])
        for m in METHODS:                         # RMSE by depth: per run, the network averaged over its seed sets
            Ps = [pred[("mlp", sd)] for sd in SEEDS] if m == "mlp" else [pred[m]]
            for z in range(len(zt)):
                lvl = np.mean([np.sqrt(((P - Y[ho])[:, level == z] ** 2).mean(1)) for P in Ps], axis=0)
                lvl_rows.append([name, m, zt[z], lvl.mean()])
        for r in EXAMPLES.get(name, []):
            i = list(ho).index(names.index(r))
            examples.update({f"{name}/{r}/{m}": pred[("mlp", SEEDS[0]) if m == "mlp" else m][i].astype(np.float32)
                             for m in METHODS})
        e = {m: np.mean([x[6] for x in rows if x[0] == name and x[1] == m]) for m in [*RULES, *BASELINES]}
        print(f"{name:9s} {len(ho):3d} held out | " + ", ".join(f"{m} {g['activation']} {g['alpha']:g}" for m, g in nets.items())
              + " | " + " ".join(f"{m} {v:.4f}" for m, v in e.items()) + f" | {clock.time() - t0:.0f} s", flush=True)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "seed", "run", "ck", "eps", "rmse_K", "bias_K", "spread_K"]); w.writerows(rows)
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "method", "z_m", "rmse_K"]); w.writerows(lvl_rows)
    with open(out / "selected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_train", "activation", "alpha", "activation_min", "alpha_min", "cv_se",
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


def holm(p):
    """Holm-adjusted p-values of a family of tests."""
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p)
    adj[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return adj


def write_summary(out, rows, sel_rows, corner):
    splits = list(dict.fromkeys(r[0] for r in rows))
    split_means = lambda s, m: [np.mean([r[6] for r in rows if r[0] == s and r[1] == m and r[2] == sd]) for sd in SEEDS]
    mean_of = lambda s, m: np.mean(list(per_run(rows, s, m).values()))
    fmt_p = lambda p: "< 0.001" if p < 1e-3 else f"{p:.3f}"
    config = {r[0]: {"mlp": f"{r[2]} {r[3]:g}", "mlp_min": f"{r[4]} {r[5]:g}"} for r in sel_rows}

    def table(title, stat):
        lines = [f"**{title}**", "", "| split | held out | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
        for s in splits:
            v = {m: stat(list(per_run(rows, s, m).values())) for m in METHODS}
            best = min(v, key=v.get)
            lines.append(f"| {s} | {len(per_run(rows, s, 'mlp'))} | "
                         + " | ".join((f"**{v[m]:.3f}**" if m == best else f"{v[m]:.3f}") for m in METHODS) + " |")
        return lines + [""]

    lines = ["# Results (last-year mean T, water cells, K)", "",
             f"Network: per-run RMSE averaged over {len(SEEDS)} initialisation sets of 5 members; hyper-parameters "
             "by the one-standard-error rule unless stated.", ""]
    lines += table("Mean RMSE over the held-out runs", np.mean) + table("Median RMSE over the held-out runs", np.median)
    lines += ["Network mean RMSE / nearest run's: " + ", ".join(f"{s} {mean_of(s, 'mlp') / mean_of(s, 'nearest'):.2f}"
                                                                for s in splits), ""]
    for m, title in (("mlp", "one-standard-error rule, reported"), ("mlp_min", "minimum CV error")):
        res = {s: paired(per_run(rows, s, m), per_run(rows, s, "gp_s")) for s in splits}
        fam = [s for s in splits if s in MAIN]
        adj = dict(zip(fam, holm([res[s][3] for s in fam])))
        lines += [f"**Network ({title}) against the GP on s** (per-run RMSE difference, network minus GP; negative = "
                  f"network better; Holm correction over the {len(fam)} evaluations other than four runs)", "",
                  "| split | configuration | network mean | range over seed sets | GP on s | difference "
                  "| 95 % bootstrap interval | Wilcoxon p | Holm p | network wins |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in splits:
            sm = split_means(s, m); dm, lo, hi, pv, wins, n = res[s]
            lines.append(f"| {s} | {config[s][m]} | {np.mean(sm):.4f} | {max(sm) - min(sm):.4f} | {mean_of(s, 'gp_s'):.4f} | "
                         f"{dm:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {fmt_p(pv)} | {fmt_p(adj[s]) if s in adj else '-'} | {wins}/{n} |")
        lines.append("")
    lines += [f"**Corner run {corner}** (when held out)", "", "| split | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for s in splits:
        if corner in per_run(rows, s, "mlp"):
            lines.append(f"| {s} | " + " | ".join(f"{per_run(rows, s, m)[corner]:.3f}" for m in METHODS) + " |")
    lines += ["", "**Network: hyper-parameters and ensemble spread** (CV RMSE: per-run RMSE of the 3 x 5-fold "
              "cross-validation, averaged over the repeats and the runs; standard error: std over the runs of the best "
              "configuration's per-run CV RMSE / sqrt(runs); spread: one-standard-error choice)", "",
              "| split | training runs | one-SE rule | its CV RMSE | minimum rule | best CV RMSE | standard error | spread |",
              "|---|---|---|---|---|---|---|---|"]
    keys = [(g["activation"], g["alpha"]) for g in GRID]
    for (s, n_tr, act, alpha, act_min, alpha_min, se, *scores) in sel_rows:
        spread = np.mean([r[8] for r in rows if r[0] == s and r[1] == "mlp"])
        lines.append(f"| {s} | {n_tr} | {act} {alpha:g} | {scores[keys.index((act, alpha))]:.4f} | {act_min} {alpha_min:g} "
                     f"| {min(scores):.4f} | {se:.4f} | {spread:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
