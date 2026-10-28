"""Test RMSE of the network for every grid configuration on each hold-out split (first seed set): a check of how
much the result depends on the hyper-parameter choice. It is not used for the selection.

    python -m det.sensitivity [--data data/veros_acc_lastyear.npz] [--results det/results]

Reads selected.csv (cross-validation scores and standard error, from det.run) and metrics.csv (the GP on s), writes
sensitivity.csv (split, activation, alpha, within one standard error of the best CV score, test mean RMSE) and
prints, per split, the range of the test RMSE over the configurations that cross-validation cannot tell apart.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
from joblib import Parallel, delayed

from det.methods import MLP
from det.run import GRID, SPLITS
from extract_data import choose_holdout


def test_rmse(g, tr, ho, Z, Y):
    return np.sqrt(((MLP(**g, seed=0).fit(Z[tr], Y[tr]).predict(Z[ho]) - Y[ho]) ** 2).mean(1)).mean()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/veros_acc_lastyear.npz")
    p.add_argument("--results", default="det/results")
    a = p.parse_args(argv)
    res = Path(a.results)
    d = np.load(a.data)
    water = ~d["mask_land"]
    Y = d["temp"][:, water].astype(np.float64)
    logp = np.c_[np.log(d["run_ck"]), np.log(d["run_eps"])]
    Z = (logp - logp.mean(0)) / logp.std(0)
    cv = {r["split"]: r for r in csv.DictReader(open(res / "selected.csv"))}
    gp_s = {}
    for r in csv.DictReader(open(res / "metrics.csv")):
        if r["method"] == "gp_s":
            gp_s.setdefault(r["split"], []).append(float(r["rmse_K"]))

    tasks = []
    for name, s in SPLITS.items():
        ho = np.asarray(choose_holdout(d["run_ck"], d["run_eps"], s["mode"], s.get("n", 10), s.get("seed", 0),
                                       s.get("rows"), s.get("block"), s.get("points", ())))
        tasks += [(name, g, np.setdiff1d(np.arange(len(Y)), ho), ho) for g in GRID]
    err = Parallel(n_jobs=-1)(delayed(test_rmse)(g, tr, ho, Z, Y) for _, g, tr, ho in tasks)

    rows = []
    for (name, g, _, _), e in zip(tasks, err):
        c = cv[name]
        score, best, se = float(c[f"cv_{g['activation']}_{g['alpha']:g}"]), min(float(c[f"cv_{x['activation']}_{x['alpha']:g}"]) for x in GRID), float(c["cv_se"])
        chosen = g["activation"] == c["activation"] and g["alpha"] == float(c["alpha"])
        rows.append([name, g["activation"], g["alpha"], int(score <= best + se), int(chosen), round(float(e), 5)])
    with open(res / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "activation", "alpha", "within_1se", "chosen", "test_rmse_K"]); w.writerows(rows)
    for name in SPLITS:
        sub = [r for r in rows if r[0] == name]
        eq = [r[5] for r in sub if r[3]]
        gp = np.mean(gp_s[name])
        print(f"{name:9s} chosen {next(r[5] for r in sub if r[4]):.4f} | within 1 SE: {len(eq)} configurations, "
              f"{min(eq):.4f}-{max(eq):.4f}, {sum(x < gp for x in eq)} below GP on s {gp:.4f} | all ten {min(r[5] for r in sub):.4f}-{max(r[5] for r in sub):.4f}")


if __name__ == "__main__":
    main()
