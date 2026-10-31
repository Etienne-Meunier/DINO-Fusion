"""Test RMSE of the network for every grid configuration on each hold-out split, with every seed set: a check of how
much the result depends on the hyper-parameter choice, and how stable the choice is under other cross-validation
partitions. It is not used for the selection.

    python -m det.sensitivity [--data data/veros_acc_lastyear.npz] [--results det/results]

Reads selected.csv (both rules' choices, the cross-validation scores and their standard error, from det.run) and
metrics.csv (the network and the GP on s), writes sensitivity.csv (split, configuration, its CV RMSE, within one
standard error of the best CV score, chosen by either rule, test mean RMSE over the seed sets and its range, and
the CV RMSE and both rules' choices with the partitions seeded ALT instead of 0) and prints, per split, the range of
the test RMSE over the configurations that cross-validation cannot tell apart. It also checks that the chosen
configurations reproduce metrics.csv exactly.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
from joblib import Parallel, delayed

from det.methods import MLP, select
from det.run import GRID, SEEDS, SPLITS
from extract_data import choose_holdout

ALT = 10                      # other cross-validation partitions: KFold random_state 10, 11, 12 instead of 0, 1, 2


def test_rmse(g, sd, tr, ho, Z, Y):
    """Per-run test RMSE of the network with configuration g and seed set sd."""
    err = MLP(**g, seed=sd).fit(Z[tr], Y[tr]).predict(Z[ho]) - Y[ho]
    return np.array([np.sqrt((e ** 2).mean()) for e in err])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/veros_acc_lastyear.npz")
    p.add_argument("--results", default="det/results")
    a = p.parse_args(argv)
    res = Path(a.results)
    d = np.load(a.data)
    water = ~d["mask_land"]
    Y = d["temp"][:, water].astype(np.float64)
    names = [str(n) for n in d["run_names"]]
    logp = np.c_[np.log(d["run_ck"]), np.log(d["run_eps"])]
    Z = (logp - logp.mean(0)) / logp.std(0)
    cv = {r["split"]: r for r in csv.DictReader(open(res / "selected.csv"))}
    ref = {}                                                    # (split, method, seed, run) -> RMSE from det.run
    for r in csv.DictReader(open(res / "metrics.csv")):
        if r["method"] in ("mlp", "mlp_min", "gp_s"):
            ref[(r["split"], r["method"], r["seed"], r["run"])] = float(r["rmse_K"])

    tasks, alt = [], {}
    for name, s in SPLITS.items():
        ho = np.asarray(choose_holdout(d["run_ck"], d["run_eps"], s["mode"], s.get("n", 10), s.get("seed", 0),
                                       s.get("rows"), s.get("block"), s.get("points", ())))
        tr = np.setdiff1d(np.arange(len(Y)), ho)
        tasks += [(name, g, sd, tr, ho) for g in GRID for sd in SEEDS]
        pick, scores, _ = select(MLP, GRID, Z[tr], Y[tr], seed=ALT)
        alt[name] = scores, (pick["activation"], pick["alpha"]), tuple(GRID[int(np.argmin(scores))].values())
    err = Parallel(n_jobs=-1)(delayed(test_rmse)(g, sd, tr, ho, Z, Y) for _, g, sd, tr, ho in tasks)

    rows, gap = [], 0.0
    for name in SPLITS:
        c = cv[name]
        score = {(x["activation"], x["alpha"]): float(c[f"cv_{x['activation']}_{x['alpha']:g}"]) for x in GRID}
        best, se = min(score.values()), float(c["cv_se"])                 # as in methods.select, at full precision
        for g in GRID:
            key = (g["activation"], g["alpha"])
            e = [(sd, ho, x) for (n, h, sd, _, ho), x in zip(tasks, err) if n == name and h == g]
            means = [x.mean() for _, _, x in e]
            chosen = [m for m, (act, al) in (("mlp", ("activation", "alpha")), ("mlp_min", ("activation_min", "alpha_min")))
                      if key == (c[act], float(c[al]))]
            for m in chosen:                                              # reproduces det.run?
                gap = max(gap, max(abs(x[i] - ref[(name, m, str(sd), names[r])]) for sd, ho, x in e for i, r in enumerate(ho)))
            rows.append([name, *key, score[key], int(score[key] <= best + se), int("mlp" in chosen), int("mlp_min" in chosen),
                         float(np.mean(means)), float(max(means) - min(means)), float(alt[name][0][GRID.index(g)]),
                         int(key == alt[name][1]), int(key == alt[name][2])])
    with open(res / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "activation", "alpha", "cv_rmse_K", "within_1se", "chosen", "chosen_min", "test_rmse_K",
                    "test_range_K", "cv_rmse_alt_K", "chosen_alt", "chosen_min_alt"])
        w.writerows(rows)
    print(f"largest difference from metrics.csv for the chosen configurations: {gap:.1e} K")
    for name in SPLITS:
        sub = [r for r in rows if r[0] == name]
        eq = sorted(r[7] for r in sub if r[4])
        gp = np.mean([v for (s, m, _, _), v in ref.items() if s == name and m == "gp_s"])
        pick = next(r[7] for r in sub if r[5])
        print(f"{name:9s} chosen {pick:.4f} (rank {eq.index(pick) + 1} of {len(eq)} within 1 SE) | within 1 SE: "
              f"{eq[0]:.4f}-{eq[-1]:.4f}, {sum(x < gp for x in eq)} below GP on s {gp:.4f} | all ten "
              f"{min(r[7] for r in sub):.4f}-{max(r[7] for r in sub):.4f} | largest seed range {max(r[8] for r in sub):.4f}")
        cfg = lambda col: " ".join(f"{r[1]} {r[2]:g}" for r in sub if r[col])
        print(f"{'':9s} choice, partitions 0 / {ALT}: one SE {cfg(5)} / {cfg(10)}, minimum {cfg(6)} / {cfg(11)}")


if __name__ == "__main__":
    main()
