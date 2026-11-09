"""Predictions at every grid point for each hold-out split: the input of the deterministic evaluation reports
(MiniVeros-Diffusion, branch deterministic-evaluation, package deterministic_evaluation).

    python -m det.grid [--data data/veros_acc_lastyear.npz] [--results det/results] [--out data/det_grid]

For each split of det.run, two methods are fitted on the split's training runs and predict all 100 runs, training runs
included:
    mlp    the reported network: the one-standard-error choice of selected.csv, first initialisation set (SEEDS[0],
           5 members), as in the report's figures
    gp_s   the GP on s
Writes <out>/<split>.npz with temp_mlp and temp_gp_s (R, Z, Y, X) float32, NaN on land; training_run_id and
held_out_run_id (sorted run ids); run_names, run_ck, run_eps, zt, mask_land; activation, alpha, seed of the network.
Every held-out run's RMSE is checked against metrics.csv. Leave-one-out has no single training set and is skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import time as clock
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

from det.methods import GP, MLP
from det.run import SEEDS, SPLITS
from extract_data import choose_holdout

METHODS = ("mlp", "gp_s")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/veros_acc_lastyear.npz")
    p.add_argument("--results", default="det/results", help="folder with selected.csv and metrics.csv of det.run")
    p.add_argument("--out", default="data/det_grid")
    p.add_argument("--splits", default=",".join(SPLITS))
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.data)
    water = ~d["mask_land"]
    Y = d["temp"][:, water].astype(np.float64)
    run_ck, run_eps, names = d["run_ck"], d["run_eps"], [str(n) for n in d["run_names"]]
    logp = np.c_[np.log(run_ck), np.log(run_eps)]
    Z = (logp - logp.mean(0)) / logp.std(0)                       # as det.run
    R = len(Y)
    selected = {r["split"]: r for r in csv.DictReader(open(Path(a.results) / "selected.csv"))}
    reference = {(r["split"], r["method"], r["seed"], r["run"]): float(r["rmse_K"])
                 for r in csv.DictReader(open(Path(a.results) / "metrics.csv"))}

    for name in a.splits.split(","):
        t0 = clock.time()
        s = SPLITS[name]
        ho = np.sort(np.asarray(choose_holdout(run_ck, run_eps, s["mode"], s.get("n", 10), s.get("seed", 0),
                                               s.get("rows"), s.get("block"), s.get("points", ()))))
        tr = np.setdiff1d(np.arange(R), ho)
        act, alpha = selected[name]["activation"], float(selected[name]["alpha"])
        pred = {"mlp": MLP(activation=act, alpha=alpha, seed=SEEDS[0]).fit(Z[tr], Y[tr]).predict(Z),
                "gp_s": GP(on="s").fit(Z[tr], Y[tr]).predict(Z)}
        worst = 0.0
        for m, P in pred.items():                                  # the held-out errors must be those of det.run
            seed = str(SEEDS[0]) if m == "mlp" else ""
            for r in ho:
                rmse = float(np.sqrt(((P[r] - Y[r]) ** 2).mean()))
                worst = max(worst, abs(rmse - reference[(name, m, seed, names[r])]))
        assert worst < 1e-9, f"{name}: predictions differ from metrics.csv by {worst:.1e} K"
        fields = {}
        for m, P in pred.items():
            f = np.full((R,) + water.shape, np.nan, np.float32)
            f[:, water] = P
            fields[f"temp_{m}"] = f
        np.savez(out / f"{name}.npz", **fields, training_run_id=tr, held_out_run_id=ho, run_names=np.array(names),
                 run_ck=run_ck, run_eps=run_eps, zt=d["zt"], mask_land=d["mask_land"], activation=act, alpha=alpha,
                 seed=SEEDS[0])
        print(f"{name:9s} {len(tr):3d} training, {len(ho):3d} held out | mlp {act} {alpha:g} | held-out RMSE "
              f"mlp {np.mean([np.sqrt(((pred['mlp'][r] - Y[r]) ** 2).mean()) for r in ho]):.4f} "
              f"gp_s {np.mean([np.sqrt(((pred['gp_s'][r] - Y[r]) ** 2).mean()) for r in ho]):.4f} K | "
              f"max |RMSE - metrics.csv| {worst:.1e} | {clock.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
