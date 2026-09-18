"""Side-by-side table of evaluation metrics for several runs.

    python compare_runs.py results/full_3std results/full_6std
Each argument is a folder holding an evaluate.py ``metrics.csv``. Prints a markdown table of the
mean over hold-out runs for every (method, metric) pair, one column per run.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(folder: Path) -> dict[tuple[str, str], float]:
    acc = defaultdict(list)
    with open(folder / "metrics.csv") as f:
        for row in csv.DictReader(f):
            acc[(row["method"], row["metric"])].append(float(row["value"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main(folders: list[str]) -> None:
    runs = {Path(f).name: load(Path(f)) for f in folders}
    keys = sorted({k for m in runs.values() for k in m}, key=lambda k: (k[1], k[0]))
    order = ["rmse_K", "bias_domain_mean_K", "spread_K", "temp_inversion_frac", "salt_max_abs_err", "n_neighbours"]
    keys.sort(key=lambda k: (order.index(k[1]) if k[1] in order else 99, k[0]))
    w = max(len(f"{m} / {k}") for m, k in keys)
    print(f"| {'method / metric':<{w}} | " + " | ".join(f"{r:>14}" for r in runs) + " |")
    print(f"|{'-' * (w + 2)}|" + "|".join("-" * 16 for _ in runs) + "|")
    for m, k in keys:
        cells = [f"{runs[r][(m, k)]:>14.4f}" if (m, k) in runs[r] else f"{'':>14}" for r in runs]
        print(f"| {f'{m} / {k}':<{w}} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
