"""The report's tables, from the files written by det.run and det.sensitivity, so that no number is copied by hand.

    python -m det.tables [--results det/results]

Writes to --results: table_main.tex (RMSE per split and method), table_paired.tex (the network against the GP on s
under both selection rules) and table_sensitivity.tex (test RMSE over the configurations, stability of the choice);
report/report.tex inputs them. Prints the main table in Markdown, for the README.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from det.run import MAIN, METHODS, RULES, SEEDS, holm, paired, per_run

ORDER = ["scattered", "band", "centre5", "centre7", "loo", "top", "outer51", "outer75"]   # interpolation, extrapolation
LABEL = {"scattered": "scattered", "band": "band (three rows)", "centre5": r"centre $5\times5$",
         "centre7": r"centre $7\times7$", "loo": "leave-one-out", "top": "top row", "outer51": "outer 51",
         "outer75": "outer 75", "four": "four runs"}


def cfg(act, alpha):
    """A configuration: alpha as a power of ten, with the activation only when it is tanh."""
    a = "$1$" if float(alpha) == 1 else f"$10^{{{int(round(np.log10(float(alpha))))}}}$"
    return a if act == "relu" else f"tanh {a}"


def fmt_p(p):
    return "$<0.001$" if p < 1e-3 else f"{p:.3f}"


def table_main(rows, sel, mean, seed_range):
    tex = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
           r"split & held out / train & network & range & median & GP on $s$ & GP on $z$ & RBF & nearest & mean state \\",
           r"\midrule"]
    md = ["| split | held out | network | range over initialisation sets | GP on s | GP on z | RBF | nearest | mean state |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in ORDER + ["four"]:
        v = {m: mean(s, m) for m in METHODS}
        best = min(v, key=v.get)
        n = len(per_run(rows, s, "mlp"))
        if s != "four":
            cell = lambda m: rf"\textbf{{{v[m]:.3f}}}" if m == best else f"{v[m]:.3f}"
            tex.append(f"{LABEL[s]} & {n} / {sel[s]['n_train']} & {cell('mlp')} & {seed_range(s, 'mlp'):.4f} & "
                       f"{np.median(list(per_run(rows, s, 'mlp').values())):.3f} & "
                       + " & ".join(cell(m) for m in METHODS[1:]) + r" \\")
            if s == "loo":
                tex.append(r"\midrule")
        cell = lambda m: f"**{v[m]:.3f}**" if m == best else f"{v[m]:.3f}"
        label = LABEL[s].replace(r"$5\times5$", "5 x 5").replace(r"$7\times7$", "7 x 7").replace("band (three rows)", "band of 3 rows")
        md.append(f"| {label + (' (failure case)' if s == 'four' else '')} | {n} | {cell('mlp')} | {seed_range(s, 'mlp'):.4f} | "
                  + " | ".join(cell(m) for m in METHODS[1:]) + " |")
    return tex + [r"\bottomrule", r"\end{tabular}"], md


def table_paired(rows, sel, mean):
    assert sorted(ORDER) == sorted(MAIN)                              # the Holm family is the main comparison
    res = {m: {s: paired(per_run(rows, s, m), per_run(rows, s, "gp_s")) for s in ORDER} for m in RULES}
    adj = {m: dict(zip(ORDER, holm([res[m][s][3] for s in ORDER]))) for m in RULES}
    tex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
           r" & \multicolumn{3}{c}{one-standard-error rule (reported)} & \multicolumn{5}{c}{minimum CV RMSE} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-9}",
           r"split & difference [95\,\% interval] & $p$ & Holm $p$ & choice & network & difference & $p$ & Holm $p$ \\",
           r"\midrule"]
    num = lambda x: f"${x:+.4f}$"
    for s in ORDER:
        d, lo, hi, p, _, _ = res["mlp"][s]
        dm, _, _, pm, _, _ = res["mlp_min"][s]
        tex.append(f"{LABEL[s]} & {num(d)} [{num(lo)}, {num(hi)}] & {fmt_p(p)} & {fmt_p(adj['mlp'][s])} & "
                   f"{cfg(sel[s]['activation_min'], sel[s]['alpha_min'])} & {mean(s, 'mlp_min'):.3f} & {num(dm)} & "
                   f"{fmt_p(pm)} & {fmt_p(adj['mlp_min'][s])}" + r" \\")
        if s == "loo":
            tex.append(r"\midrule")
    return tex + [r"\bottomrule", r"\end{tabular}"]


def table_sensitivity(sens, mean):
    tex = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           r" & \multicolumn{4}{c}{configurations within one standard error} & & "
           r"\multicolumn{2}{c}{choice, other partitions} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}",
           r"split & number & test RMSE & below GP on $s$ & rank of the choice & all ten & one SE & minimum \\",
           r"\midrule"]
    for s in [x for x in ORDER if x != "loo"] + ["four"]:
        sub = [r for r in sens if r["split"] == s]
        eq = sorted(float(r["test_rmse_K"]) for r in sub if r["within_1se"] == "1")
        pick = next(float(r["test_rmse_K"]) for r in sub if r["chosen"] == "1")
        allten = [float(r["test_rmse_K"]) for r in sub]

        def other(new, old):                                          # the choice with the other partitions
            a, b = (next(r for r in sub if r[c] == "1") for c in (new, old))
            return "same" if a is b else cfg(a["activation"], a["alpha"])

        if s == "four":                                               # KFold(4) is leave-one-out: nothing to compare
            tex.append(r"\midrule")
        tex.append(f"{LABEL[s]} & {len(eq)} & {eq[0]:.3f}--{eq[-1]:.3f} & {sum(x < mean(s, 'gp_s') for x in eq)} & "
                   f"{eq.index(pick) + 1} & {min(allten):.3f}--{max(allten):.3f} & {other('chosen_alt', 'chosen') if s != 'four' else '--'} & "
                   f"{other('chosen_min_alt', 'chosen_min') if s != 'four' else '--'}" + r" \\")
    return tex + [r"\bottomrule", r"\end{tabular}"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="det/results")
    res = Path(p.parse_args(argv).results)
    rows = [[r["split"], r["method"], r["seed"], r["run"], float(r["ck"]), float(r["eps"]), float(r["rmse_K"])]
            for r in csv.DictReader(open(res / "metrics.csv"))]
    sel = {r["split"]: r for r in csv.DictReader(open(res / "selected.csv"))}
    sens = list(csv.DictReader(open(res / "sensitivity.csv")))
    mean = lambda s, m: float(np.mean(list(per_run(rows, s, m).values())))
    split_means = lambda s, m: [np.mean([r[6] for r in rows if r[0] == s and r[1] == m and r[2] == str(sd)]) for sd in SEEDS]
    seed_range = lambda s, m: max(split_means(s, m)) - min(split_means(s, m))
    tex, md = table_main(rows, sel, mean, seed_range)
    (res / "table_main.tex").write_text("\n".join(tex) + "\n")
    (res / "table_paired.tex").write_text("\n".join(table_paired(rows, sel, mean)) + "\n")
    (res / "table_sensitivity.tex").write_text("\n".join(table_sensitivity(sens, mean)) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
