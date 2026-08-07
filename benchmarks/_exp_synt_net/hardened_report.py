import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _exp_synt_net.hardened import RESULTS_CSV


def cell_id(r):
    if r["kind"] == "real":
        return f"real:{r['net']}"
    return f"lfr:n{int(r['n_cfg'])}:mu{r['mu']:g}"


def main():
    df = pd.read_csv(RESULTS_CSV)
    print("status counts:\n", df["status"].value_counts().to_string(), "\n")
    ok = df[df["status"] == "ok"].copy()
    ok["cell"] = ok.apply(cell_id, axis=1)

    metric = sys.argv[1] if len(sys.argv) > 1 else "ami"
    agg = ok.pivot_table(index="cell", columns="alg", values=metric,
                         aggfunc=["mean", "std"]).round(4)
    print(f"=== {metric} (mean / std) ===")
    print(agg.to_string())

    if wilcoxon is None:
        print("\nscipy not installed; skipping significance tests")
        return
    print(f"\n=== Wilcoxon signed-rank: SMOCC vs others on {metric} "
          f"(paired by seed, Bonferroni per cell) ===")
    for cell, sub in ok.groupby("cell"):
        ours = sub[sub["alg"] == "SMOCC"].set_index("seed")[metric].dropna()
        others = [a for a in sub["alg"].unique() if a != "SMOCC"]
        m = len(others)
        for alg in sorted(others):
            theirs = sub[sub["alg"] == alg].set_index("seed")[metric].dropna()
            common = ours.index.intersection(theirs.index)
            if len(common) < 5:
                continue
            a, b = ours[common], theirs[common]
            if np.allclose(a, b):
                verdict = "tie"
            else:
                p = wilcoxon(a, b).pvalue * m
                direction = "SMOCC>" if a.mean() > b.mean() else "SMOCC<"
                verdict = f"{direction} p_adj={min(p, 1):.4f}" \
                          f"{' *' if p < 0.05 else ''}"
            print(f"{cell:24s} vs {alg:14s} n={len(common):2d} "
                  f"d={a.mean() - b.mean():+.4f}  {verdict}")


if __name__ == "__main__":
    main()
