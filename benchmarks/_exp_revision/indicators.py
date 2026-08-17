"""Multi-objective indicators over the evolved fronts (M6).

Every front-returning detector runs on the same graph. Objective vectors are
computed in one common space, the reference front is the non-dominated set of
the pooled objective vectors of all methods and all runs on that graph, and
hypervolume and IGD+ are measured against it after a shared min-max
normalisation. The reference point is (1.1, 1.1) in normalised space.

    python -m _exp_revision.indicators
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Shim, Sink, densify, lfr  # noqa: E402
from _exp_revision.selector import objectives  # noqa: E402

FIELDS = ["net", "kind", "n_cfg", "mu", "seed", "alg", "space", "n", "m",
          "front_size", "nd_size", "hv", "igdplus", "spread", "time", "stamp"]
KEY = ["net", "n_cfg", "mu", "seed", "alg", "space"]

# Objective space: the pair each method's own search optimises is not shared,
# so indicators are reported in both two-objective spaces.
SPACES = {"intra_inter": (0, 1), "kkm_rc": (2, 3)}


def front_of(pymocd, alg, shim):
    if alg == "SMOCC":
        r = pymocd.smocc_probe(shim)
        return r["front"]
    return {"HP-MOCD": pymocd.hpmocd_fronts,
            "NSGA-III CCM": pymocd.ccm_fronts,
            "NSGA-III KRM": pymocd.krm_fronts,
            "MOGA-Net": pymocd.moga_net_fronts,
            "MMCoMO": pymocd.mmcomo_fronts}[alg](shim)


def nondominated(pts):
    keep = np.ones(len(pts), dtype=bool)
    for i, a in enumerate(pts):
        if not keep[i]:
            continue
        dom = np.all(pts <= a, axis=1) & np.any(pts < a, axis=1)
        if dom.any():
            keep[i] = False
    return pts[keep]


def hypervolume2d(pts, ref):
    """Exact 2-D hypervolume of a minimisation front."""
    p = nondominated(np.asarray(pts, dtype=float))
    p = p[(p[:, 0] < ref[0]) & (p[:, 1] < ref[1])]
    if len(p) == 0:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in p:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def igd_plus(front, ref):
    """IGD+ of an approximation front against a reference front."""
    f, r = np.asarray(front, dtype=float), np.asarray(ref, dtype=float)
    if len(f) == 0:
        return float("inf")
    d = np.sqrt((np.maximum(f[None, :, :] - r[:, None, :], 0.0) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def spread(pts):
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(d.std() / d.mean()) if d.mean() > 0 else 0.0


def run_graph(sink, pymocd, net, kind, n_cfg, mu, seed, n, edges, algs):
    pending = [a for a in algs
               if any(not sink.has({"net": net, "n_cfg": n_cfg, "mu": mu,
                                    "seed": seed, "alg": a, "space": s})
                      for s in SPACES)]
    if not pending:
        return
    shim = Shim(n, edges)
    obj, times = {}, {}
    for alg in algs:
        t0 = time.perf_counter()
        try:
            fr = front_of(pymocd, alg, shim)
        except Exception as e:  # noqa: BLE001
            print(f"  {alg}: FAILED {e}", flush=True)
            continue
        times[alg] = time.perf_counter() - t0
        labs = [densify(p, n) for p in fr]
        obj[alg] = np.array([objectives(n, edges, p)[:4] for p in labs])
        print(f"  {alg}: |F|={len(labs)} in {times[alg]:.1f}s", flush=True)
    if not obj:
        return

    pooled = np.vstack(list(obj.values()))
    for space, cols in SPACES.items():
        c = list(cols)
        allp = pooled[:, c]
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        rng = np.where(hi - lo > 0, hi - lo, 1.0)
        norm = lambda x: (x - lo) / rng
        ref_front = nondominated(norm(allp))
        ref_point = np.array([1.1, 1.1])
        for alg, o in obj.items():
            row = {"net": net, "n_cfg": n_cfg, "mu": mu, "seed": seed,
                   "alg": alg, "space": space}
            if sink.has(row):
                continue
            z = norm(o[:, c])
            nd = nondominated(z)
            sink.write(dict(row, kind=kind, n=n, m=int(len(edges)),
                            front_size=len(z), nd_size=len(nd),
                            hv=round(hypervolume2d(nd, ref_point), 6),
                            igdplus=round(igd_plus(nd, ref_front), 6),
                            spread=round(spread(nd), 6),
                            time=round(times.get(alg, 0.0), 4)))


def main():
    import pymocd
    from _exp_real_net.networks import LOADERS
    pymocd.max_cores(int(os.environ.get("REV_THREADS", "48")))
    sink = Sink("indicators.csv", FIELDS, KEY)

    big = ["SMOCC", "HP-MOCD", "NSGA-III CCM", "NSGA-III KRM", "MOGA-Net"]
    small = big + ["MMCoMO"]

    nets = os.environ.get(
        "REV_IND_NETS",
        "karate,dolphins,polbooks,football,lesmis,florentine,email_eu").split(",")
    for net in nets:
        edges, n, _gt, _ev = LOADERS[net]()
        algs = small if n <= 2000 else big
        print(f"{net} (n={n})", flush=True)
        run_graph(sink, pymocd, net, "real", "", "", 0, n, edges, algs)

    grid = json.loads(os.environ.get(
        "REV_IND_LFR",
        '[[300,0.3],[300,0.5],[600,0.3],[600,0.5],[1000,0.3],[1000,0.5],'
        '[2000,0.3],[2000,0.5],[10000,0.3],[10000,0.5]]'))
    runs = int(os.environ.get("REV_IND_RUNS", "5"))
    for n_cfg, mu in grid:
        algs = small if n_cfg <= 2000 else big
        for seed in range(runs):
            edges, _gt = lfr(n_cfg, mu, seed)
            print(f"lfr n={n_cfg} mu={mu} s={seed}", flush=True)
            run_graph(sink, pymocd, f"lfr{n_cfg}", "lfr", n_cfg, mu, seed,
                      n_cfg, edges, algs)
    print(f"INDICATORS_DONE -> {os.path.join(OUT, 'indicators.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
