"""Does the reference LFR generator deliver the requested mixing (M10)?

Measures realized mixing, degree and planted-community statistics for the
networkx generator the campaign used and for the reference generator of
``lfr_reference.py``, over the campaign parameters. Acceptance: the reference
generator's realized mu must be within 0.02 of nominal mu and its mean degree
within 1.0 of 20.

networkx graphs already cached under ``data/lfr`` are read from there, which
is exactly what the campaign ran; the remaining cells are generated in memory
and never written to that cache.

    python -m _exp_revision.lfr_verify
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision import lfr_reference  # noqa: E402
from _exp_revision.common import (LFR_PARAMS, Sink, flat,  # noqa: E402
                                  summarise)

VER_N = [1_000, 10_000, 50_000]
VER_MU = [0.1, 0.3, 0.5, 0.7]
VER_SEEDS = 5

GENS = ["networkx", "reference"]

FIELDS = (["gen", "n_cfg", "mu", "seed", "seed_value", "n", "m", "mu_real",
           "deg_mean", "deg_max", "deg_min", "k", "t_gen"]
          + [f"csz_{k}" for k in
             ("n", "mean", "med", "p05", "p95", "min", "max")]
          + ["stamp"])
KEY = ["gen", "n_cfg", "mu", "seed"]

TOL_MU = 0.02
TOL_DEG = 1.0


def nx_graph(n, mu, seed):
    """The campaign's networkx graph: from data/lfr if cached, else in memory."""
    import networkx as nx
    from _exp_synt_net.gen_graphs import path_for
    p = path_for(n, mu, seed)
    if os.path.exists(p):
        d = np.load(p)
        return d["edges"], d["gt"].astype(np.int64), seed
    for attempt in range(5):
        s = seed + 1000 * attempt
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=n, mu=mu, seed=s, **LFR_PARAMS)
        except Exception:
            if attempt == 4:
                raise
            continue
        gt = np.empty(n, dtype=np.int64)
        for node in G:
            gt[node] = min(G.nodes[node]["community"])
        edges = np.array([(u, v) for u, v in nx.Graph(G).edges() if u != v],
                         dtype=np.uint32)
        return edges, gt, s


def measure(n, edges, gt):
    e = edges.astype(np.int64)
    deg = np.bincount(e.reshape(-1), minlength=n)
    _, sizes = np.unique(gt, return_counts=True)
    return {"n": int(n), "m": int(len(e)),
            "mu_real": round(float((gt[e[:, 0]] != gt[e[:, 1]]).mean()), 6),
            "deg_mean": round(float(deg.mean()), 4),
            "deg_max": int(deg.max()), "deg_min": int(deg.min()),
            "k": int(len(sizes)),
            **flat("csz", summarise(sizes))}


def one(sink, gen, n, mu, seed):
    base = {"gen": gen, "n_cfg": n, "mu": mu, "seed": seed}
    if sink.has(base):
        return
    t0 = time.perf_counter()
    if gen == "networkx":
        edges, gt, sval = nx_graph(n, mu, seed)
    else:
        edges, gt, sval = lfr_reference.generate(n, mu, seed)
        gt = gt.astype(np.int64)
    dt = time.perf_counter() - t0
    row = dict(base, seed_value=sval, t_gen=round(dt, 3),
               **measure(n, edges, gt))
    sink.write(row)
    print(f"{gen:>9} n={n} mu={mu} seed={seed} mu_real={row['mu_real']:.4f} "
          f"deg_mean={row['deg_mean']:.2f} k={row['k']} in {dt:.1f}s",
          flush=True)


def aggregate(path):
    acc = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (int(r["n_cfg"]), float(r["mu"]), r["gen"])
            acc.setdefault(k, []).append(r)
    return acc


def table(path):
    acc = aggregate(path)
    cols = ("mu_real", "deg_mean", "deg_max", "k", "csz_min", "csz_max")
    print()
    print("realized statistics, means over seeds; nominal k=20, max_degree=50, "
          "community sizes in [20, 100]")
    head = (f"{'n':>7} {'mu_nom':>7} {'gen':>10} {'seeds':>6} "
            + " ".join(f"{c:>9}" for c in cols))
    print(head)
    print("-" * len(head))
    for n in VER_N:
        for mu in VER_MU:
            for gen in GENS:
                rows = acc.get((n, mu, gen))
                if not rows:
                    continue
                v = {c: np.mean([float(r[c]) for r in rows]) for c in cols}
                print(f"{n:>7} {mu:>7.1f} {gen:>10} {len(rows):>6} "
                      + " ".join(f"{v[c]:>9.3f}" for c in cols))
            print()


def acceptance(path):
    acc = aggregate(path)
    print("acceptance: |mu_real - mu_nominal| <= "
          f"{TOL_MU} and |deg_mean - {LFR_PARAMS['average_degree']}| "
          f"<= {TOL_DEG}, per cell, over seed means")
    head = (f"{'n':>7} {'mu_nom':>7} {'d_mu(ref)':>10} {'d_deg(ref)':>11} "
            f"{'verdict':>8} | {'d_mu(nx)':>9} {'d_deg(nx)':>10}")
    print(head)
    print("-" * len(head))
    ok = True
    for n in VER_N:
        for mu in VER_MU:
            ref = acc.get((n, mu, "reference"))
            nxr = acc.get((n, mu, "networkx"))
            if not ref:
                continue
            dmu = abs(np.mean([float(r["mu_real"]) for r in ref]) - mu)
            ddeg = abs(np.mean([float(r["deg_mean"]) for r in ref])
                       - LFR_PARAMS["average_degree"])
            good = dmu <= TOL_MU and ddeg <= TOL_DEG
            ok &= good
            s = ""
            if nxr:
                s = (f"{abs(np.mean([float(r['mu_real']) for r in nxr]) - mu):>9.3f} "
                     f"{abs(np.mean([float(r['deg_mean']) for r in nxr]) - LFR_PARAMS['average_degree']):>10.3f}")
            print(f"{n:>7} {mu:>7.1f} {dmu:>10.4f} {ddeg:>11.4f} "
                  f"{'PASS' if good else 'FAIL':>8} | {s}")
    print()
    print("VERIFY_VERDICT " + ("PASS" if ok else "FAIL"))


def main():
    sink = Sink("lfr_verify.csv", FIELDS, KEY)
    for n in VER_N:
        for mu in VER_MU:
            for seed in range(VER_SEEDS):
                for gen in GENS:
                    one(sink, gen, n, mu, seed)
    table(sink.path)
    acceptance(sink.path)


if __name__ == "__main__":
    main()
