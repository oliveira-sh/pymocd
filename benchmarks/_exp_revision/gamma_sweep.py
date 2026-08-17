"""Leiden-CPM resolution sweep (the oracle asymmetry the review flags).

The campaign ran CPM at one density-anchored gamma. This sweeps gamma so the
paper can report what an informed choice would have bought, matching the
ground-truth oracle it already grants its own selector.

    python -m _exp_revision.gamma_sweep
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink, lfr, score  # noqa: E402

FIELDS = ["net", "kind", "n_cfg", "mu", "seed", "gamma", "gamma_rule", "n",
          "m", "k", "gt_k", "nmi", "ami", "ari", "modularity", "time", "stamp"]
KEY = ["net", "kind", "n_cfg", "mu", "seed", "gamma_rule"]

MULTIPLIERS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def run_cell(sink, net, kind, n_cfg, mu, seed, n, edges, gt):
    import igraph as ig
    density = len(edges) / (n * (n - 1) / 2)
    for mult in MULTIPLIERS:
        rule = f"density x{mult:g}"
        base = {"net": net, "kind": kind, "n_cfg": n_cfg, "mu": mu,
                "seed": seed, "gamma_rule": rule}
        if sink.has(base):
            continue
        gamma = density * mult
        g = ig.Graph(n=n, edges=edges)
        t0 = time.perf_counter()
        part = g.community_leiden(objective_function="CPM", resolution=gamma)
        dt = time.perf_counter() - t0
        lab = np.asarray(part.membership, dtype=np.int64)
        row = dict(base, gamma=gamma, n=n, m=len(edges), time=round(dt, 4))
        row.update(score(gt, lab, n=n, edges=edges))
        if gt is not None:
            row["gt_k"] = int(len(np.unique(gt)))
        sink.write(row)


def main():
    from _exp_real_net.networks import LOADERS
    sink = Sink("gamma_sweep.csv", FIELDS, KEY)
    grid = json.loads(os.environ.get(
        "REV_GAMMA_LFR",
        '[[50000,0.1],[50000,0.3],[50000,0.5],[50000,0.6],'
        '[100000,0.3],[100000,0.5],[10000,0.3],[10000,0.5]]'))
    runs = int(os.environ.get("REV_GAMMA_RUNS", "10"))
    for n_cfg, mu in grid:
        for seed in range(runs):
            edges, gt = lfr(n_cfg, mu, seed)
            run_cell(sink, f"lfr{n_cfg}", "lfr", n_cfg, mu, seed, n_cfg,
                     edges, gt)
        print(f"lfr n={n_cfg} mu={mu} done", flush=True)

    nets = os.environ.get(
        "REV_GAMMA_NETS",
        "karate,dolphins,polbooks,football,email_eu").split(",")
    for net in nets:
        edges, n, gt, eval_nodes = LOADERS[net]()
        run_cell(sink, net, "real", "", "", 0, n, edges, gt)
        print(f"{net} done", flush=True)
    print(f"GAMMA_DONE -> {os.path.join(OUT, 'gamma_sweep.csv')}", flush=True)


if __name__ == "__main__":
    main()
