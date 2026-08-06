"""Resumable LFR graph cache; run to completion before run.py so graph
generation does not spend the workers' timeout budget."""

import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardened.config import LFR_DIR, LFR_PARAMS, MU_SWEEP_MU, MU_SWEEP_N, \
    NODES_SWEEP_MU, NODES_SWEEP_N, RUNS


def path_for(n, mu, seed):
    return os.path.join(LFR_DIR, f"n{n}_mu{mu:g}_s{seed}.npz")


def generate(n, mu, seed):
    for attempt in range(5):
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=n, mu=mu, seed=seed + 1000 * attempt, **LFR_PARAMS)
            break
        except Exception:
            if attempt == 4:
                raise
    gt = np.empty(n, dtype=np.int32)
    for node in G:
        gt[node] = min(G.nodes[node]["community"])
    edges = np.array([(u, v) for u, v in nx.Graph(G).edges() if u != v],
                     dtype=np.uint32)
    return edges, gt


def ensure(n, mu, seed, log=print):
    p = path_for(n, mu, seed)
    if os.path.exists(p):
        return p
    os.makedirs(LFR_DIR, exist_ok=True)
    t0 = time.time()
    edges, gt = generate(n, mu, seed)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, edges=edges, gt=gt)
    os.replace(tmp, p)
    log(f"generated n={n} mu={mu} seed={seed} m={len(edges)} "
        f"in {time.time() - t0:.1f}s", flush=True)
    return p


def all_cells():
    cells = set()
    for n in MU_SWEEP_N:
        for mu in MU_SWEEP_MU:
            cells.add((n, mu))
    for n in NODES_SWEEP_N:
        for mu in NODES_SWEEP_MU:
            cells.add((n, mu))
    return sorted(cells)


if __name__ == "__main__":
    cells = all_cells()
    total = len(cells) * RUNS
    done = 0
    for n, mu in cells:
        for seed in range(RUNS):
            ensure(n, mu, seed)
            done += 1
            print(f"[{done}/{total}] n={n} mu={mu} seed={seed} ready",
                  flush=True)
    print("LFR_CACHE_DONE", flush=True)
