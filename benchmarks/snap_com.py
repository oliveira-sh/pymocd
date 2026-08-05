"""SCALE vs Leiden on the SNAP com- networks with ground-truth communities.

Ground truth is the top-5000 quality communities per network; they overlap and
cover only part of the graph, so NMI/AMI are computed on the nodes that belong
to exactly one ground-truth community. com-Friendster (65M nodes / 1.8B edges)
is skipped as infeasible on this machine.
"""

import gc
import gzip
import os
import sys
import time
import urllib.request

import igraph as ig
import numpy as np
import pandas as pd
import pymocd
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "snap_com")
BASE = "https://snap.stanford.edu/data/bigdata/communities/"
NETS = ["dblp", "amazon", "youtube", "lj", "orkut"]


def fetch(fname: str) -> str:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"downloading {fname}", flush=True)
        urllib.request.urlretrieve(BASE + fname, path)
    return path


class Shim:
    """Duck-typed graph over a remapped numpy edge array; pymocd iterates
    nodes()/edges() so the full python edge list never materializes."""

    def __init__(self, n: int, e: np.ndarray):
        self._n, self._e = n, e

    def nodes(self):
        return range(self._n)

    def edges(self):
        e = self._e
        for i in range(0, len(e), 1_000_000):
            yield from map(tuple, e[i : i + 1_000_000].tolist())


def load_net(name: str):
    epath = fetch(f"com-{name}.ungraph.txt.gz")
    e = pd.read_csv(
        epath, sep="\t", comment="#", header=None, dtype=np.int64
    ).values
    ids = np.unique(e)
    e = np.searchsorted(ids, e).astype(np.int64)

    cpath = fetch(f"com-{name}.top5000.cmty.txt.gz")
    gt: dict[int, int] = {}
    multi: set[int] = set()
    with gzip.open(cpath, "rt") as f:
        for ci, line in enumerate(f):
            for tok in line.split():
                v = int(tok)
                if v in gt and gt[v] != ci:
                    multi.add(v)
                gt[v] = ci
    singles = np.array(sorted(set(gt) - multi), dtype=np.int64)
    pos = np.searchsorted(ids, singles)
    present = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == singles)
    eval_nodes = pos[present]
    eval_labels = np.array([gt[int(v)] for v in singles[present]], dtype=np.int64)
    return len(ids), e, eval_nodes, eval_labels


def evaluate(lab, g_ig, eval_nodes, eval_labels):
    pred = lab[eval_nodes]
    return {
        "nmi": float(normalized_mutual_info_score(eval_labels, pred)),
        "ami": float(adjusted_mutual_info_score(eval_labels, pred)),
        "modularity": float(g_ig.modularity(lab.tolist())),
        "k": int(len(np.unique(lab))),
    }


def append_row(row: dict):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "snap_com.csv")
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )
    print(row, flush=True)


def main():
    pymocd.max_cores(int(os.environ.get("PYMOCD_MAX_CORES", "6")))
    only = set(sys.argv[1:])
    for name in NETS:
        if only and name not in only:
            continue
        print(f"=== com-{name} ===", flush=True)
        try:
            n, e, eval_nodes, eval_labels = load_net(name)
        except Exception as ex:
            print(f"com-{name} load failed: {ex}", flush=True)
            continue
        m = len(e)
        cov = len(eval_nodes) / n
        print(f"n={n} m={m} eval={len(eval_nodes)} ({cov:.1%})", flush=True)

        g_ig = ig.Graph(n=n, edges=e)

        t0 = time.time()
        part = g_ig.community_leiden(objective_function="modularity")
        t_leiden = time.time() - t0
        lab = np.asarray(part.membership, dtype=np.int64)
        append_row(
            {"network": f"com-{name}", "algorithm": "Leiden", "n": n, "m": m,
             "coverage": cov, "time": t_leiden,
             **evaluate(lab, g_ig, eval_nodes, eval_labels)}
        )

        t0 = time.time()
        part = pymocd.scale(Shim(n, e))
        t_scale = time.time() - t0
        lab = np.zeros(n, dtype=np.int64)
        for k, v in part.items():
            lab[k] = v
        append_row(
            {"network": f"com-{name}", "algorithm": "Scale", "n": n, "m": m,
             "coverage": cov, "time": t_scale,
             **evaluate(lab, g_ig, eval_nodes, eval_labels)}
        )

        del e, g_ig, lab, part, eval_nodes, eval_labels
        gc.collect()
    print("SNAP_COM_DONE", flush=True)


if __name__ == "__main__":
    main()
