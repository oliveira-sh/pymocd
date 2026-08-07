import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class Shim:
    def __init__(self, n, e):
        self._n, self._e = n, e

    def nodes(self):
        return range(self._n)

    def edges(self):
        e = self._e
        for i in range(0, len(e), 1_000_000):
            yield from map(tuple, e[i:i + 1_000_000].tolist())


def build_nx(n, edges):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges.tolist()))
    return G


def run_algorithm(alg, n, edges, seed, threads):
    import pymocd
    pymocd.max_cores(threads)
    shim = Shim(n, edges)
    lib = {
        "SMOCC": lambda: pymocd.smocc(shim),
        "HP-MOCD": lambda: pymocd.hpmocd(shim),
        "MMCoMO": lambda: pymocd.mmcomo(shim),
        "NSGA-III CCM": lambda: pymocd.ccm(shim),
        "NSGA-III KRM": lambda: pymocd.krm(shim),
        "Shi-MOCD (Q)": lambda: pymocd.mocd_q(shim),
        "Shi-MOCD (D)": lambda: pymocd.mocd_d(shim),
        "MOGA-Net": lambda: pymocd.moga_net(shim),
    }
    if alg in lib:
        t0 = time.perf_counter()
        part = lib[alg]()
        dt = time.perf_counter() - t0
        lab = np.full(n, -1, dtype=np.int64)
        for node, c in part.items():
            lab[node] = c
        unassigned = lab == -1
        lab[unassigned] = np.arange(n, dtype=np.int64)[unassigned] + n
        return lab, dt
    if alg == "Leiden":
        import igraph as ig
        g = ig.Graph(n=n, edges=edges)
        t0 = time.perf_counter()
        part = g.community_leiden(objective_function="modularity")
        dt = time.perf_counter() - t0
        return np.asarray(part.membership, dtype=np.int64), dt
    if alg == "Louvain":
        import community as community_louvain
        G = build_nx(n, edges)
        t0 = time.perf_counter()
        part = community_louvain.best_partition(G, random_state=seed)
        dt = time.perf_counter() - t0
        return np.array([part[v] for v in range(n)], dtype=np.int64), dt
    if alg == "ASYN-LPA":
        from networkx.algorithms.community import asyn_lpa_communities
        G = build_nx(n, edges)
        t0 = time.perf_counter()
        comms = list(asyn_lpa_communities(G, seed=seed))
        dt = time.perf_counter() - t0
        lab = np.empty(n, dtype=np.int64)
        for ci, comm in enumerate(comms):
            for v in comm:
                lab[v] = ci
        return lab, dt
    raise ValueError(f"unknown algorithm {alg}")


def main():
    task = json.loads(sys.argv[1])
    kind, seed, threads = task["kind"], task["seed"], task["threads"]

    if kind == "lfr":
        from _exp_synt_net.gen_graphs import ensure
        n_cfg, mu = task["n_cfg"], task["mu"]
        data = np.load(ensure(n_cfg, mu, seed, log=lambda *a, **k: None))
        edges, gt = data["edges"], data["gt"].astype(np.int64)
        n = n_cfg
        eval_nodes = np.arange(n, dtype=np.int64)
    else:
        from _exp_real_net.networks import LOADERS
        edges, n, gt, eval_nodes = LOADERS[task["net"]]()

    lab, dt = run_algorithm(task["alg"], n, edges, seed, threads)

    import igraph as ig
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         normalized_mutual_info_score)
    _, lab_dense = np.unique(lab, return_inverse=True)
    mod = float(ig.Graph(n=n, edges=edges).modularity(lab_dense.tolist()))
    out = {"status": "ok", "n": int(n), "m": int(len(edges)),
           "k": int(len(np.unique(lab))), "time": round(dt, 4),
           "modularity": round(mod, 6), "nmi": "", "ami": ""}
    if gt is not None:
        pred = lab[eval_nodes]
        out["nmi"] = round(float(normalized_mutual_info_score(gt, pred)), 6)
        out["ami"] = round(float(adjusted_mutual_info_score(gt, pred)), 6)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
