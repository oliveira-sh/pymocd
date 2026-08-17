"""Do the campaign's conclusions survive a correct LFR generator (M10)?

Scores the campaign's detector configurations, copied verbatim from
``_exp_synt_net/hardened_worker.py``, on the corrected graphs of
``lfr_reference.py``. Writes one resumable row per (detector, n, mu, seed) and
prints mean AMI per detector per mu.

Rayon-parallel detectors get the whole thread budget one run at a time; the
single-threaded ones fan out over a process pool of the same size.

    REV_THREADS=8 python -m _exp_revision.lfr_reference_run
    REF_N=1000 REF_MU=0.1,0.5 REF_SEEDS=5 python -m _exp_revision.lfr_reference_run
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision import lfr_reference  # noqa: E402
from _exp_revision.common import Shim, Sink, densify, score  # noqa: E402
from _exp_revision.common import _env_list  # noqa: E402

RUN_N = _env_list("REF_N", [1_000, 10_000])
RUN_MU = _env_list("REF_MU", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
RUN_SEEDS = int(os.environ.get("REF_SEEDS", "5"))
THREADS = int(os.environ.get("REV_THREADS", "8"))

# Tuple order = run order. The first two are rayon-parallel and take the whole
# budget alone; the rest are single-threaded and share it across processes.
PARALLEL_ALGS = ("SMOCC", "HP-MOCD")
ALGORITHMS = PARALLEL_ALGS + ("Leiden", "Leiden-CPM", "Louvain", "ASYN-LPA")

FIELDS = ["alg", "n_cfg", "mu", "seed", "seed_value", "threads", "status",
          "n", "m", "k", "gt_k", "mu_real", "time", "nmi", "ami", "ari",
          "modularity", "stamp"]
KEY = ["alg", "n_cfg", "mu", "seed"]


_CORES = []


def run_algorithm(alg, n, edges, seed, threads):
    """The campaign's invocations, from _exp_synt_net/hardened_worker.py."""
    if alg in ("SMOCC", "HP-MOCD"):
        import pymocd
        if not _CORES:  # the rayon pool is global and set once per process
            pymocd.max_cores(threads)
            _CORES.append(threads)
        shim = Shim(n, edges)
        lib = {"SMOCC": lambda: pymocd.smocc(shim),
               "HP-MOCD": lambda: pymocd.hpmocd(shim)}
        t0 = time.perf_counter()
        part = lib[alg]()
        dt = time.perf_counter() - t0
        return densify(part, n), dt
    if alg in ("Leiden", "Leiden-CPM"):
        import igraph as ig
        g = ig.Graph(n=n, edges=edges)
        kw = {"objective_function": "modularity"}
        if alg == "Leiden-CPM":
            kw = {"objective_function": "CPM",
                  "resolution": len(edges) / (n * (n - 1) / 2)}
        t0 = time.perf_counter()
        part = g.community_leiden(**kw)
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


def build_nx(n, edges):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges.tolist()))
    return G


def work(task):
    alg, n, mu, seed, threads = task
    d = np.load(lfr_reference.ensure(n, mu, seed, log=lambda *a, **k: None))
    edges, gt = d["edges"], d["gt"].astype(np.int64)
    sval = int(d["seed_value"])
    try:
        lab, dt = run_algorithm(alg, n, edges, seed, threads)
        st = "ok"
    except Exception as e:
        return dict(alg=alg, n_cfg=n, mu=mu, seed=seed, seed_value=sval,
                    threads=threads, status=f"error:{type(e).__name__}")
    e64 = edges.astype(np.int64)
    return dict(alg=alg, n_cfg=n, mu=mu, seed=seed, seed_value=sval,
                threads=threads, status=st, n=int(n), m=int(len(edges)),
                gt_k=int(len(np.unique(gt))),
                mu_real=round(float((gt[e64[:, 0]] != gt[e64[:, 1]]).mean()), 6),
                time=round(dt, 4),
                **score(gt, lab, n=int(n), edges=edges))


def report(path):
    acc = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            acc.setdefault((int(r["n_cfg"]), r["alg"], float(r["mu"])),
                           []).append(float(r["ami"]))
    for n in RUN_N:
        mus = sorted({mu for (nn, _, mu) in acc if nn == n})
        if not mus:
            continue
        print()
        print(f"mean AMI on reference LFR, n = {n}, "
              f"{RUN_SEEDS} graph seeds")
        head = f"{'detector':<12}" + "".join(f"{mu:>8.1f}" for mu in mus)
        print(head)
        print("-" * len(head))
        for alg in ALGORITHMS:
            row = [acc.get((n, alg, mu)) for mu in mus]
            if not any(row):
                continue
            print(f"{alg:<12}" + "".join(
                f"{np.mean(v):>8.3f}" if v else f"{'-':>8}" for v in row))


def main():
    lfr_reference.build()
    sink = Sink("lfr_reference.csv", FIELDS, KEY)
    cells = [(n, mu, s) for n in RUN_N for mu in RUN_MU
             for s in range(RUN_SEEDS)]
    for n, mu, s in cells:
        lfr_reference.ensure(n, mu, s, log=lambda *a, **k: None)
    for alg in ALGORITHMS:
        pending = [(alg, n, mu, s, THREADS if alg in PARALLEL_ALGS else 1)
                   for n, mu, s in cells
                   if not sink.has({"alg": alg, "n_cfg": n, "mu": mu,
                                    "seed": s})]
        if not pending:
            continue
        workers = 1 if alg in PARALLEL_ALGS else THREADS
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in pending]
            for i, f in enumerate(as_completed(futs), 1):
                row = f.result()
                sink.write(row)
                print(f"[{alg} {i}/{len(pending)}] n={row['n_cfg']} "
                      f"mu={row['mu']} seed={row['seed']} "
                      f"{row['status']} ami={row.get('ami', '')} "
                      f"k={row.get('k', '')}", flush=True)
        print(f"{alg} done in {time.perf_counter() - t0:.1f}s", flush=True)
    report(sink.path)
    print("\nLFR_REF_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
