#!/usr/bin/env python3
"""Reproduce the MMCoMO paper (Zhang, Yang, Yang & Zhang, IEEE CIM).

Checks `pymocd.mmcomo` against the published Q (Table III) and NMI (Table IV)
on the ground-truth real-world nets, and the LFR n=1000 mu-sweep trend (Fig. 6a).

Selection rules (matching the paper):
  * Q   table -> max modularity over the merged rank-1 front (label-free; this is
                 exactly what `mmcomo()` returns).
  * NMI table -> max NMI vs ground truth over the front (the paper's GT rule).
Both are read from a single `mmcomo_fronts()` call per run.

  python3 -u tests/benchmarks/mmcomo_reproduce.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pymocd
from sklearn.metrics import normalized_mutual_info_score as nmi_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "real_world_nets/data"
N_RUNS = 10
LFR_RUNS = 3


def load_gml(path):
    G = nx.read_gml(path)
    remap = {o: i for i, o in enumerate(G.nodes())}
    lab, gt = {}, {}
    for o in G.nodes():
        d = G.nodes[o]
        raw = d.get("value", d.get("group", d.get("gt", d.get("club", 0))))
        gt[remap[o]] = lab.setdefault(raw, len(lab))
    H = nx.Graph()
    H.add_nodes_from(range(len(remap)))
    for a, b in G.edges():
        if a != b:
            H.add_edge(remap[a], remap[b])
    return H, gt


def load_snap(ungraph, cmty):
    eset, nodes = set(), set()
    with open(ungraph) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b = (int(x) for x in line.split()[:2])
            if a != b:
                eset.add((a, b) if a < b else (b, a))
            nodes.add(a); nodes.add(b)
    gt_raw = {}
    with open(cmty) as f:
        for k, line in enumerate(f):
            for v in (int(x) for x in line.split()):
                gt_raw.setdefault(v, k)
                nodes.add(v)
    remap = {o: i for i, o in enumerate(sorted(nodes))}
    H = nx.Graph()
    H.add_nodes_from(range(len(remap)))
    for a, b in eset:
        H.add_edge(remap[a], remap[b])
    gt = {remap[o]: c for o, c in gt_raw.items()}
    return H, gt


# Tuple fields: (Q Table III, NMI Table IV, dQ tol, dNMI tol).
NETWORKS = [
    ("karate",   lambda: load_gml(DATA / "small/karate.gml"),   0.4198, 1.0000, 0.03, 0.05),
    ("dolphins", lambda: load_gml(DATA / "small/dolphins.gml"), 0.5268, 0.8888, 0.03, 0.06),
    ("football", lambda: load_gml(DATA / "small/football.gml"), 0.6046, 0.9180, 0.03, 0.05),
    ("polbooks", lambda: load_gml(DATA / "small/polbooks.gml"), 0.5272, 0.5374, 0.03, 0.10),
    # email-Eu-core substitutes for the paper's 5th net (no yeast data); info only.
    ("email",    lambda: load_snap(DATA / "email/email-Eu-core.txt",
                                   DATA / "email/email-Eu-core.cmty.txt"),
                                   0.5818, 0.3290, None, None),
]


def score_front(G, gt, fronts):
    """Return (best-Q, best-NMI) over the front members."""
    order = sorted(gt)
    y_true = [gt[n] for n in order]
    best_q, best_nmi = -1.0, -1.0
    for part in fronts:
        comms = {}
        for n, c in part.items():
            if c != -1:
                comms.setdefault(c, set()).add(n)
        q = nx.community.modularity(G, comms.values(), weight=None)
        y_pred = [part.get(n, -1) for n in order]
        nmi = nmi_score(y_true, y_pred)
        best_q = max(best_q, q)
        best_nmi = max(best_nmi, nmi)
    return best_q, best_nmi


def run_realworld():
    print("=" * 78)
    print("MMCoMO vs paper Tables III (Q) / IV (NMI) — real-world ground-truth nets")
    print(f"({N_RUNS} runs each; Q=max-Q over front, NMI=max-NMI over front)")
    print("=" * 78)
    hdr = f"{'net':<10}{'n':>5}{'Q (ours)':>16}{'Q*':>8}{'NMI (ours)':>16}{'NMI*':>8}  verdict"
    print(hdr)
    print("-" * 78)
    all_pass = True
    for name, loader, q_tgt, nmi_tgt, dq, dnmi in NETWORKS:
        G, gt = loader()
        qs, nmis = [], []
        for _ in range(N_RUNS):
            fronts = pymocd.mmcomo_fronts(G)
            q, nmi = score_front(G, gt, fronts)
            qs.append(q); nmis.append(nmi)
        qm, qs_ = np.mean(qs), np.std(qs)
        nm, ns_ = np.mean(nmis), np.std(nmis)
        gated = dq is not None
        if gated:
            q_ok = abs(qm - q_tgt) <= dq
            n_ok = abs(nm - nmi_tgt) <= dnmi
            verdict = "PASS" if (q_ok and n_ok) else ("Q✗" if not q_ok else "") + ("NMI✗" if not n_ok else "")
            verdict = "PASS" if (q_ok and n_ok) else "FAIL " + verdict
            all_pass = all_pass and q_ok and n_ok
        else:
            verdict = "(info)"
        print(f"{name:<10}{G.number_of_nodes():>5}"
              f"{qm:>10.4f}±{qs_:.3f}{q_tgt:>8.4f}"
              f"{nm:>10.4f}±{ns_:.3f}{nmi_tgt:>8.4f}  {verdict}")
    print("-" * 78)
    print(f"gated nets: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def make_lfr(n, mu, seed):
    # Paper uses tau1=2/tau2=1, infeasible in networkx at average_degree=20;
    # tau1=3/tau2=1.5 is the robust combo and the mu-trend is insensitive to both.
    for s in range(seed, seed + 25):
        try:
            G = nx.LFR_benchmark_graph(
                n=n, tau1=3.0, tau2=1.5, mu=mu,
                average_degree=20, max_degree=50,
                min_community=20, max_community=100,
                seed=s, max_iters=2000,
            )
            G.remove_edges_from(nx.selfloop_edges(G))
            gt = {}
            for v in G.nodes():
                gt[v] = min(G.nodes[v]["community"])  # canonical community id
            return G, gt
        except (nx.ExceededMaxIterations, nx.NetworkXError):
            continue
    return None, None


def run_lfr():
    print()
    print("=" * 78)
    print("LFR n=1000 mu-sweep (Fig. 6a): expect strong NMI for mu<=0.6, dropoff mu>0.6")
    print(f"(d_ave=20, d_max=50, comm 20-100, tau1=3, tau2=1.5 [networkx-feasible];"
          f" {LFR_RUNS} runs/point, max-NMI)")
    print("=" * 78)
    print(f"{'mu':>6}{'NMI (ours)':>16}{'k_found':>10}")
    print("-" * 78)
    results = {}
    for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        nmis, ks = [], []
        for r in range(LFR_RUNS):
            G, gt = make_lfr(1000, mu, seed=100 + 31 * r)
            if G is None:
                continue
            fronts = pymocd.mmcomo_fronts(G)
            order = sorted(gt)
            y_true = [gt[n] for n in order]
            best, bk = -1.0, 0
            for part in fronts:
                y_pred = [part.get(n, -1) for n in order]
                v = nmi_score(y_true, y_pred)
                if v > best:
                    best, bk = v, len(set(y_pred))
            nmis.append(best); ks.append(bk)
        if not nmis:
            print(f"{mu:>6.1f}{'(LFR gen failed)':>16}")
            continue
        nm = np.mean(nmis)
        results[mu] = nm
        print(f"{mu:>6.1f}{nm:>10.4f}±{np.std(nmis):.3f}{int(np.mean(ks)):>10}")
    print("-" * 78)
    lo = [results[m] for m in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6) if m in results]
    hi = [results[m] for m in (0.7, 0.8) if m in results]
    trend_ok = bool(lo) and bool(hi) and (np.mean(lo) - np.mean(hi) > 0.1) and np.mean(lo) > 0.8
    print(f"trend mu<=0.6 mean={np.mean(lo):.3f}  mu>0.6 mean={np.mean(hi):.3f}  "
          f"-> {'OK (strong then dropoff)' if trend_ok else 'CHECK'}")
    return trend_ok


if __name__ == "__main__":
    t0 = time.time()
    rw = run_realworld()
    tr = run_lfr()
    print()
    print("=" * 78)
    print(f"DONE in {time.time()-t0:.0f}s | real-world gated: "
          f"{'PASS' if rw else 'FAIL'} | LFR trend: {'PASS' if tr else 'CHECK'}")
    print("=" * 78)
    sys.exit(0 if rw else 1)
