"""Can post-processing fix the over-fragmentation? (design probe)

The campaign shows one failure mode: above a mixing of 0.4 SMOCC returns far
more communities than the graph holds, its homogeneity stays high and its
completeness collapses. The front oracle is only 0.015 AMI above the deployed
pick, so the front does not already contain the answer and a better selector
cannot supply it. Something has to build coarser candidates.

This probe scores three cheap ways of building them, on the fronts the shipped
algorithm actually returns:

    agglom     greedy agglomeration of adjacent communities under an objective,
               emitting every level as a candidate, in the spirit of the second
               phase of Louvain but driven by the evolved similarity
    consensus  connected components of the edges on which at least a fraction
               of the front's members agree, which the similarity update
               already computes
    both       agglomeration applied to the consensus partition

Each is scored against the ground truth and against the deployed choice, so a
gain here is an upper bound on what the same idea would buy inside the search.

    python -m _exp_revision.merge_probe
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Shim, Sink, densify, lfr  # noqa: E402
from _exp_revision.selector import modularity, objectives, pick  # noqa: E402

FIELDS = ["net", "kind", "n_cfg", "mu", "seed", "variant", "n", "m",
          "front_size", "k", "gt_k", "nmi", "ami", "ari", "hom", "cmp",
          "modularity", "time", "stamp"]
KEY = ["net", "n_cfg", "mu", "seed", "variant"]


def community_graph(n, edges, lab):
    """Communities as nodes, summed edge weights between them."""
    _, c = np.unique(lab, return_inverse=True)
    k = int(c.max() + 1)
    eu, ev = c[edges[:, 0]], c[edges[:, 1]]
    keep = eu != ev
    a, b = eu[keep], ev[keep]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * k + hi
    uniq, cnt = np.unique(key, return_counts=True)
    src, dst = (uniq // k).astype(np.int64), (uniq % k).astype(np.int64)
    deg = np.bincount(edges.ravel(), minlength=n)
    vol = np.bincount(c, weights=deg, minlength=k)
    internal = np.bincount(c[eu[~keep]], minlength=k)
    return c, k, src, dst, cnt.astype(float), vol, internal


def agglomerate(n, edges, lab, m, levels=12):
    """Greedy merge of adjacent communities by modularity gain.

    Returns a chain of partitions from the input granularity down to the
    coarsest the criterion accepts, sampled at `levels` points. Emitting the
    chain rather than its endpoint matters: the endpoint is where modularity's
    resolution limit has finished merging, which is far past the planted
    granularity, while some level along the way is close to it.
    """
    c, k, src, dst, w, vol, internal = community_graph(n, edges, lab)
    parent = np.arange(k)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    two_m = 2.0 * m
    # Modularity gain of merging r and s: 2*(w_rs/2m - vol_r*vol_s/(2m)^2).
    gain = w / m - (vol[src] * vol[dst]) / (two_m * two_m) * 2.0
    order = np.argsort(-gain)
    vol_w = vol.copy()
    out = []
    for idx in order:
        if gain[idx] <= 0:
            break
        r, s = find(src[idx]), find(dst[idx])
        if r == s:
            continue
        g = w[idx] / m - 2.0 * vol_w[r] * vol_w[s] / (two_m * two_m)
        if g <= 0:
            continue
        parent[s] = r
        vol_w[r] += vol_w[s]
        out.append((r, s))
    if not out:
        return None
    if not out:
        return []
    # Replay the accepted merges, snapshotting the partition at `levels`
    # evenly spaced points along the chain.
    marks = set(np.unique(np.linspace(0, len(out) - 1, levels).astype(int)))
    parent = np.arange(k)
    chain = []
    for i, (r, s_) in enumerate(out):
        parent[find(s_)] = find(r)
        if i in marks:
            root = np.array([find(j) for j in range(k)])
            chain.append(root[c])
    return chain


def co_association(n, edges, front):
    """Fraction of front members placing each edge's endpoints together."""
    agree = np.zeros(len(edges), dtype=np.float64)
    for p in front:
        agree += (p[edges[:, 0]] == p[edges[:, 1]])
    return agree / max(len(front), 1)


def consensus(n, edges, agree, tau):
    """Connected components of the edges at least `tau` of the front agree on."""
    keep = edges[agree >= tau]
    lab = np.arange(n, dtype=np.int64)

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in keep:
        a, b = find(int(u)), find(int(v))
        if a != b:
            parent[b] = a
    return np.array([find(i) for i in range(n)])


def score(gt, lab, n, edges):
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         homogeneity_completeness_v_measure,
                                         normalized_mutual_info_score)
    hom, cmp_, _ = homogeneity_completeness_v_measure(gt, lab)
    return {"k": int(len(np.unique(lab))),
            "nmi": round(float(normalized_mutual_info_score(gt, lab)), 6),
            "ami": round(float(adjusted_mutual_info_score(gt, lab)), 6),
            "ari": round(float(adjusted_rand_score(gt, lab)), 6),
            "hom": round(float(hom), 6), "cmp": round(float(cmp_), 6),
            "modularity": round(modularity(n, edges, lab), 6)}


TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def variants(n, edges, front, sel, gt, m):
    """Every candidate this probe compares, as {name: partition}."""
    out = {"deployed": front[sel]}

    chain = agglomerate(n, edges, front[sel], m)
    if chain:
        pool = [front[sel]] + chain
        out["agglom"] = pool[pick("deployed", n, edges, pool, gt)]
        if gt is not None:
            out["agglom_oracle"] = pool[pick("oracle", n, edges, pool, gt)]

    agree = co_association(n, edges, front)
    cons = []
    for tau in TAUS:
        c = consensus(n, edges, agree, tau)
        if len(np.unique(c)) > 1:
            cons.append(c)
    if cons:
        out["consensus"] = cons[pick("deployed", n, edges, cons, gt)]
        if gt is not None:
            out["consensus_oracle"] = cons[pick("oracle", n, edges, cons, gt)]
        grown = list(cons)
        for c in cons:
            grown.extend(agglomerate(n, edges, c, m))
        out["consensus_agglom"] = grown[pick("deployed", n, edges, grown, gt)]
        if gt is not None:
            out["consensus_agglom_oracle"] = grown[
                pick("oracle", n, edges, grown, gt)]

    # Everything at once, chosen label-free and by oracle: the ceiling of the
    # whole idea against what a deployed rule could actually take from it.
    pool = [front[sel]] + list(chain or [])
    pool += [v for k_, v in out.items() if k_ != "deployed"]
    out["all"] = pool[pick("deployed", n, edges, pool, gt)]
    if gt is not None:
        out["all_oracle"] = pool[pick("oracle", n, edges, pool, gt)]
    return out


def main():
    import pymocd
    pymocd.max_cores(int(os.environ.get("REV_THREADS", "12")))
    sink = Sink("merge_probe.csv", FIELDS, KEY)
    grid = json.loads(os.environ.get(
        "REV_MP_LFR",
        '[[10000,0.5],[10000,0.6],[50000,0.5],[50000,0.6],[50000,0.7],'
        '[100000,0.5],[100000,0.6]]'))
    runs = int(os.environ.get("REV_MP_RUNS", "5"))
    for n, mu in grid:
        for seed in range(runs):
            if sink.has({"net": f"lfr{n}", "n_cfg": n, "mu": mu, "seed": seed,
                         "variant": "all_oracle"}):
                continue
            edges, gt = lfr(n, mu, seed)
            m = len(edges)
            t0 = time.perf_counter()
            res = pymocd.smocc_probe(Shim(n, edges))
            t_search = time.perf_counter() - t0
            front = [densify(p, n) for p in res["front"]]
            for name, lab in variants(n, edges, front, res["selected"], gt,
                                      m).items():
                row = {"net": f"lfr{n}", "kind": "lfr", "n_cfg": n, "mu": mu,
                       "seed": seed, "variant": name, "n": n, "m": m,
                       "front_size": len(front),
                       "gt_k": int(len(np.unique(gt))),
                       "time": round(t_search, 3)}
                row.update(score(gt, lab, n, edges))
                sink.write(row)
            print(f"n={n} mu={mu} s={seed} done in {t_search:.1f}s", flush=True)
    print(f"MERGE_PROBE_DONE -> {os.path.join(OUT, 'merge_probe.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
