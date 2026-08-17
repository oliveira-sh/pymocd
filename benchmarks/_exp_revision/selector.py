"""Label-free selection rules over one stored front (M8, Q11).

Every rule sees the identical candidate set: the refined rank-1 front SMOCC
hands to its selector. The oracle row is an upper bound, not an achievable
result. Rules:

    deployed   min-max normalised sum of the four objectives (the shipped rule)
    maxq       maximum Newman-Girvan modularity
    knee       least Euclidean distance to the utopia point of the normalised
               four-objective front
    knee2      the same over the two search objectives only
    mdl        least description length of a degree-corrected block model
    kmed       the member whose community count is the median of the front
    random     a uniformly drawn member, averaged over the front (expectation)
    oracle     the member of maximal AMI against the ground truth

    python -m _exp_revision.selector
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Shim, Sink, densify, lfr  # noqa: E402

FIELDS = ["net", "kind", "n_cfg", "mu", "seed", "rule", "n", "m", "front_size",
          "k", "gt_k", "nmi", "ami", "ari", "modularity", "time", "stamp"]
KEY = ["net", "n_cfg", "mu", "seed", "rule"]


def objectives(n, edges, lab):
    """(intra, inter, kkm, rc) for one partition, matching the Rust code."""
    _, c = np.unique(lab, return_inverse=True)
    k = c.max() + 1
    m = len(edges)
    eu, ev = c[edges[:, 0]], c[edges[:, 1]]
    same = eu == ev
    l_in = np.bincount(eu[same], minlength=k)
    deg = np.bincount(edges.ravel(), minlength=n)
    vol = np.bincount(c, weights=deg, minlength=k)
    size = np.bincount(c, minlength=k).astype(float)
    intra = 1.0 - same.sum() / m
    inter = float(((vol / (2.0 * m)) ** 2).sum())
    kkm = 2.0 * (n - k) - float((2.0 * l_in / size).sum())
    rc = float(((vol - 2.0 * l_in) / size).sum())
    return intra, inter, kkm, rc, k


def modularity(n, edges, lab):
    _, c = np.unique(lab, return_inverse=True)
    k = c.max() + 1
    m = len(edges)
    eu, ev = c[edges[:, 0]], c[edges[:, 1]]
    l_in = np.bincount(eu[eu == ev], minlength=k)
    deg = np.bincount(edges.ravel(), minlength=n)
    vol = np.bincount(c, weights=deg, minlength=k)
    return float((l_in / m - (vol / (2.0 * m)) ** 2).sum())


def dcsbm_dl(n, edges, lab):
    """Description length of a degree-corrected block model, in nats.

    Entropy term of Karrer-Newman plus the partition and block-matrix
    description costs used by Peixoto's minimum-description-length criterion.
    """
    _, c = np.unique(lab, return_inverse=True)
    k = int(c.max() + 1)
    m = len(edges)
    eu, ev = c[edges[:, 0]], c[edges[:, 1]]
    e_rs = np.zeros((k, k))
    np.add.at(e_rs, (eu, ev), 1.0)
    np.add.at(e_rs, (ev, eu), 1.0)
    deg = np.bincount(edges.ravel(), minlength=n).astype(float)
    e_r = e_rs.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(e_rs > 0, e_rs * np.log(e_rs / np.outer(e_r, e_r)), 0.0)
    entropy = -0.5 * t.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy -= float(np.where(deg > 0, deg * np.log(deg), 0.0).sum())
    nr = np.bincount(c, minlength=k).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        part = n * np.log(n) - float(np.where(nr > 0, nr * np.log(nr), 0).sum())
    nb = k * (k + 1) / 2
    blocks = nb * np.log1p(m / max(nb, 1.0)) + m * np.log1p(nb / max(m, 1.0))
    return entropy + part + blocks


def pick(rule, n, edges, fronts, gt):
    obj = np.array([objectives(n, edges, p)[:4] for p in fronts])
    if rule == "deployed":
        lo, hi = obj.min(axis=0), obj.max(axis=0)
        rng = hi - lo
        z = np.where(rng > 0, (obj - lo) / np.where(rng > 0, rng, 1.0), 0.0)
        return int(z.sum(axis=1).argmin())
    if rule == "maxq":
        return int(np.argmax([modularity(n, edges, p) for p in fronts]))
    if rule in ("knee", "knee2"):
        cols = slice(0, 4) if rule == "knee" else slice(0, 2)
        o = obj[:, cols]
        lo, hi = o.min(axis=0), o.max(axis=0)
        rng = hi - lo
        z = np.where(rng > 0, (o - lo) / np.where(rng > 0, rng, 1.0), 0.0)
        return int((z ** 2).sum(axis=1).argmin())
    if rule == "mdl":
        return int(np.argmin([dcsbm_dl(n, edges, p) for p in fronts]))
    if rule == "kmed":
        ks = np.array([len(np.unique(p)) for p in fronts])
        return int(np.abs(ks - np.median(ks)).argmin())
    if rule == "oracle":
        from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
        return int(np.argmax([ami(gt, p) for p in fronts]))
    raise ValueError(rule)


RULES = ["deployed", "maxq", "knee", "knee2", "mdl", "kmed", "oracle"]


def evaluate(sink, net, kind, n_cfg, mu, seed, n, edges, gt, fronts, elapsed):
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         normalized_mutual_info_score)
    rules = RULES if gt is not None else [r for r in RULES if r != "oracle"]
    for rule in rules:
        base = {"net": net, "kind": kind, "n_cfg": n_cfg, "mu": mu,
                "seed": seed, "rule": rule}
        if sink.has(base):
            continue
        t0 = time.perf_counter()
        j = pick(rule, n, edges, fronts, gt)
        dt = time.perf_counter() - t0
        lab = fronts[j]
        row = dict(base, n=n, m=len(edges), front_size=len(fronts),
                   k=int(len(np.unique(lab))), time=round(dt, 4),
                   modularity=round(modularity(n, edges, lab), 6))
        if gt is not None:
            row["gt_k"] = int(len(np.unique(gt)))
            row["nmi"] = round(float(normalized_mutual_info_score(gt, lab)), 6)
            row["ami"] = round(float(adjusted_mutual_info_score(gt, lab)), 6)
            row["ari"] = round(float(adjusted_rand_score(gt, lab)), 6)
        sink.write(row)

    # The expected score of a uniformly drawn front member, as a floor.
    base = {"net": net, "kind": kind, "n_cfg": n_cfg, "mu": mu, "seed": seed,
            "rule": "random"}
    if not sink.has(base) and gt is not None:
        a = [adjusted_mutual_info_score(gt, p) for p in fronts]
        nm = [normalized_mutual_info_score(gt, p) for p in fronts]
        ar = [adjusted_rand_score(gt, p) for p in fronts]
        sink.write(dict(base, n=n, m=len(edges), front_size=len(fronts),
                        k=int(np.mean([len(np.unique(p)) for p in fronts])),
                        gt_k=int(len(np.unique(gt))),
                        nmi=round(float(np.mean(nm)), 6),
                        ami=round(float(np.mean(a)), 6),
                        ari=round(float(np.mean(ar)), 6), time=0,
                        modularity=""))


def main():
    import pymocd
    from _exp_real_net.networks import LOADERS
    pymocd.max_cores(int(os.environ.get("REV_THREADS", "48")))
    sink = Sink("selector.csv", FIELDS, KEY)

    # A front is |F| partitions of n labels; materialising one for a graph of
    # a million vertices costs tens of gigabytes in Python objects, so the
    # real-network arm stops at the two SNAP graphs that fit comfortably.
    small = os.environ.get(
        "REV_SEL_NETS",
        "karate,dolphins,polbooks,football,email_eu,lesmis,florentine,"
        "dblp,amazon").split(",")
    for net in small:
        if net not in LOADERS:
            print(f"skip unknown net {net}", flush=True)
            continue
        edges, n, gt, eval_nodes = LOADERS[net]()
        t0 = time.perf_counter()
        res = pymocd.smocc_probe(Shim(n, edges))
        dt = time.perf_counter() - t0
        fronts = [densify(p, n) for p in res["front"]]
        if gt is None:
            evaluate(sink, net, "real", "", "", 0, n, edges, None, fronts, dt)
        else:
            sub = [p[eval_nodes] for p in fronts]
            evaluate(sink, net, "real", "", "", 0, n, edges, gt, sub, dt)
        print(f"{net}: |F|={len(fronts)} in {dt:.1f}s", flush=True)

    grid = json.loads(os.environ.get(
        "REV_SEL_LFR", '[[10000,0.3],[10000,0.5],[50000,0.3],[50000,0.5],'
                       '[50000,0.6],[100000,0.5]]'))
    runs = int(os.environ.get("REV_SEL_RUNS", "5"))
    for n_cfg, mu in grid:
        for seed in range(runs):
            edges, gt = lfr(n_cfg, mu, seed)
            t0 = time.perf_counter()
            res = pymocd.smocc_probe(Shim(n_cfg, edges))
            dt = time.perf_counter() - t0
            fronts = [densify(p, n_cfg) for p in res["front"]]
            evaluate(sink, f"lfr{n_cfg}", "lfr", n_cfg, mu, seed, n_cfg,
                     edges, gt, fronts, dt)
            print(f"lfr n={n_cfg} mu={mu} s={seed}: |F|={len(fronts)} "
                  f"in {dt:.1f}s", flush=True)
    print(f"SELECTOR_DONE -> {os.path.join(OUT, 'selector.csv')}", flush=True)


if __name__ == "__main__":
    main()
