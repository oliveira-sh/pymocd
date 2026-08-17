"""The non-modularity baselines the review asks for (M9d, Section 3 comment).

Infomap is the scalable map-equation detector and the control for the
resolution-limit claim; DC-SBM is the generative-model control, a
degree-corrected stochastic block model fitted by description length.

graph-tool, the reference SBM implementation, has no wheel on PyPI and cannot
be installed into the campaign venv, so DC-SBM here is our own agglomerative
fit. It is not equivalent to graph-tool's inference: see ``run_dcsbm``.

Both expose ``run_x(n, edges, seed) -> (labels, seconds)``, the contract of
``_exp_synt_net.hardened_worker.run_algorithm``: ``labels`` is a dense int64
vector of length ``n`` whose isolated nodes get their own label.

    python -m _exp_revision.extra_baselines   # self-check on one cached cell
"""

import os
import sys
import time
from math import lgamma, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# Infomap's stochastic search is seeded explicitly; one trial per seed keeps the
# per-seed spread visible instead of averaging it away inside the library.
INFOMAP_TRIALS = int(os.environ.get("REV_INFOMAP_TRIALS", "1"))

# The DC-SBM seed partition deliberately over-splits (Leiden above resolution 1)
# so the description length can merge downwards from a fine start.
DCSBM_RES = float(os.environ.get("REV_DCSBM_RES", "4.0"))


def densify(n, part):
    """A node -> module dict to a dense label vector, isolated nodes split."""
    lab = np.full(n, -1, dtype=np.int64)
    for node, c in part.items():
        lab[node] = c
    loose = lab == -1
    lab[loose] = np.arange(n, dtype=np.int64)[loose] + n
    return lab


def run_infomap(n, edges, seed):
    """Two-level (flat) Infomap on an undirected unweighted edge list.

    Infomap rejects seed 0, so the campaign seed s is passed as s + 1. Runs are
    bit-reproducible for a fixed seed and vary across seeds.
    """
    import infomap
    idx = np.ascontiguousarray(np.asarray(edges, dtype=np.int64).T)
    im = infomap.Infomap.from_edge_index(idx, num_nodes=n, directed=False)
    t0 = time.perf_counter()
    im.run(two_level=True, flow_model="undirected", seed=seed + 1,
           num_trials=INFOMAP_TRIALS, silent=True)
    dt = time.perf_counter() - t0
    return densify(n, im.get_modules()), dt


def leiden_seed(n, edges, seed, resolution):
    import igraph as ig
    import random
    ig.set_random_number_generator(random.Random(seed))
    g = ig.Graph(n=n, edges=[tuple(map(int, e)) for e in edges])
    part = g.community_leiden(objective_function="modularity",
                              resolution=resolution, n_iterations=2)
    return np.asarray(part.membership, dtype=np.int64)


def _blocks(n, lab, edges):
    """Block edge counts, block degrees, block sizes and degree histograms."""
    b = len(np.unique(lab))
    _, lab = np.unique(lab, return_inverse=True)
    r, s = lab[edges[:, 0]], lab[edges[:, 1]]
    lo, hi = np.minimum(r, s), np.maximum(r, s)
    pair, cnt = np.unique(lo.astype(np.int64) * b + hi, return_counts=True)
    adj = [dict() for _ in range(b)]
    self_e = np.zeros(b, dtype=np.float64)
    for p, c in zip(pair.tolist(), cnt.tolist()):
        u, v = divmod(p, b)
        if u == v:
            self_e[u] = 2.0 * c
        else:
            adj[u][v] = float(c)
            adj[v][u] = float(c)
    kdeg = np.bincount(r, minlength=b).astype(np.float64) \
        + np.bincount(s, minlength=b).astype(np.float64)
    size = np.bincount(lab, minlength=b).astype(np.float64)

    deg = np.bincount(edges[:, 0].astype(np.int64), minlength=n) \
        + np.bincount(edges[:, 1].astype(np.int64), minlength=n)
    hist = [dict() for _ in range(b)]
    key, cnt = np.unique(lab.astype(np.int64) * (deg.max() + 1) + deg,
                         return_counts=True)
    for p, c in zip(key.tolist(), cnt.tolist()):
        hist[p // (int(deg.max()) + 1)][p % (int(deg.max()) + 1)] = float(c)
    return lab, adj, self_e, kdeg, size, hist


def _f(x):
    return x * log(x) if x > 0.0 else 0.0


def _lbinom(a, k):
    if k < 0 or a < k:
        return 0.0
    return lgamma(a + 1.0) - lgamma(k + 1.0) - lgamma(a - k + 1.0)


def _prior(n, m, b, size, kdeg, hist):
    """Peixoto's uniform priors: block affinities, partition, degree sequence.

    ``q(e_r, n_r)``, the number of degree sequences of a block, is taken in the
    small-block form ``binom(e_r - 1, n_r - 1) / n_r!``, which together with the
    within-block degree multiset leaves ``-sum_k ln n^r_k!``.
    """
    p = (_lbinom(b * (b + 1) / 2.0 + m - 1.0, m)
         + _lbinom(n - 1.0, b - 1.0)
         + lgamma(n + 1.0) - sum(lgamma(s + 1.0) for s in size if s > 0))
    for r, h in enumerate(hist):
        if size[r] > 0:
            p += _lbinom(kdeg[r] - 1.0, size[r] - 1.0) \
                - sum(lgamma(c + 1.0) for c in h.values())
    return p


def _loglik(adj, self_e, kdeg):
    """Karrer-Newman degree-corrected profile log-likelihood, endpoint counts."""
    tot = sum(_f(e) for e in self_e) \
        + sum(_f(v) for a in adj for v in a.values())
    return tot - 2.0 * sum(_f(k) for k in kdeg)


def _merge_gain(r, s, adj, self_e, kdeg, size, hist):
    """Pair-specific part of the description-length change for merging r and s.

    The terms that depend on the block count alone are the same for every
    candidate pair, so they are added only when a merge is accepted.
    """
    ers = adj[r].get(s, 0.0)
    d = _f(self_e[r] + self_e[s] + 2.0 * ers) - _f(self_e[r]) - _f(self_e[s]) \
        - 2.0 * _f(ers)
    a, b_ = (adj[r], adj[s]) if len(adj[r]) < len(adj[s]) else (adj[s], adj[r])
    for t, e in a.items():
        if t == r or t == s:
            continue
        o = b_.get(t)
        if o is not None:
            d += 2.0 * (_f(e + o) - _f(e) - _f(o))
    d -= 2.0 * (_f(kdeg[r] + kdeg[s]) - _f(kdeg[r]) - _f(kdeg[s]))

    g = -0.5 * d - lgamma(size[r] + size[s] + 1.0) \
        + lgamma(size[r] + 1.0) + lgamma(size[s] + 1.0) \
        + _lbinom(kdeg[r] + kdeg[s] - 1.0, size[r] + size[s] - 1.0) \
        - _lbinom(kdeg[r] - 1.0, size[r] - 1.0) \
        - _lbinom(kdeg[s] - 1.0, size[s] - 1.0)
    hr, hs = (hist[r], hist[s]) if len(hist[r]) < len(hist[s]) \
        else (hist[s], hist[r])
    for k, c in hr.items():
        o = hs.get(k)
        if o is not None:
            g -= lgamma(c + o + 1.0) - lgamma(c + 1.0) - lgamma(o + 1.0)
    return g


def run_dcsbm(n, edges, seed):
    """Degree-corrected SBM by agglomerative description-length minimisation.

    The seed partition is an over-split Leiden run; adjacent blocks are then
    merged greedily while the microcanonical DC-SBM description length is
    tracked, and the partition is rewound to the minimum.

    Two limits are deliberate and must be quoted with any result. The search
    only coarsens its seed, with no block-splitting or node-level refinement,
    so a structure finer than the seed is unreachable. And ``q(e_r, n_r)`` in
    the degree prior uses the small-block approximation, which is loose for
    large blocks. Empirically the fit tracks the ground truth at low mixing and
    under-resolves it as mixing rises.
    """
    import heapq

    t0 = time.perf_counter()
    lab0 = leiden_seed(n, edges, seed, DCSBM_RES)
    lab0, adj, self_e, kdeg, size, hist = _blocks(n, lab0, edges)
    b = len(adj)
    m = float(len(edges))

    dl = -0.5 * _loglik(adj, self_e, kdeg) + _prior(n, m, b, size, kdeg, hist)
    alive = np.ones(b, dtype=bool)
    stamp = np.zeros(b, dtype=np.int64)
    heap = [(_merge_gain(r, s, adj, self_e, kdeg, size, hist), r, s, 0, 0)
            for r in range(b) for s in adj[r] if r < s]
    heapq.heapify(heap)

    # The block graph densifies as it coarsens, so merging all the way to one
    # block costs more than it can ever win back: stop once the description
    # length has been rising steadily past half the best block count.
    best_dl, best_step, best_b, history = dl, 0, b, []
    while b > 1 and heap:
        gain, r, s, vr, vs = heapq.heappop(heap)
        if not (alive[r] and alive[s]) or vr != stamp[r] or vs != stamp[s]:
            continue
        dl += gain + _lbinom((b - 1) * b / 2.0 + m - 1.0, m) \
            - _lbinom(b * (b + 1) / 2.0 + m - 1.0, m) \
            + _lbinom(n - 1.0, b - 2.0) - _lbinom(n - 1.0, b - 1.0)
        if len(adj[r]) < len(adj[s]):
            r, s = s, r
        history.append((r, s))
        ers = adj[r].pop(s, 0.0)
        adj[s].pop(r, None)
        self_e[r] += self_e[s] + 2.0 * ers
        kdeg[r] += kdeg[s]
        size[r] += size[s]
        for t, e in adj[s].items():
            adj[t].pop(s, None)
            adj[r][t] = adj[r].get(t, 0.0) + e
            adj[t][r] = adj[r][t]
            stamp[t] += 1
        adj[s] = {}
        if len(hist[r]) < len(hist[s]):
            hist[r], hist[s] = hist[s], hist[r]
        for k, c in hist[s].items():
            hist[r][k] = hist[r].get(k, 0.0) + c
        hist[s] = {}
        alive[s] = False
        stamp[r] += 1
        b -= 1
        if dl < best_dl:
            best_dl, best_step, best_b = dl, len(history), b
        elif b * 2 <= best_b:
            break
        for t in adj[r]:
            if alive[t]:
                heapq.heappush(
                    heap, (_merge_gain(r, t, adj, self_e, kdeg, size, hist),
                           r, t, stamp[r], stamp[t]))

    root = np.arange(len(adj), dtype=np.int64)
    for r, s in history[:best_step]:
        root[root == root[s]] = root[r]
    lab = root[lab0]
    return np.unique(lab, return_inverse=True)[1].astype(np.int64), \
        time.perf_counter() - t0


ALGORITHMS = {"Infomap": run_infomap, "DC-SBM": run_dcsbm}


def main():
    from _exp_revision.common import lfr, score
    n, mu, seed = 10_000, 0.5, 0
    edges, gt = lfr(n, mu, seed)
    for name, fn in ALGORITHMS.items():
        lab, dt = fn(n, edges, seed)
        print(f"{name} n={n} mu={mu} seed={seed} "
              f"{score(gt, lab, n, edges)} t={dt:.3f}s", flush=True)


if __name__ == "__main__":
    main()
