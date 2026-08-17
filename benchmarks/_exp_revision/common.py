"""Shared plumbing for the revision experiments.

Every experiment writes one resumable CSV under ``benchmarks/results/revision``.
Rows carry the full configuration, so a run can always be traced back to its
seed and its arm.
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

OUT = os.path.join(BENCH, "results", "revision")

from _exp_synt_net.hardened import LFR_PARAMS  # noqa: E402

# The cells the revision experiments sweep. The mixing sweep is the campaign's
# own; the size sweep is trimmed at 2.5e5 because the arms are run many times.
ABL_MU_N = [50_000, 100_000]
ABL_MU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ABL_SIZE_N = [10_000, 250_000]
ABL_SIZE_MU = [0.3, 0.5]

# Small graphs where the dense diffusion kernel fits, for the fidelity study.
SMALL_N = [300, 600, 1_000, 2_000]
SMALL_MU = [0.3, 0.5]


def _env_list(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [float(x) if "." in x else int(x)
            for x in raw.split(",") if x.strip()]


def cells(mu_n=None, mu=None, size_n=None, size_mu=None):
    mu_n = _env_list("REV_MU_N", ABL_MU_N) if mu_n is None else mu_n
    mu = _env_list("REV_MU", ABL_MU) if mu is None else mu
    size_n = _env_list("REV_SIZE_N", ABL_SIZE_N) if size_n is None else size_n
    size_mu = _env_list("REV_SIZE_MU", ABL_SIZE_MU) if size_mu is None else size_mu
    c = {(n, m) for n in mu_n for m in mu}
    c |= {(n, m) for n in size_n for m in size_mu}
    return sorted(c)


def lfr(n, mu, seed):
    """Load (or generate) one cached LFR graph."""
    from _exp_synt_net.gen_graphs import ensure
    d = np.load(ensure(n, mu, seed, log=lambda *a, **k: None))
    return d["edges"], d["gt"].astype(np.int64)


class Shim:
    """The zero-copy graph view pymocd accepts."""

    def __init__(self, n, e):
        self._n, self._e = n, e

    def nodes(self):
        return range(self._n)

    def edges(self):
        e = self._e
        for i in range(0, len(e), 1_000_000):
            yield from map(tuple, e[i:i + 1_000_000].tolist())


def densify(part, n):
    """A pymocd partition dict to a dense label vector, isolated nodes split."""
    lab = np.full(n, -1, dtype=np.int64)
    for node, c in part.items():
        lab[node] = c
    loose = lab == -1
    lab[loose] = np.arange(n, dtype=np.int64)[loose] + n
    return lab


def score(gt, lab, n=None, edges=None):
    """The campaign's scoring: NMI/AMI/ARI against ground truth, plus Q."""
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         normalized_mutual_info_score)
    out = {}
    if gt is not None:
        out["nmi"] = round(float(normalized_mutual_info_score(gt, lab)), 6)
        out["ami"] = round(float(adjusted_mutual_info_score(gt, lab)), 6)
        out["ari"] = round(float(adjusted_rand_score(gt, lab)), 6)
    out["k"] = int(len(np.unique(lab)))
    if edges is not None and n is not None:
        import igraph as ig
        _, dense = np.unique(lab, return_inverse=True)
        out["modularity"] = round(
            float(ig.Graph(n=n, edges=edges).modularity(dense.tolist())), 6)
    return out


class Sink:
    """Append-only CSV with a resume key."""

    def __init__(self, name, fields, key_fields):
        os.makedirs(OUT, exist_ok=True)
        self.path = os.path.join(OUT, name)
        self.fields = fields
        self.key_fields = key_fields
        self.done = set()
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    self.done.add(self.key(r))
        new = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def key(self, row):
        return tuple(str(row.get(k, "")) for k in self.key_fields)

    def has(self, row):
        return self.key(row) in self.done

    def write(self, row):
        full = {k: "" for k in self.fields}
        full.update(row)
        full["stamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.w.writerow(full)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.done.add(self.key(full))


def summarise(v):
    """Compact summary of an integer sample, for CSV storage."""
    if len(v) == 0:
        return {"n": 0, "mean": "", "med": "", "p05": "", "p95": "",
                "min": "", "max": ""}
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4),
            "med": float(np.median(a)), "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()),
            "max": float(a.max())}


def flat(prefix, s):
    return {f"{prefix}_{k}": v for k, v in s.items()}
