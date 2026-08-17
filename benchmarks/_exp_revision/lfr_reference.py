"""Correct LFR generator for the revision (M10).

The campaign used ``networkx.LFR_benchmark_graph``, which does not deliver the
requested mixing (appendix "Nominal Versus Realized Mixing"). This module
drives the reference Lancichinetti-Fortunato-Radicchi ``binary_networks``
program instead, vendored under ``_exp_revision/vendor/LFR-benchmark``, and
returns ``(edges, gt)`` in the npz layout of ``_exp_synt_net/gen_graphs.py``.

The reference program takes its seed from a ``time_seed.dat`` in the working
directory, so every call runs in a private temporary directory holding the
seed this module derived from the campaign seed. The seed actually used is
stored in the npz as ``seed_value``.

    python -m _exp_revision.lfr_reference --n 1000 --mu 0.3 --seed 0
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from _exp_revision.common import LFR_PARAMS  # noqa: E402

SRC = os.path.join(HERE, "vendor", "LFR-benchmark", "src", "benchm.cpp")
BIN = os.path.join(HERE, "vendor", "LFR-benchmark", "lfr_binary")
REF_DIR = os.path.join(BENCH, "data", "lfr_ref")

# ran2 accepts seeds in [1, 2147483399]; the stride keeps the campaign seeds
# far apart in the stream and leaves room for the retry offset.
SEED_BASE = 1_000_003
SEED_STRIDE = 7_919

FLAGS = {"tau1": "-t1", "tau2": "-t2", "average_degree": "-k",
         "max_degree": "-maxk", "min_community": "-minc",
         "max_community": "-maxc"}


def build(force=False):
    """Compile the reference program if the binary is missing."""
    if os.path.exists(BIN) and not force:
        return BIN
    cxx = os.environ.get("CXX", "g++")
    if shutil.which(cxx) is None:
        raise RuntimeError(f"{cxx} not found; cannot build the LFR reference")
    subprocess.run([cxx, "-O3", "-funroll-loops", "-w", "-o", BIN, SRC],
                   check=True)
    return BIN


def seed_value(seed, attempt=0):
    """The ran2 seed the reference program is handed for a campaign seed."""
    return SEED_BASE + SEED_STRIDE * int(seed) + attempt


def _read(work, n):
    edges = np.loadtxt(os.path.join(work, "network.dat"), dtype=np.int64)
    edges = edges.reshape(-1, 2) - 1
    edges = edges[edges[:, 0] < edges[:, 1]]
    edges = np.ascontiguousarray(edges.astype(np.uint32))
    comm = np.loadtxt(os.path.join(work, "community.dat"), dtype=np.int64)
    comm = comm.reshape(-1, 2)
    gt = np.empty(n, dtype=np.int32)
    gt[comm[:, 0] - 1] = comm[:, 1]
    return edges, gt


def generate(n, mu, seed, params=None):
    """One reference LFR graph. Returns (edges, gt, seed_value)."""
    params = LFR_PARAMS if params is None else params
    exe = build()
    argv = [exe, "-N", str(int(n)), "-mu", f"{mu:g}"]
    for name, flag in FLAGS.items():
        argv += [flag, f"{params[name]:g}"]
    for attempt in range(5):
        s = seed_value(seed, attempt)
        work = tempfile.mkdtemp(prefix="lfrref-")
        try:
            with open(os.path.join(work, "time_seed.dat"), "w") as f:
                f.write(f"{s}\n")
            p = subprocess.run(argv, cwd=work, capture_output=True, text=True)
            ok = (p.returncode == 0
                  and os.path.exists(os.path.join(work, "network.dat")))
            if ok:
                edges, gt = _read(work, int(n))
                return edges, gt, s
            if attempt == 4:
                raise RuntimeError(
                    f"LFR reference failed for n={n} mu={mu} seed={seed}: "
                    f"{p.stdout[-500:]}{p.stderr[-500:]}")
        finally:
            shutil.rmtree(work, ignore_errors=True)


def path_for(n, mu, seed):
    return os.path.join(REF_DIR, f"n{n}_mu{mu:g}_s{seed}.npz")


def ensure(n, mu, seed, log=print):
    """Cache one reference graph under benchmarks/data/lfr_ref."""
    p = path_for(n, mu, seed)
    if os.path.exists(p):
        return p
    os.makedirs(REF_DIR, exist_ok=True)
    t0 = time.time()
    edges, gt, s = generate(n, mu, seed)
    tmp = p + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, edges=edges, gt=gt,
                        seed_value=np.int64(s))
    os.replace(tmp, p)
    log(f"generated n={n} mu={mu} seed={seed} seed_value={s} m={len(edges)} "
        f"in {time.time() - t0:.1f}s", flush=True)
    return p


def load(n, mu, seed):
    """(edges, gt) for one cached reference graph, generating it if needed."""
    d = np.load(ensure(n, mu, seed, log=lambda *a, **k: None))
    return d["edges"], d["gt"].astype(np.int64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    edges, gt, s = generate(a.n, a.mu, a.seed)
    inter = float((gt[edges[:, 0]] != gt[edges[:, 1]]).mean())
    deg = np.bincount(edges.reshape(-1).astype(np.int64), minlength=a.n)
    print(f"n={a.n} mu_nominal={a.mu} seed={a.seed} seed_value={s} "
          f"m={len(edges)} mu_real={inter:.4f} deg_mean={deg.mean():.3f} "
          f"deg_max={deg.max()} k={len(np.unique(gt))} "
          f"in {time.time() - t0:.1f}s")
