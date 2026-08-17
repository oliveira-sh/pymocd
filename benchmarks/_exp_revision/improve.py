"""Two candidate improvements, measured against the shipped algorithm.

The campaign shows one failure mode: above a mixing of 0.4 SMOCC returns far
too many communities, keeps its homogeneity and loses its completeness. The
front oracle sits 0.015 AMI above the deployed pick, so the front does not hold
the answer and a better selector cannot supply it. Two changes build the
missing candidates:

    coarsen  the refinement also emits an agglomerative chain per front member,
             so the pool holds partitions coarser than anything the search
             produced
    seed     the initial micro population takes its first slots from a cheap
             near-linear detector, so the fine search starts from the right
             granularity instead of from random neighbour labels
    robust   the selector drops the degenerate front members before setting
             each objective's scale, and anchors that scale at the 5th and
             95th percentiles of what remains rather than at the extremes
    floor    the consensus update never leaves an edge below a floor, so a cut
             every elite agrees on does not become permanent

    python -m _exp_revision.improve
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink, _env_list  # noqa: E402

ABL_NO_COARSEN = 1 << 4

def arm(seed=None, sel=0, floor=0.0, coarsen=False):
    return dict(abl=0 if coarsen else ABL_NO_COARSEN, seed=seed, sel=sel,
                floor=floor)


ARMS = {
    "shipped":        arm(),
    "coarsen":        arm(coarsen=True),
    "seed-lpa":       arm(seed="lpa"),
    "seed-infomap":   arm(seed="infomap"),
    "robust":         arm(sel=1),
    "robust+coarsen": arm(sel=1, coarsen=True),
    "robust+seed":    arm(sel=1, seed="lpa"),
    "all":            arm(sel=1, seed="lpa", coarsen=True),
    # The consensus update has an absorbing state: an edge every elite cuts
    # decays to zero weight, after which the decoder cannot cross it and no
    # later elite can rejoin its endpoints. A floor keeps the merge reachable.
    "floor-0.01":     arm(floor=0.01),
    "floor-0.05":     arm(floor=0.05),
    "floor-0.10":     arm(floor=0.10),
    "floor-0.20":     arm(floor=0.20),
    "robust+floor":   arm(sel=1, floor=0.05),
}

FIELDS = ["arm", "n_cfg", "mu", "seed", "status", "n", "m", "k", "gt_k",
          "nmi", "ami", "ari", "hom", "cmp", "modularity", "front_size",
          "seeds_used", "time", "t_seed", "threads", "stamp"]
KEY = ["arm", "n_cfg", "mu", "seed"]

RUNS = int(os.environ.get("REV_IMP_RUNS", "10"))
WORKERS = int(os.environ.get("REV_IMP_WORKERS", "5"))
THREADS = int(os.environ.get("REV_IMP_THREADS", "8"))
N = _env_list("REV_IMP_N", [10_000, 50_000])
MU = _env_list("REV_IMP_MU", [0.3, 0.4, 0.5, 0.6])
NSEED = int(os.environ.get("REV_IMP_NSEED", "10"))


def make_seeds(kind, n, edges, seed, count):
    """`count` copies of a cheap partition, perturbed apart after the first.

    One seed would occupy one slot; a handful spreads the same structure over
    several slots so crossover has something to recombine.
    """
    import numpy as np
    if kind is None:
        return [], 0.0
    t0 = time.perf_counter()
    if kind == "infomap":
        from _exp_revision.extra_baselines import run_infomap
        base, _ = run_infomap(n, edges, seed)
    elif kind == "lpa":
        from networkx.algorithms.community import asyn_lpa_communities
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(map(tuple, edges.tolist()))
        base = np.empty(n, dtype=np.int64)
        for ci, comm in enumerate(asyn_lpa_communities(g, seed=seed)):
            for v in comm:
                base[v] = ci
    else:
        raise ValueError(kind)
    dt = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    out = [{int(v): int(c) for v, c in enumerate(base)}]
    for _ in range(count - 1):
        p = base.copy()
        # Perturb a twentieth of the vertices onto a random neighbour's label,
        # so the seeded slots are not identical individuals.
        hit = rng.random(n) < 0.05
        idx = np.flatnonzero(hit)
        if len(idx):
            pick = rng.integers(0, len(edges), size=len(idx))
            p[idx] = base[edges[pick, 1]]
        out.append({int(v): int(c) for v, c in enumerate(p)})
    return out, dt


def worker(task):
    import numpy as np
    import pymocd
    from _exp_revision.common import Shim, densify, lfr, score
    pymocd.max_cores(THREADS)
    cfg = ARMS[task["arm"]]
    edges, gt = lfr(task["n_cfg"], task["mu"], task["seed"])
    n = task["n_cfg"]
    seeds, t_seed = make_seeds(cfg["seed"], n, edges, task["seed"], NSEED)
    t0 = time.perf_counter()
    res = pymocd.smocc_probe(Shim(n, edges), abl=cfg["abl"], seeds=seeds,
                             select_mode=cfg["sel"], w_floor=cfg["floor"])
    dt = time.perf_counter() - t0
    lab = densify(res["front"][res["selected"]], n)
    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    hom, cmp_, _ = homogeneity_completeness_v_measure(gt, lab)
    out = {"status": "ok", "n": n, "m": int(len(edges)),
           "gt_k": int(len(np.unique(gt))), "time": round(dt, 4),
           "t_seed": round(t_seed, 4), "threads": THREADS,
           "front_size": res["diag"]["front_size"],
           "seeds_used": res["diag"]["seeds_used"],
           "hom": round(float(hom), 6), "cmp": round(float(cmp_), 6)}
    out.update(score(gt, lab, n=n, edges=edges))
    print("RESULT " + json.dumps(out), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(json.loads(sys.argv[2]))
    only = {a.strip() for a in os.environ.get("REV_IMP_ARMS", "").split(",")
            if a.strip()}
    arms = [a for a in ARMS if not only or a in only]
    sink = Sink("improve.csv", FIELDS, KEY)
    tasks = [{"arm": a, "n_cfg": n, "mu": mu, "seed": s}
             for a in arms for n in N for mu in MU for s in range(RUNS)]
    tasks = [t for t in tasks if not sink.has(t)]
    tasks.sort(key=lambda t: (t["n_cfg"], t["mu"], t["arm"], t["seed"]))
    print(f"arms={arms} pending={len(tasks)} workers={WORKERS}", flush=True)
    lock = __import__("threading").Lock()
    done = [0]
    t0 = time.time()

    def one(t):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker",
             json.dumps(t)], capture_output=True, text=True, timeout=43200)
        row = dict(t)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                row.update(json.loads(line[len("RESULT "):]))
                break
        else:
            row["status"] = "error"
            tail = (proc.stderr or "").strip().splitlines()
            print(f"ERROR {t}: {tail[-1] if tail else proc.returncode}",
                  flush=True)
        with lock:
            sink.write(row)
            done[0] += 1
            if done[0] % 10 == 0:
                el = time.time() - t0
                print(f"[{done[0]}/{len(tasks)}] {el/60:.1f} min, eta "
                      f"{(len(tasks)-done[0])*el/max(done[0],1)/60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            f.result()
    print(f"IMPROVE_DONE -> {os.path.join(OUT, 'improve.csv')}", flush=True)


if __name__ == "__main__":
    main()
