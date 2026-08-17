"""A second LFR regime: few large communities (M9f).

The campaign's only synthetic family bounds community sizes in [20, 100], so
the number of planted communities grows with n and every modularity-based
detector meets its resolution limit. That is the regime the resolution-limit
argument predicts, which makes it the wrong place to stop. This module
generates the opposite family with the reference generator, communities of
1,000 to 5,000 vertices so their count stays small as n grows, and scores the
detectors that run at this scale on it.

    python -m _exp_revision.regime

Environment:
    REV_REG_N        graph sizes (default 50000,100000)
    REV_REG_MU       mixing values (default 0.1..0.7)
    REV_REG_RUNS     seeds per cell (default 10)
    REV_REG_ALGS     detectors (default the six that run at this scale)
    REV_REG_WORKERS  concurrent jobs
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from _exp_revision.common import OUT, Sink, _env_list  # noqa: E402

# Few large communities: the count stays near n/2500 instead of growing with n.
REGIME = dict(tau1=2.5, tau2=1.5, average_degree=20, max_degree=50,
              min_community=1000, max_community=5000)

N = _env_list("REV_REG_N", [50_000, 100_000])
MU = _env_list("REV_REG_MU", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
RUNS = int(os.environ.get("REV_REG_RUNS", "10"))
WORKERS = int(os.environ.get("REV_REG_WORKERS", "8"))
ALGS = [a for a in os.environ.get(
    "REV_REG_ALGS",
    "SMOCC,HP-MOCD,Louvain,Leiden,Leiden-CPM,Infomap,ASYN-LPA").split(",") if a]

FIELDS = ["alg", "kind", "net", "n_cfg", "mu", "seed", "status", "n", "m", "k",
          "time", "nmi", "ami", "ari", "gt_k", "mu_real", "modularity",
          "threads", "stamp"]
KEY = ["alg", "kind", "net", "n_cfg", "mu", "seed"]
CACHE = os.path.join(BENCH, "data", "lfr_large")


def path_for(n, mu, seed):
    return os.path.join(CACHE, f"n{n}_mu{mu:g}_s{seed}.npz")


def ensure(n, mu, seed):
    import numpy as np
    from _exp_revision import lfr_reference
    p = path_for(n, mu, seed)
    if os.path.exists(p):
        return p
    os.makedirs(CACHE, exist_ok=True)
    edges, gt, s = lfr_reference.generate(n, mu, seed, params=REGIME)
    tmp = p + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, edges=edges, gt=gt, seed_value=np.int64(s))
    os.replace(tmp, p)
    return p


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(json.loads(sys.argv[2]))
    from _exp_revision import lfr_reference
    lfr_reference.build()
    sink = Sink("regime.csv", FIELDS, KEY)
    tasks = []
    for n in N:
        for mu in MU:
            for seed in range(RUNS):
                ensure(n, mu, seed)
                for alg in ALGS:
                    row = {"alg": alg, "kind": "lfr", "net": "", "n_cfg": n,
                           "mu": mu, "seed": seed}
                    if not sink.has(row):
                        tasks.append(row)
    print(f"cache ready; pending={len(tasks)} workers={WORKERS}", flush=True)
    lock = __import__("threading").Lock()
    done = [0]
    t0 = time.time()

    def one(row):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker",
             json.dumps(row)], capture_output=True, text=True, timeout=43200)
        out = dict(row)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                out.update(json.loads(line[len("RESULT "):]))
                break
        else:
            out["status"] = "error"
            tail = (proc.stderr or "").strip().splitlines()
            print(f"ERROR {row}: {tail[-1] if tail else proc.returncode}",
                  flush=True)
        with lock:
            sink.write(out)
            done[0] += 1
            if done[0] % 20 == 0:
                el = time.time() - t0
                print(f"[{done[0]}/{len(tasks)}] {el/60:.1f} min, eta "
                      f"{(len(tasks)-done[0])*el/max(done[0],1)/60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            f.result()
    print(f"REGIME_DONE -> {os.path.join(OUT, 'regime.csv')}", flush=True)


def worker(task):
    import numpy as np
    sys.path.insert(0, BENCH)
    from _exp_synt_net.hardened_worker import run_algorithm
    import pymocd
    threads = 6 if task["alg"] in ("SMOCC", "HP-MOCD") else 1
    pymocd.max_cores(threads)
    d = np.load(path_for(task["n_cfg"], task["mu"], task["seed"]))
    edges, gt = d["edges"], d["gt"].astype(np.int64)
    n = int(task["n_cfg"])
    if task["alg"] == "Infomap":
        from _exp_revision.extra_baselines import ALGORITHMS
        lab, dt = ALGORITHMS["Infomap"](n, edges, task["seed"])
    else:
        lab, dt = run_algorithm(task["alg"], n, edges, task["seed"], threads)
    import igraph as ig
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         normalized_mutual_info_score)
    _, dense = np.unique(lab, return_inverse=True)
    print("RESULT " + json.dumps({
        "status": "ok", "n": n, "m": int(len(edges)),
        "k": int(len(np.unique(lab))), "time": round(dt, 4),
        "gt_k": int(len(np.unique(gt))),
        "mu_real": round(float((gt[edges[:, 0]] != gt[edges[:, 1]]).mean()), 6),
        "nmi": round(float(normalized_mutual_info_score(gt, lab)), 6),
        "ami": round(float(adjusted_mutual_info_score(gt, lab)), 6),
        "ari": round(float(adjusted_rand_score(gt, lab)), 6),
        "modularity": round(float(ig.Graph(n=n, edges=edges)
                                  .modularity(dense.tolist())), 6),
        "threads": threads}), flush=True)


if __name__ == "__main__":
    main()
