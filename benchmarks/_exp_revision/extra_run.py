"""Score the review's missing baselines (Infomap, DC-SBM) on the campaign's
own graphs, with the campaign's metrics and columns (M9d).

Only LFR cells already present in the shared cache are run; missing ones are
skipped rather than generated, because another process owns that cache. The
seed indexes both the graph and the detector, as in the main campaign.

    python -m _exp_revision.extra_run

Environment:
    REV_EXTRA_ALGS   comma-separated baselines (default Infomap,DC-SBM)
    REV_EXTRA_NETS   comma-separated real networks (default the seven small)
    REV_EXTRA_RUNS   seeds per cell (default 20)
    REV_EXTRA_WORKERS  concurrent jobs (default 8)
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

from _exp_revision.common import OUT, Sink  # noqa: E402
from _exp_synt_net.gen_graphs import all_cells, path_for  # noqa: E402

RUNS = int(os.environ.get("REV_EXTRA_RUNS", "20"))
WORKERS = int(os.environ.get("REV_EXTRA_WORKERS", "8"))
ALGS = [s for s in os.environ.get("REV_EXTRA_ALGS", "Infomap,DC-SBM").split(",")
        if s]
NETS = [s for s in os.environ.get(
    "REV_EXTRA_NETS",
    "karate,dolphins,polbooks,football,lesmis,florentine,email_eu").split(",")
    if s]
TIMEOUT = int(os.environ.get("REV_EXTRA_TIMEOUT", "43200"))
# 0 restricts the run to the real networks, so a heavy LFR sweep and a cheap
# real-network sweep can be dispatched separately.
DO_LFR = os.environ.get("REV_EXTRA_LFR", "1") != "0"
# The DC-SBM agglomeration is pure Python and costs minutes at 1e5 nodes, so it
# is capped; Infomap is native and runs on every cached cell.
MAX_N = {"DC-SBM": int(os.environ.get("REV_DCSBM_MAX_N", "50000"))}

# The main campaign's header, verbatim, so the two CSVs concatenate.
FIELDS = ["alg", "kind", "net", "n_cfg", "mu", "seed", "status", "n", "m", "k",
          "time", "nmi", "ami", "ari", "hom", "cmp", "vm", "gt_k", "mu_real",
          "modularity", "part", "threads", "stamp"]
KEY = ["alg", "kind", "net", "n_cfg", "mu", "seed"]


def worker(task):
    import numpy as np
    from _exp_revision.extra_baselines import ALGORITHMS

    if task["kind"] == "lfr":
        n = task["n_cfg"]
        d = np.load(path_for(n, task["mu"], task["seed"]))
        edges, gt = d["edges"], d["gt"].astype(np.int64)
        eval_nodes = np.arange(n, dtype=np.int64)
    else:
        from _exp_real_net.networks import LOADERS
        edges, n, gt, eval_nodes = LOADERS[task["net"]]()

    lab, dt = ALGORITHMS[task["alg"]](n, edges, task["seed"])

    import igraph as ig
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         homogeneity_completeness_v_measure,
                                         normalized_mutual_info_score)
    _, lab_dense = np.unique(lab, return_inverse=True)
    mod = float(ig.Graph(n=n, edges=edges).modularity(lab_dense.tolist()))
    out = {"status": "ok", "n": int(n), "m": int(len(edges)),
           "k": int(len(np.unique(lab))), "time": round(dt, 4),
           "modularity": round(mod, 6), "nmi": "", "ami": "", "ari": "",
           "hom": "", "cmp": "", "vm": "", "gt_k": "", "mu_real": ""}
    if gt is not None:
        pred = lab[eval_nodes]
        out["nmi"] = round(float(normalized_mutual_info_score(gt, pred)), 6)
        out["ami"] = round(float(adjusted_mutual_info_score(gt, pred)), 6)
        out["ari"] = round(float(adjusted_rand_score(gt, pred)), 6)
        hom, cmp_, vm = homogeneity_completeness_v_measure(gt, pred)
        out["hom"], out["cmp"], out["vm"] = \
            round(float(hom), 6), round(float(cmp_), 6), round(float(vm), 6)
        out["gt_k"] = int(len(np.unique(gt)))
    if task["kind"] == "lfr":
        out["mu_real"] = round(
            float((gt[edges[:, 0]] != gt[edges[:, 1]]).mean()), 6)

    slug = re.sub(r"\W+", "-", "_".join(map(str, [
        task["alg"], task["kind"], task.get("net") or task.get("n_cfg"),
        task.get("mu", ""), task["seed"]])))
    parts = os.path.join(BENCH, "data", "parts")
    os.makedirs(parts, exist_ok=True)
    np.savez_compressed(os.path.join(parts, slug + ".npz"),
                        labels=lab.astype(np.int32))
    out["part"] = f"data/parts/{slug}.npz"
    print("RESULT " + json.dumps(out), flush=True)


def tasks(sink):
    out = []
    for alg in ALGS:
        for n, mu in (all_cells() if DO_LFR else []):
            for seed in range(RUNS):
                if n > MAX_N.get(alg, sys.maxsize):
                    continue
                if not os.path.exists(path_for(n, mu, seed)):
                    continue
                t = {"alg": alg, "kind": "lfr", "net": "", "n_cfg": n,
                     "mu": mu, "seed": seed}
                if not sink.has(t):
                    out.append(t)
        for net in NETS:
            for seed in range(RUNS):
                t = {"alg": alg, "kind": "real", "net": net, "n_cfg": "",
                     "mu": "", "seed": seed}
                if not sink.has(t):
                    out.append(t)
    out.sort(key=lambda t: (t["n_cfg"] or 0, t["alg"], t["net"], t["mu"] or 0,
                            t["seed"]))
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
        return

    sink = Sink("extra_baselines.csv", FIELDS, KEY)
    pending = tasks(sink)
    print(f"algs={ALGS} nets={NETS} runs={RUNS} workers={WORKERS} "
          f"pending={len(pending)}", flush=True)

    lock = __import__("threading").Lock()
    t_start = time.time()
    done = [0]

    def one(t):
        row = dict(t, threads=1)
        try:
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "extra_run.py"),
                 "--worker", json.dumps(t)],
                capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        else:
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT "):
                    row.update(json.loads(line[len("RESULT "):]))
                    break
            else:
                row["status"] = "error"
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                print(f"ERROR {t['alg']} {t.get('net') or t['n_cfg']} "
                      f"mu={t['mu']} s={t['seed']}: "
                      f"{tail[-1] if tail else proc.returncode}", flush=True)
        with lock:
            sink.write(row)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(pending):
                el = time.time() - t_start
                rate = done[0] / max(el, 1e-9)
                print(f"[{done[0]}/{len(pending)}] {el/60:.1f} min elapsed, "
                      f"eta {(len(pending)-done[0])/max(rate,1e-9)/60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(one, t) for t in pending]):
            f.result()
    print(f"EXTRA_DONE -> {os.path.join(OUT, 'extra_baselines.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
