import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SMOKE = bool(os.environ.get("HARD_SMOKE"))
RUNS = 2 if SMOKE else int(os.environ.get("HARD_RUNS", "20"))
TIMEOUT = int(os.environ.get("HARD_TIMEOUT", "60" if SMOKE else "43200"))
WORKERS = int(os.environ.get("HARD_WORKERS", "2" if SMOKE else "48"))
BIG_N = int(os.environ.get("HARD_BIG_N", "250000"))
BIG_WORKERS = int(os.environ.get("HARD_BIG_WORKERS", "1" if SMOKE else "4"))
ALL_THREADS = int(os.environ.get("HARD_ALL_THREADS", "2" if SMOKE else "48"))

# Rayon-parallel detectors: run one at a time with the whole machine, BEFORE
# the single-threaded algorithms (which fan out one worker per core). Tuple
# order = run order.
PARALLEL_ALGS = ("SMOCC", "SMOCC-II", "HP-MOCD")

LFR_DIR = os.path.join(BENCH, "data", "lfr")
OUT = os.path.join(BENCH, "results", "hardened")
RESULTS_CSV = os.path.join(OUT, "results.csv")

LFR_PARAMS = dict(tau1=2.5, tau2=1.5, average_degree=20, max_degree=50,
                  min_community=20, max_community=100)

if SMOKE:
    MU_SWEEP_N = [300]
    MU_SWEEP_MU = [0.3, 0.5]
    NODES_SWEEP_N = [300, 600]
    NODES_SWEEP_MU = [0.3]
else:
    MU_SWEEP_N = [50_000, 100_000]
    MU_SWEEP_MU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    NODES_SWEEP_N = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    NODES_SWEEP_MU = [0.3, 0.5]

ALGORITHMS = {
    "SMOCC":       dict(deterministic=True,  max_nodes=None,      needs="shim"),
    "SMOCC-II":    dict(deterministic=True,  max_nodes=None,      needs="shim"),
    "HP-MOCD":     dict(deterministic=False, max_nodes=None,      needs="shim"),
    "MMCoMO":      dict(deterministic=False, max_nodes=None,      needs="shim",
                        real_max_nodes=2_000),
    "NSGA-III CCM": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "NSGA-III KRM": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "Shi-MOCD (Q)": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "Shi-MOCD (D)": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "MOGA-Net":    dict(deterministic=False, max_nodes=None,      needs="shim"),
    "Louvain":     dict(deterministic=False, max_nodes=2_000_000, needs="nx"),
    "Leiden":      dict(deterministic=False, max_nodes=None,      needs="ig"),
    # gamma = graph density; CPM's resolution-free counterpoint to modularity
    "Leiden-CPM":  dict(deterministic=False, max_nodes=None,      needs="ig"),
    "ASYN-LPA":    dict(deterministic=False, max_nodes=2_000_000, needs="nx"),
}

CSV_FIELDS = ["alg", "kind", "net", "n_cfg", "mu", "seed", "status", "n", "m",
              "k", "time", "nmi", "ami", "ari", "hom", "cmp", "vm", "gt_k",
              "mu_real", "modularity", "part", "threads", "stamp"]


def row_key(r):
    def norm(v):
        s = str(v)
        if s in ("", "nan"):
            return ""
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return s
    return (r["alg"], r["kind"], norm(r["net"]), norm(r["n_cfg"]),
            norm(r["mu"]), norm(r["seed"]))


def task_key(t):
    return row_key({"alg": t["alg"], "kind": t["kind"],
                    "net": t.get("net", ""), "n_cfg": t.get("n_cfg", ""),
                    "mu": t.get("mu", ""), "seed": t["seed"]})


def lfr_tasks():
    tasks = []
    cells = [(n, mu) for n in MU_SWEEP_N for mu in MU_SWEEP_MU]
    cells += [(n, mu) for n in NODES_SWEEP_N for mu in NODES_SWEEP_MU]
    for n, mu in sorted(set(cells)):
        for alg, info in ALGORITHMS.items():
            if info["max_nodes"] is not None and n > info["max_nodes"]:
                continue
            for seed in range(RUNS):
                tasks.append({"alg": alg, "kind": "lfr", "n_cfg": n, "mu": mu,
                              "seed": seed, "_n": n,
                              "_family": ("lfr", alg, mu)})
    return tasks


def load_done():
    done, had_error = set(), set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                (done if r["status"] in ("ok", "timeout", "skipped")
                 else had_error).add(row_key(r))
    return done, had_error


class Sink:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        new = not os.path.exists(RESULTS_CSV)
        self.f = open(RESULTS_CSV, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS)
        if new:
            self.w.writeheader()
            self._flush()

    def _flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def write(self, task, status, threads, extra=None):
        row = {k: "" for k in CSV_FIELDS}
        row.update({"alg": task["alg"], "kind": task["kind"],
                    "net": task.get("net", ""),
                    "n_cfg": task.get("n_cfg", ""),
                    "mu": task.get("mu", ""), "seed": task["seed"],
                    "status": status, "threads": threads,
                    "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        for k, v in (extra or {}).items():
            if k in row:
                row[k] = v
        self.w.writerow(row)
        self._flush()


def run_task(task, threads):
    payload = {k: v for k, v in task.items() if not k.startswith("_")}
    payload["threads"] = threads
    cmd = [sys.executable,
           os.path.join(BENCH, "_exp_synt_net", "hardened_worker.py"),
           json.dumps(payload)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "timeout", {"time": round(time.time() - t0, 1)}, ""
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return "ok", json.loads(line[len("RESULT "):]), ""
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return "error", {"time": round(time.time() - t0, 1)}, \
        detail[-1] if detail else f"exit {proc.returncode}"


def run_campaign(tasks):
    done, had_error = load_done()
    pending = [t for t in tasks if task_key(t) not in done]
    retries = sum(1 for t in pending if task_key(t) in had_error)
    print(f"tasks total={len(tasks)} done={len(tasks) - len(pending)} "
          f"pending={len(pending)} (of which retries={retries}) "
          f"timeout={TIMEOUT}s", flush=True)

    sink = Sink()
    pruned = {}

    def dispatch(task, threads):
        fam, size = task["_family"], task["_n"]
        if fam in pruned and size > pruned[fam]:
            sink.write(task, "skipped", threads)
            print(f"SKIP  {task_key(task)} (timeout at {pruned[fam]})",
                  flush=True)
            return
        status, extra, err = run_task(task, threads)
        if status == "timeout":
            pruned[fam] = min(pruned.get(fam, size), size)
        sink.write(task, status, threads, extra)
        msg = f"{status.upper():7s} {task_key(task)} " \
              f"t={extra.get('time', '?')}s"
        if status == "ok" and extra.get("ami") != "":
            msg += f" ami={extra['ami']}"
        if err:
            msg += f" [{err[:200]}]"
        print(msg, flush=True)

    exclusive = [t for t in pending if t["alg"] in PARALLEL_ALGS]
    exclusive.sort(key=lambda t: (PARALLEL_ALGS.index(t["alg"]), t["_n"],
                                  task_key(t)))
    serial = [t for t in pending if t["alg"] not in PARALLEL_ALGS]
    serial.sort(key=lambda t: (t["_n"], task_key(t)))

    def run_pool(batch, workers):
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = set()
            for t in batch:
                futs.add(pool.submit(dispatch, t, 1))
                while len(futs) >= workers * 2:
                    done_f, futs = wait(futs, return_when=FIRST_COMPLETED)
                    for f in done_f:
                        f.result()
            for f in futs:
                f.result()

    for t in exclusive:
        dispatch(t, ALL_THREADS)

    run_pool([t for t in serial if t["_n"] < BIG_N], WORKERS)
    # ponytail: few concurrent multi-GB graph loads; raise HARD_BIG_WORKERS if RAM allows
    run_pool([t for t in serial if t["_n"] >= BIG_N], BIG_WORKERS)


if __name__ == "__main__":
    run_campaign(lfr_tasks())
    print("HARDENED_SYNT_DONE", flush=True)
