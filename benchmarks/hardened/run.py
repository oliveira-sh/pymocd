"""Resumable orchestrator: one subprocess per run with a hard timeout, every
row fsynced on completion; rows with status error are re-run on restart,
ok/timeout/skipped are final."""

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from config import (ALGORITHMS, CSV_FIELDS, EXCLUSIVE_N, EXCLUSIVE_THREADS,
                    MU_SWEEP_MU, MU_SWEEP_N, NODES_SWEEP_MU, NODES_SWEEP_N,
                    OUT, REAL_SMALL, REAL_SNAP, RESULTS_CSV, RUNS, THREADS,
                    TIMEOUT, WORKERS, norm_task_key, row_key)
from realnets import LOADERS  # noqa: F401

REAL_SIZES = {"karate": 34, "dolphins": 62, "lesmis": 77, "florentine": 15,
              "polbooks": 105, "football": 115, "email_eu": 1005,
              "dblp": 317_080, "amazon": 334_863, "youtube": 1_134_890,
              "lj": 3_997_962, "orkut": 3_072_441}


def build_tasks():
    tasks, seen = [], set()

    def add(t):
        info = ALGORITHMS[t["alg"]]
        cap = info["max_nodes"]
        if cap is not None and t["_n"] > cap:
            return
        k = norm_task_key(t)
        if k in seen:
            return
        seen.add(k)
        tasks.append(t)

    for net in REAL_SMALL + REAL_SNAP:
        n = REAL_SIZES[net]
        for alg, info in ALGORITHMS.items():
            runs = 1 if info["deterministic"] else RUNS
            for seed in range(runs):
                add({"alg": alg, "kind": "real", "net": net, "seed": seed,
                     "_n": n, "_family": ("real", alg), "_size": n})
    lfr_cells = [(n, mu) for n in MU_SWEEP_N for mu in MU_SWEEP_MU]
    lfr_cells += [(n, mu) for n in NODES_SWEEP_N for mu in NODES_SWEEP_MU]
    for n, mu in sorted(set(lfr_cells)):
        for alg in ALGORITHMS:
            for seed in range(RUNS):
                add({"alg": alg, "kind": "lfr", "n_cfg": n, "mu": mu,
                     "seed": seed, "_n": n, "_family": ("lfr", alg, mu),
                     "_size": n})
    tasks.sort(key=lambda t: (t["_n"], t["kind"], t["alg"], t["seed"]))
    return tasks


def load_done():
    done, had_error = set(), set()
    if not os.path.exists(RESULTS_CSV):
        return done, had_error
    with open(RESULTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            k = row_key(r)
            if r["status"] in ("ok", "timeout", "skipped"):
                done.add(k)
            else:
                had_error.add(k)
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
    cmd = [sys.executable, os.path.join(HERE, "worker.py"),
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


def main():
    tasks = build_tasks()
    done, had_error = load_done()
    pending = [t for t in tasks if norm_task_key(t) not in done]
    retries = sum(1 for t in pending if norm_task_key(t) in had_error)
    print(f"tasks total={len(tasks)} done={len(tasks) - len(pending)} "
          f"pending={len(pending)} (of which retries={retries}) "
          f"timeout={TIMEOUT}s", flush=True)

    sink = Sink()
    pruned = {}

    def dispatch(task, threads):
        fam, size = task["_family"], task["_size"]
        if fam in pruned and size > pruned[fam]:
            sink.write(task, "skipped", threads)
            print(f"SKIP  {norm_task_key(task)} (timeout at "
                  f"{pruned[fam]})", flush=True)
            return
        status, extra, err = run_task(task, threads)
        if status == "timeout":
            pruned[fam] = min(pruned.get(fam, size), size)
        sink.write(task, status, threads, extra)
        msg = f"{status.upper():7s} {norm_task_key(task)} " \
              f"t={extra.get('time', '?')}s"
        if status == "ok" and extra.get("ami") != "":
            msg += f" ami={extra['ami']}"
        if err:
            msg += f" [{err[:200]}]"
        print(msg, flush=True)

    parallel = [t for t in pending if t["_n"] < EXCLUSIVE_N]
    exclusive = [t for t in pending if t["_n"] >= EXCLUSIVE_N]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = set()
        for t in parallel:
            futs.add(pool.submit(dispatch, t, THREADS))
            while len(futs) >= WORKERS * 2:
                done_f, futs = wait(futs, return_when=FIRST_COMPLETED)
                for f in done_f:
                    f.result()
        for f in futs:
            f.result()

    for t in exclusive:
        dispatch(t, EXCLUSIVE_THREADS)

    print("HARDENED_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
