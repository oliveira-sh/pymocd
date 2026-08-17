"""One-factor-at-a-time sensitivity over the parameters the paper fixes (m3).

Each factor moves alone from the default configuration, so the effect of every
constant the review calls out is separable: the transfer interval, the
population size, the generation budget, the centre-ceiling factor, and the two
variation rates.

    python -m _exp_revision.sensitivity
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.arms import DIAG_KEYS, FIELDS, KEY, SUMMARY  # noqa: E402
from _exp_revision.common import OUT, Sink  # noqa: E402

FACTORS = {
    "gap": [1, 2, 5, 10, 25, 100],
    "pop_size": [25, 50, 100, 200],
    "num_gens": [25, 50, 100, 200],
    "macro_cap": [0.5, 1.0, 4.0, 16.0],
    "micro_mut": [0.01, 0.1, 0.25, 0.5, 0.9],
    "cross_rate": [0.0, 0.3, 0.7, 1.0],
}
DEFAULTS = {"gap": 10, "pop_size": 100, "num_gens": 100, "macro_cap": 1.0,
            "micro_mut": 0.5, "cross_rate": 0.7}


def main():
    runs = int(os.environ.get("REV_SENS_RUNS", "10"))
    threads = int(os.environ.get("REV_THREADS", "8"))
    workers = int(os.environ.get("REV_WORKERS", "5"))
    grid = json.loads(os.environ.get(
        "REV_SENS_LFR", '[[50000,0.3],[50000,0.6],[100000,0.5]]'))
    sink = Sink("sensitivity.csv", FIELDS, KEY)

    tasks = []
    for factor, values in FACTORS.items():
        for v in values:
            arm = f"{factor}={v}"
            cfg = {factor: v}
            for n_cfg, mu in grid:
                for seed in range(runs):
                    row = {"arm": arm, "n_cfg": n_cfg, "mu": mu, "seed": seed}
                    if not sink.has(row):
                        tasks.append((arm, cfg, n_cfg, mu, seed))
    tasks.sort(key=lambda t: (t[2], t[0], t[3], t[4]))
    print(f"factors={list(FACTORS)} cells={len(grid)} runs={runs} "
          f"pending={len(tasks)}", flush=True)

    lock = __import__("threading").Lock()
    done = [0]
    t_start = time.time()

    def one(task):
        arm, cfg, n, mu, seed = task
        payload = {"cfg": cfg, "n": n, "mu": mu, "seed": seed,
                   "threads": threads}
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "arm_worker.py"),
             json.dumps(payload)], capture_output=True, text=True,
            timeout=43200)
        row = {"arm": arm, "family": "sensitivity", "n_cfg": n, "mu": mu,
               "seed": seed, "threads": threads,
               "cfg": json.dumps(cfg, sort_keys=True)}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                row.update(json.loads(line[len("RESULT "):]))
                break
        else:
            row["status"] = "error"
        with lock:
            sink.write(row)
            done[0] += 1
            if done[0] % 25 == 0:
                el = time.time() - t_start
                print(f"[{done[0]}/{len(tasks)}] {el/60:.1f} min, eta "
                      f"{(len(tasks)-done[0])*el/max(done[0],1)/60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            f.result()
    print(f"SENSITIVITY_DONE -> {os.path.join(OUT, 'sensitivity.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
