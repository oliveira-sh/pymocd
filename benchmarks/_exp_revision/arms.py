"""Arm sweeps over the synthetic grid.

One "arm" is a named SMOCC configuration. The runner scores every arm on every
(n, mu, seed) cell and records both the partition quality and the per-stage
diagnostics, so the ablation (M2), the macro-rate study (M1) and the
distribution questions (Q1-Q5, Q10) all come out of one campaign.

    python -m _exp_revision.arms rates    # macro mutation rate and operator
    python -m _exp_revision.arms ablation # component ablation
    python -m _exp_revision.arms objectives

Environment:
    REV_RUNS      seeds per cell (default 20)
    REV_THREADS   rayon threads per job (default 12)
    REV_WORKERS   concurrent jobs (default 4)
    REV_CELLS     "mu" | "size" | "all" (default all)
    REV_ARMS      comma-separated arm names to restrict to
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink, cells  # noqa: E402

RUNS = int(os.environ.get("REV_RUNS", "20"))
THREADS = int(os.environ.get("REV_THREADS", "12"))
WORKERS = int(os.environ.get("REV_WORKERS", "4"))

DIAG_KEYS = ["front_from_micro", "front_from_macro", "front_from_guidance",
             "front_size", "front_size_refined", "front4_size", "front4_only",
             "decode_calls", "cmax", "t_total", "t_micro", "t_macro",
             "t_exchange", "t_post"]
SUMMARY = ["sweeps", "centres_init", "centres_off", "centres_pop",
           "centres_influence", "guidance_injected", "guidance_survived",
           "influence_injected", "influence_survived"]
STATS = ["mean", "med", "p05", "p95", "min", "max"]

FIELDS = (["arm", "family", "n_cfg", "mu", "seed", "status", "n", "m", "k",
           "gt_k", "time", "nmi", "ami", "ari", "modularity", "threads",
           "cfg", "stamp"]
          + DIAG_KEYS
          + [f"{p}_{s}" for p in SUMMARY for s in STATS])
KEY = ["arm", "n_cfg", "mu", "seed"]

# ``mut_rate`` tokens resolved per graph: "1/n" and "1/sqrt" scale with size.
RATE_ARMS = {
    "shipped":      dict(mut_rate=0.5,      mac_mode=0),
    "flat-0.1":     dict(mut_rate=0.1,      mac_mode=0),
    "flat-0.01":    dict(mut_rate=0.01,     mac_mode=0),
    "flat-1/sqrt":  dict(mut_rate="1/sqrt", mac_mode=0),
    "flat-1/n":     dict(mut_rate="1/n",    mac_mode=0),
    "keep-0.3":     dict(mut_rate=0.3,      mac_mode=1),
    "keep-0.1":     dict(mut_rate=0.1,      mac_mode=1),
    "keep-0.02":    dict(mut_rate=0.02,     mac_mode=1),
}

ABLATION_ARMS = {
    "full":        dict(),
    "no-macro":    dict(abl=1),
    "no-guidance": dict(abl=2),
    "no-influence": dict(abl=4),
    "no-transfer": dict(abl=6),
    "no-wupdate":  dict(abl=8),
    "no-refine":   dict(refine=False),
}

OBJECTIVE_ARMS = {
    "het-160":  dict(obj_mode=160),
    "homo-mic": dict(obj_mode=6),
    "homo-mac": dict(obj_mode=0),
    "swapped":  dict(obj_mode=106),
}

FRONT_ARMS = {
    "front-2obj": dict(front_mode=0),
    "front-4obj": dict(front_mode=1),
}

SUITES = {"rates": RATE_ARMS, "ablation": ABLATION_ARMS,
          "objectives": OBJECTIVE_ARMS, "front": FRONT_ARMS}


def resolve(cfg, n):
    out = dict(cfg)
    r = out.get("mut_rate")
    if r == "1/n":
        out["mut_rate"] = 1.0 / n
    elif r == "1/sqrt":
        out["mut_rate"] = 1.0 / (n ** 0.5)
    return out


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    arms = SUITES[suite]
    only = {s.strip() for s in os.environ.get("REV_ARMS", "").split(",")
            if s.strip()}
    if only:
        arms = {k: v for k, v in arms.items() if k in only}

    which = os.environ.get("REV_CELLS", "all")
    if which == "mu":
        grid = cells(size_n=[], size_mu=[])
    elif which == "size":
        grid = cells(mu_n=[], mu=[])
    else:
        grid = cells()

    sink = Sink(f"arms_{suite}.csv", FIELDS, KEY)
    tasks = []
    for arm, cfg in arms.items():
        for n, mu in grid:
            for seed in range(RUNS):
                row = {"arm": arm, "n_cfg": n, "mu": mu, "seed": seed}
                if not sink.has(row):
                    tasks.append((arm, cfg, n, mu, seed))
    tasks.sort(key=lambda t: (t[2], t[0], t[3], t[4]))
    print(f"suite={suite} arms={list(arms)} cells={len(grid)} runs={RUNS} "
          f"pending={len(tasks)} workers={WORKERS} threads={THREADS}",
          flush=True)

    lock = __import__("threading").Lock()
    t_start = time.time()
    done = [0]

    def one(task):
        arm, cfg, n, mu, seed = task
        payload = {"cfg": resolve(cfg, n), "n": n, "mu": mu, "seed": seed,
                   "threads": THREADS}
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "arm_worker.py"),
             json.dumps(payload)],
            capture_output=True, text=True, timeout=43200)
        row = {"arm": arm, "family": suite, "n_cfg": n, "mu": mu, "seed": seed,
               "threads": THREADS, "cfg": json.dumps(cfg, sort_keys=True)}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                row.update(json.loads(line[len("RESULT "):]))
                break
        else:
            row["status"] = "error"
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            print(f"ERROR {arm} n={n} mu={mu} s={seed}: "
                  f"{tail[-1] if tail else proc.returncode}", flush=True)
        with lock:
            sink.write(row)
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(tasks):
                el = time.time() - t_start
                rate = done[0] / max(el, 1e-9)
                print(f"[{done[0]}/{len(tasks)}] {el/60:.1f} min elapsed, "
                      f"eta {(len(tasks)-done[0])/max(rate,1e-9)/60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for f in as_completed([pool.submit(one, t) for t in tasks]):
            f.result()
    print(f"ARMS_DONE {suite} -> {os.path.join(OUT, f'arms_{suite}.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
