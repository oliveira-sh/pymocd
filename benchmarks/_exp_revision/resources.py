"""Peak memory and wall-clock against graph size (Q13, and the runtime-vs-n
curve the review asks for).

Each run is a fresh process. ``rss_base`` is the peak resident set after the
graph is loaded and handed to the library; ``rss_peak`` is the peak over the
whole run, so ``rss_peak - rss_base`` isolates what the search itself holds.

    python -m _exp_revision.resources
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink  # noqa: E402

DIAG = ["front_from_micro", "front_from_macro", "front_from_guidance",
        "front_size", "front4_size", "front4_only", "decode_calls", "cmax",
        "t_micro", "t_macro", "t_exchange", "t_post"]
SUMMARY = ["sweeps", "fallback_rounds", "centres_init", "centres_off",
           "centres_pop", "centres_influence", "guidance_survived",
           "influence_survived"]
STATS = ["n", "mean", "med", "p05", "p95", "min", "max"]

FIELDS = (["alg", "kind", "net", "n_cfg", "mu", "seed", "threads", "n", "m",
           "k", "ami", "time", "rss_base_mb", "rss_peak_mb", "rss_search_mb",
           "bytes_per_node", "bytes_per_edge", "stamp"]
          + DIAG + [f"{p}_{s}" for p in SUMMARY for s in STATS])
KEY = ["alg", "kind", "net", "n_cfg", "mu", "seed", "threads"]


def main():
    sink = Sink("resources.csv", FIELDS, KEY)
    threads = int(os.environ.get("REV_THREADS", "48"))
    runs = int(os.environ.get("REV_RES_RUNS", "3"))
    algs = os.environ.get("REV_RES_ALGS", "SMOCC,HP-MOCD").split(",")
    grid = json.loads(os.environ.get(
        "REV_RES_LFR",
        '[[10000,0.3],[50000,0.3],[100000,0.3],[250000,0.3],[500000,0.3],'
        '[1000000,0.3],[10000,0.5],[50000,0.5],[100000,0.5],[250000,0.5],'
        '[500000,0.5],[1000000,0.5]]'))
    nets = [s for s in os.environ.get("REV_RES_NETS", "").split(",") if s]

    tasks = []
    for alg in algs:
        for n_cfg, mu in grid:
            for seed in range(runs):
                tasks.append({"alg": alg, "kind": "lfr", "net": "",
                              "n_cfg": n_cfg, "mu": mu, "seed": seed,
                              "threads": threads})
        for net in nets:
            tasks.append({"alg": alg, "kind": "real", "net": net,
                          "n_cfg": "", "mu": "", "seed": 0,
                          "threads": threads})
    tasks = [t for t in tasks if not sink.has(t)]
    print(f"pending={len(tasks)}", flush=True)

    for t in tasks:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "resource_worker.py"),
             json.dumps(t)], capture_output=True, text=True, timeout=43200)
        row = dict(t)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                row.update(json.loads(line[len("RESULT "):]))
                break
        else:
            tail = (proc.stderr or "").strip().splitlines()
            print(f"ERROR {t}: {tail[-1] if tail else proc.returncode}",
                  flush=True)
            continue
        sink.write(row)
        print(f"{t['alg']} {t.get('net') or t['n_cfg']} mu={t['mu']} "
              f"s={t['seed']}: {row['rss_peak_mb']} MB peak, "
              f"{row['time']} s", flush=True)
    print(f"RESOURCES_DONE -> {os.path.join(OUT, 'resources.csv')}", flush=True)


if __name__ == "__main__":
    main()
