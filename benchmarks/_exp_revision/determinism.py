"""Byte-reproducibility across thread counts (m5, Q15).

Runs the same graph at several thread counts and hashes the returned partition
and the whole refined front. A single differing byte fails the check.

    python -m _exp_revision.determinism
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink  # noqa: E402

FIELDS = ["alg", "kind", "net", "n_cfg", "mu", "seed", "threads", "n", "m",
          "k", "time", "hash_selected", "hash_front", "front_size", "stamp"]
KEY = ["alg", "kind", "net", "n_cfg", "mu", "seed", "threads"]

THREADS = [1, 2, 4, 8, 16, 48]


def main():
    sink = Sink("determinism.csv", FIELDS, KEY)
    grid = json.loads(os.environ.get(
        "REV_DET_LFR", '[[1000,0.3],[10000,0.3],[10000,0.5],[50000,0.5]]'))
    nets = os.environ.get("REV_DET_NETS", "karate,email_eu,dblp").split(",")

    tasks = []
    for alg in ["SMOCC"]:
        for th in THREADS:
            for n_cfg, mu in grid:
                tasks.append({"alg": alg, "kind": "lfr", "net": "",
                              "n_cfg": n_cfg, "mu": mu, "seed": 0,
                              "threads": th})
            for net in nets:
                tasks.append({"alg": alg, "kind": "real", "net": net,
                              "n_cfg": "", "mu": "", "seed": 0, "threads": th})
    for t in [t for t in tasks if not sink.has(t)]:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "determinism_worker.py"),
             json.dumps(t)], capture_output=True, text=True, timeout=43200)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                sink.write(dict(t, **json.loads(line[len("RESULT "):])))
                break
        else:
            tail = (proc.stderr or "").strip().splitlines()
            print(f"ERROR {t}: {tail[-1] if tail else proc.returncode}",
                  flush=True)

    import csv
    rows = list(csv.DictReader(open(sink.path)))
    ok = True
    for keyf in {(r["kind"], r["net"], r["n_cfg"], r["mu"], r["seed"])
                 for r in rows}:
        grp = [r for r in rows
               if (r["kind"], r["net"], r["n_cfg"], r["mu"], r["seed"]) == keyf]
        hs = {r["hash_selected"] for r in grp}
        hf = {r["hash_front"] for r in grp}
        state = "IDENTICAL" if len(hs) == 1 and len(hf) == 1 else "DIVERGED"
        if state == "DIVERGED":
            ok = False
        print(f"{keyf}: {len(grp)} thread counts "
              f"{sorted(int(r['threads']) for r in grp)} -> {state}",
              flush=True)
    print(f"DETERMINISM_{'OK' if ok else 'FAILED'} -> "
          f"{os.path.join(OUT, 'determinism.csv')}", flush=True)


if __name__ == "__main__":
    main()
