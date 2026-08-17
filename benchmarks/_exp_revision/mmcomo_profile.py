"""Where MMCoMO's cost actually goes (M5, Q8, Q9).

Times the dense kernel construction separately from the co-evolutionary search
and records peak resident set, so the paper can attribute the observed scaling
to the right term instead of to an unqualified "quadratic barrier". Both
detectors run in the same Rust library on the same machine, which removes the
language and parallelism confound of the published comparison.

    python -m _exp_revision.mmcomo_profile
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Sink  # noqa: E402

FIELDS = ["alg", "n_cfg", "mu", "seed", "threads", "n", "m", "k", "gt_k",
          "ami", "nmi", "ari", "t_total", "t_kernel", "t_search",
          "rss_base_mb", "rss_peak_mb", "rss_search_mb", "bytes_per_pair",
          "stamp"]
KEY = ["alg", "n_cfg", "mu", "seed", "threads"]


def main():
    sink = Sink("mmcomo_profile.csv", FIELDS, KEY)
    threads = int(os.environ.get("REV_THREADS", "8"))
    grid = json.loads(os.environ.get(
        "REV_MM_LFR",
        '[[300,0.3],[600,0.3],[1000,0.3],[2000,0.3],[4000,0.3],[8000,0.3]]'))
    runs = int(os.environ.get("REV_MM_RUNS", "3"))
    algs = os.environ.get("REV_MM_ALGS", "MMCoMO,SMOCC").split(",")

    for alg in algs:
        for n_cfg, mu in grid:
            for seed in range(runs):
                t = {"alg": alg, "n_cfg": n_cfg, "mu": mu, "seed": seed,
                     "threads": threads}
                if sink.has(t):
                    continue
                proc = subprocess.run(
                    [sys.executable,
                     os.path.join(HERE, "mmcomo_worker.py"), json.dumps(t)],
                    capture_output=True, text=True, timeout=43200)
                for line in proc.stdout.splitlines():
                    if line.startswith("RESULT "):
                        row = dict(t, **json.loads(line[len("RESULT "):]))
                        sink.write(row)
                        print(f"{alg} n={n_cfg} s={seed}: total="
                              f"{row['t_total']}s kernel={row['t_kernel']}s "
                              f"peak={row['rss_peak_mb']}MB", flush=True)
                        break
                else:
                    tail = (proc.stderr or "").strip().splitlines()
                    print(f"ERROR {t}: {tail[-1] if tail else proc.returncode}",
                          flush=True)
    print(f"MMCOMO_PROFILE_DONE -> {os.path.join(OUT, 'mmcomo_profile.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
