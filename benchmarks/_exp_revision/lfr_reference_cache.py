"""Build the corrected LFR cache for the revision (M10).

Populates ``benchmarks/data/lfr_ref`` with reference-generator graphs over
n x mu x seed. The existing ``data/lfr`` cache (networkx) is never touched.
Resumable: a cell already on disk is skipped. Reports per-size timings and the
projected cost of the full grid.

    python -m _exp_revision.lfr_reference_cache
    REF_N=1000,10000 REF_WORKERS=8 python -m _exp_revision.lfr_reference_cache
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision import lfr_reference  # noqa: E402
from _exp_revision.common import _env_list  # noqa: E402

REF_N = _env_list("REF_N", [1_000, 10_000, 50_000, 100_000])
REF_MU = _env_list("REF_MU", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
REF_SEEDS = int(os.environ.get("REF_SEEDS", "10"))
WORKERS = int(os.environ.get("REF_WORKERS", "8"))


def build_one(task):
    n, mu, seed = task
    t0 = time.perf_counter()
    fresh = not os.path.exists(lfr_reference.path_for(n, mu, seed))
    lfr_reference.ensure(n, mu, seed, log=lambda *a, **k: None)
    return n, mu, seed, time.perf_counter() - t0, fresh


def main():
    lfr_reference.build()
    os.makedirs(lfr_reference.REF_DIR, exist_ok=True)
    tasks = [(n, mu, s) for n in REF_N for mu in REF_MU
             for s in range(REF_SEEDS)]
    print(f"grid: n={REF_N} mu={REF_MU} seeds=0..{REF_SEEDS - 1} "
          f"({len(tasks)} cells) workers={WORKERS}", flush=True)
    times, done, t_wall = {}, 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(build_one, t): t for t in tasks}
        for f in as_completed(futs):
            n, mu, seed, dt, fresh = f.result()
            done += 1
            if fresh:
                times.setdefault(n, []).append(dt)
            print(f"[{done}/{len(tasks)}] n={n} mu={mu} seed={seed} "
                  f"{'built' if fresh else 'cached'} in {dt:.1f}s", flush=True)
    wall = time.perf_counter() - t_wall
    print(f"\nwall {wall:.1f}s over {WORKERS} workers", flush=True)
    print(f"{'n':>8} {'built':>6} {'mean_s':>8} {'max_s':>8} "
          f"{'proj_cpu_s':>11}")
    total = 0.0
    for n in REF_N:
        v = times.get(n)
        if not v:
            print(f"{n:>8} {0:>6} {'-':>8} {'-':>8} {'-':>11}")
            continue
        mean = sum(v) / len(v)
        cell = mean * len(REF_MU) * REF_SEEDS
        total += cell
        print(f"{n:>8} {len(v):>6} {mean:>8.2f} {max(v):>8.2f} {cell:>11.0f}")
    print(f"projected full grid: {total:.0f} CPU-s "
          f"({total / 3600:.2f} CPU-h), {total / WORKERS / 3600:.2f} h "
          f"at {WORKERS} workers")
    print("LFR_REF_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
