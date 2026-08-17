"""One profiled MMCoMO or SMOCC run."""

import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import Shim, densify, lfr  # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    t = json.loads(sys.argv[1])
    import pymocd
    pymocd.max_cores(t["threads"])
    edges, gt = lfr(t["n_cfg"], t["mu"], t["seed"])
    n = t["n_cfg"]
    shim = Shim(n, edges)
    base = peak_mb()

    t_kernel = ""
    if t["alg"] == "MMCoMO":
        # The kernel is the first thing mmcomo builds; time it on its own by
        # asking SMOCC's probe for the same dense object.
        t0 = time.perf_counter()
        pymocd.smocc_decode_media(shim, [[0] * n], 0.05)
        t_kernel = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        part = pymocd.mmcomo(shim)
        t_total = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        res = pymocd.smocc_probe(shim)
        part = res["front"][res["selected"]]
        t_total = time.perf_counter() - t0
    peak = peak_mb()

    lab = densify(part, n)
    from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                         adjusted_rand_score,
                                         normalized_mutual_info_score)
    out = {"n": int(n), "m": int(len(edges)), "k": int(len(np.unique(lab))),
           "gt_k": int(len(np.unique(gt))), "t_total": round(t_total, 4),
           "t_kernel": t_kernel,
           "t_search": round(t_total - (t_kernel or 0.0), 4),
           "rss_base_mb": round(base, 1), "rss_peak_mb": round(peak, 1),
           "rss_search_mb": round(peak - base, 1),
           "bytes_per_pair": round((peak - base) * 1024 * 1024 / (n * n), 3),
           "ami": round(float(adjusted_mutual_info_score(gt, lab)), 6),
           "nmi": round(float(normalized_mutual_info_score(gt, lab)), 6),
           "ari": round(float(adjusted_rand_score(gt, lab)), 6)}
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
