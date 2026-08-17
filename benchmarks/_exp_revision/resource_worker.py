"""One measured run: peak resident set before and after the search."""

import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import Shim, densify, flat, lfr, summarise  # noqa: E402

DIAG = ["front_from_micro", "front_from_macro", "front_from_guidance",
        "front_size", "front4_size", "front4_only", "decode_calls", "cmax",
        "t_micro", "t_macro", "t_exchange", "t_post"]
SUMMARY = ["sweeps", "fallback_rounds", "centres_init", "centres_off",
           "centres_pop", "centres_influence", "guidance_survived",
           "influence_survived"]


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    t = json.loads(sys.argv[1])
    import pymocd
    pymocd.max_cores(t["threads"])

    gt = None
    if t["kind"] == "lfr":
        edges, gt = lfr(t["n_cfg"], t["mu"], t["seed"])
        n = t["n_cfg"]
    else:
        from _exp_real_net.networks import LOADERS
        edges, n, gt, eval_nodes = LOADERS[t["net"]]()
    shim = Shim(n, edges)
    base = peak_mb()

    t0 = time.perf_counter()
    diag = {}
    if t["alg"] == "SMOCC":
        res = pymocd.smocc_probe(shim)
        part = res["front"][res["selected"]]
        front_size = len(res["front"])
        d = res["diag"]
        diag = {k: d[k] for k in DIAG}
        for pre in SUMMARY:
            diag.update(flat(pre, summarise(d[pre])))
    elif t["alg"] == "HP-MOCD":
        part = pymocd.hpmocd(shim)
        front_size = ""
    else:
        part = getattr(pymocd, t["alg"])(shim)
        front_size = ""
    dt = time.perf_counter() - t0
    peak = peak_mb()

    lab = densify(part, n)
    out = {"n": int(n), "m": int(len(edges)), "k": int(len(np.unique(lab))),
           "time": round(dt, 4), "rss_base_mb": round(base, 1),
           "rss_peak_mb": round(peak, 1),
           "rss_search_mb": round(peak - base, 1),
           "bytes_per_node": round((peak - base) * 1024 * 1024 / n, 1),
           "bytes_per_edge": round((peak - base) * 1024 * 1024 / len(edges), 1),
           "front_size": front_size}
    out.update(diag)
    if gt is not None:
        from sklearn.metrics.cluster import adjusted_mutual_info_score
        pred = lab if t["kind"] == "lfr" else lab[eval_nodes]
        out["ami"] = round(float(adjusted_mutual_info_score(gt, pred)), 6)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
