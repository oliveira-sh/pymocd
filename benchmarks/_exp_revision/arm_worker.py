"""One instrumented SMOCC run in its own process, so a crash or an OOM costs
one cell rather than the campaign."""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import Shim, densify, flat, lfr, score, summarise  # noqa: E402

SUMMARY = ["sweeps", "centres_init", "centres_off", "centres_pop",
           "centres_influence", "guidance_injected", "guidance_survived",
           "influence_injected", "influence_survived"]
DIAG_KEYS = ["front_from_micro", "front_from_macro", "front_from_guidance",
             "front_size", "front_size_refined", "front4_size", "front4_only",
             "decode_calls", "cmax", "t_total", "t_micro", "t_macro",
             "t_exchange", "t_post"]


def main():
    task = json.loads(sys.argv[1])
    import pymocd
    pymocd.max_cores(task["threads"])
    edges, gt = lfr(task["n"], task["mu"], task["seed"])
    n = task["n"]
    shim = Shim(n, edges)

    t0 = time.perf_counter()
    res = pymocd.smocc_probe(shim, **task["cfg"])
    dt = time.perf_counter() - t0

    front = res["front"]
    lab = densify(front[res["selected"]], n)
    out = {"status": "ok", "n": n, "m": int(len(edges)),
           "time": round(dt, 4), "gt_k": int(len(np.unique(gt)))}
    out.update(score(gt, lab, n=n, edges=edges))
    d = res["diag"]
    for k in DIAG_KEYS:
        out[k] = d[k]
    for p in SUMMARY:
        out.update(flat(p, summarise(d[p])))
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
