"""One SMOCC run at a fixed thread count, hashed."""

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import Shim, densify, lfr  # noqa: E402


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int32)).hexdigest()[:16]


def main():
    t = json.loads(sys.argv[1])
    import pymocd
    pymocd.max_cores(t["threads"])
    if t["kind"] == "lfr":
        edges, _gt = lfr(t["n_cfg"], t["mu"], t["seed"])
        n = t["n_cfg"]
    else:
        from _exp_real_net.networks import LOADERS
        edges, n, _gt, _ev = LOADERS[t["net"]]()
    t0 = time.perf_counter()
    res = pymocd.smocc_probe(Shim(n, edges))
    dt = time.perf_counter() - t0
    front = [densify(p, n) for p in res["front"]]
    sel = front[res["selected"]]
    print("RESULT " + json.dumps({
        "n": int(n), "m": int(len(edges)), "k": int(len(np.unique(sel))),
        "time": round(dt, 4), "front_size": len(front),
        "hash_selected": digest(sel),
        "hash_front": digest(np.concatenate(front)),
    }), flush=True)


if __name__ == "__main__":
    main()
