import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _exp_synt_net.hardened import ALGORITHMS, RUNS, SMOKE, run_campaign

REAL_SIZES = {"karate": 34, "dolphins": 62, "lesmis": 77, "florentine": 15,
              "polbooks": 105, "football": 115, "email_eu": 1005,
              "dblp": 317_080, "amazon": 334_863, "youtube": 1_134_890,
              "lj": 3_997_962, "orkut": 3_072_441}
REAL_SMALL = ["karate", "dolphins", "lesmis", "florentine", "polbooks",
              "football", "email_eu"]
REAL_SNAP = [] if SMOKE else ["dblp", "amazon", "youtube", "lj", "orkut"]


def real_tasks():
    tasks = []
    for net in REAL_SMALL + REAL_SNAP:
        n = REAL_SIZES[net]
        for alg, info in ALGORITHMS.items():
            if info["max_nodes"] is not None and n > info["max_nodes"]:
                continue
            runs = 1 if info["deterministic"] else RUNS
            for seed in range(runs):
                tasks.append({"alg": alg, "kind": "real", "net": net,
                              "seed": seed, "_n": n,
                              "_family": ("real", alg)})
    return tasks


if __name__ == "__main__":
    run_campaign(real_tasks())
    print("HARDENED_REAL_DONE", flush=True)
