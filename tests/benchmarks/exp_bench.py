"""Ablation & next-gen experiment harness for PRISM.

Run from tests/benchmarks:
    python exp_bench.py <tag>

Saves results/exp_<tag>.json. Cheap: n=500, 3 seeds, μ∈[0.1..0.8].
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils import generate_lfr_benchmark, evaluate_communities
import igraph as ig
import pymocd

MUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = [0, 1, 2]
N = 500
SWARM = 60
GENS = 60


def leiden(G, seed):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    part = ig_g.community_leiden(objective_function="modularity")
    return {nodes[i]: part.membership[i] for i in range(ig_g.vcount())}


def prism(G, seed, **kw):
    polish = kw.pop("polish_iters", 20)
    model = pymocd.Prism(G, swarm_size=SWARM, num_gens=GENS, **kw)
    return model.run(polish_iters=polish)


def run(variant_name, algo_fn, tag):
    out = defaultdict(list)
    for mu in MUS:
        for s in SEEDS:
            G, gt = generate_lfr_benchmark(n=N, mu=mu, seed=s)
            t0 = time.time()
            comm = algo_fn(G, s)
            dur = time.time() - t0
            ev = evaluate_communities(G, comm, gt, convert=False)
            out[mu].append({
                "ami": ev["ami"],
                "nmi": ev["nmi"],
                "mod": ev["modularity"],
                "time": dur,
            })
            print(f"  {variant_name} μ={mu} s={s}: AMI={ev['ami']:.3f} "
                  f"NMI={ev['nmi']:.3f} Q={ev['modularity']:.3f} t={dur:.2f}s",
                  flush=True)
    agg = {}
    for mu, runs in out.items():
        agg[str(mu)] = {
            "ami": float(np.mean([r["ami"] for r in runs])),
            "nmi": float(np.mean([r["nmi"] for r in runs])),
            "mod": float(np.mean([r["mod"] for r in runs])),
            "time": float(np.mean([r["time"] for r in runs])),
            "ami_std": float(np.std([r["ami"] for r in runs])),
        }
    path = f"results/exp_{tag}.json"
    os.makedirs("results", exist_ok=True)
    prev = {}
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
    prev[variant_name] = agg
    with open(path, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"  -> saved {path}[{variant_name}]")
    return agg


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "default"
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["leiden", "prism"]
    for v in variants:
        if v == "leiden":
            run("leiden", leiden, tag)
        elif v == "prism":
            run("prism", prism, tag)
        elif v == "prism_no_polish":
            run("prism_no_polish", lambda G, s: prism(G, s, polish_iters=0), tag)
