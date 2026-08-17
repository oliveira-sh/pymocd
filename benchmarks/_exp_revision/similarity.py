"""Sparse similarity against the dense diffusion kernel it replaces (M4, Q7).

Two experiments on graphs small enough for an n x n kernel.

``decode``  Fix the graph and the centre set, then decode the same genomes
            under (a) unit edge weights, (b) the kernel restricted to the
            edges, (c) the full dense kernel with nearest-centre assignment.
            Reports pairwise agreement and the distance between the restricted
            kernel and the similarity SMOCC ends the run with.

``search``  Run the whole search under each medium and compare partition
            quality, front size and per-generation cost.

    python -m _exp_revision.similarity
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _exp_revision.common import OUT, Shim, Sink, densify, lfr  # noqa: E402

DEC_FIELDS = ["net", "n_cfg", "mu", "seed", "beta", "n", "m", "n_genomes",
              "ami_unit_kedge", "ami_unit_dense", "ami_kedge_dense",
              "ari_unit_kedge", "ari_unit_dense", "ari_kedge_dense",
              "k_unit", "k_kedge", "k_dense",
              "ami_gt_unit", "ami_gt_kedge", "ami_gt_dense",
              "l2_kedge_wfinal", "linf_kedge_wfinal", "corr_kedge_wfinal",
              "l2_kedge_unit", "mean_kedge", "sd_kedge", "mean_wfinal",
              "t_kernel", "stamp"]
DEC_KEY = ["net", "n_cfg", "mu", "seed", "beta"]

SEA_FIELDS = ["net", "n_cfg", "mu", "seed", "medium", "beta", "n", "m", "k",
              "gt_k", "nmi", "ami", "ari", "modularity", "front_size",
              "t_total", "t_macro", "t_micro", "t_exchange", "sweeps_mean",
              "stamp"]
SEA_KEY = ["net", "n_cfg", "mu", "seed", "medium", "beta"]

MEDIA = {"sparse": 0, "kernel-edge": 1, "dense": 2}


def sample_genomes(n, cmax, count, rng):
    """Centre sets drawn the way macro initialisation draws them."""
    out = []
    for _ in range(count):
        c = rng.integers(1, cmax + 1)
        g = np.zeros(n, dtype=np.uint8)
        g[rng.choice(n, size=int(c), replace=False)] = 1
        out.append(g.tolist())
    return out


def decode_case(sink, pymocd, net, n_cfg, mu, seed, n, edges, gt, beta,
                count=32):
    base = {"net": net, "n_cfg": n_cfg, "mu": mu, "seed": seed, "beta": beta}
    if sink.has(base):
        return
    from sklearn.metrics.cluster import (adjusted_mutual_info_score as ami,
                                         adjusted_rand_score as ari)
    rng = np.random.default_rng(1000 + seed)
    cmax = int(np.ceil(np.sqrt(n)))
    genomes = sample_genomes(n, cmax, count, rng)
    shim = Shim(n, edges)
    t0 = time.perf_counter()
    med = pymocd.smocc_decode_media(shim, genomes, beta)
    t_kernel = time.perf_counter() - t0

    unit = [np.asarray(x) for x in med["unit"]]
    kedge = [np.asarray(x) for x in med["kernel_edge"]]
    dense = [np.asarray(x) for x in med["dense"]]
    pair = lambda f, a, b: float(np.mean([f(x, y) for x, y in zip(a, b)]))

    res = pymocd.smocc_probe(shim, want_w=True)
    w_final = np.asarray(res["w_values"], dtype=float)
    ke = np.asarray(med["w_kernel_edge"], dtype=float)
    row = dict(base, n=n, m=int(len(edges)), n_genomes=count,
               t_kernel=round(t_kernel, 4))
    row.update({
        "ami_unit_kedge": round(pair(ami, unit, kedge), 6),
        "ami_unit_dense": round(pair(ami, unit, dense), 6),
        "ami_kedge_dense": round(pair(ami, kedge, dense), 6),
        "ari_unit_kedge": round(pair(ari, unit, kedge), 6),
        "ari_unit_dense": round(pair(ari, unit, dense), 6),
        "ari_kedge_dense": round(pair(ari, kedge, dense), 6),
        "k_unit": round(float(np.mean([len(np.unique(x)) for x in unit])), 3),
        "k_kedge": round(float(np.mean([len(np.unique(x)) for x in kedge])), 3),
        "k_dense": round(float(np.mean([len(np.unique(x)) for x in dense])), 3),
    })
    if gt is not None:
        row["ami_gt_unit"] = round(float(np.mean([ami(gt, x) for x in unit])), 6)
        row["ami_gt_kedge"] = round(float(np.mean([ami(gt, x) for x in kedge])), 6)
        row["ami_gt_dense"] = round(float(np.mean([ami(gt, x) for x in dense])), 6)
    if w_final.size == ke.size and ke.size:
        # The kernel carries an arbitrary scale; compare after normalising each
        # medium to unit mean, which is the scale a max-vote decoder sees.
        kn = ke / ke.mean() if ke.mean() else ke
        wn = w_final / w_final.mean() if w_final.mean() else w_final
        row["l2_kedge_wfinal"] = round(float(np.linalg.norm(kn - wn) / np.sqrt(kn.size)), 6)
        row["linf_kedge_wfinal"] = round(float(np.abs(kn - wn).max()), 6)
        row["corr_kedge_wfinal"] = round(float(np.corrcoef(ke, w_final)[0, 1]), 6)
        row["l2_kedge_unit"] = round(float(np.linalg.norm(kn - 1.0) / np.sqrt(kn.size)), 6)
        row["mean_kedge"] = round(float(ke.mean()), 8)
        row["sd_kedge"] = round(float(ke.std()), 8)
        row["mean_wfinal"] = round(float(w_final.mean()), 6)
    sink.write(row)


def search_case(sink, pymocd, net, n_cfg, mu, seed, n, edges, gt, beta):
    from _exp_revision.common import score
    for medium, code in MEDIA.items():
        base = {"net": net, "n_cfg": n_cfg, "mu": mu, "seed": seed,
                "medium": medium, "beta": beta}
        if sink.has(base):
            continue
        shim = Shim(n, edges)
        t0 = time.perf_counter()
        res = pymocd.smocc_probe(shim, sim_mode=code, beta=beta)
        dt = time.perf_counter() - t0
        lab = densify(res["front"][res["selected"]], n)
        d = res["diag"]
        row = dict(base, n=n, m=int(len(edges)), time=round(dt, 4),
                   front_size=d["front_size"], t_total=round(d["t_total"], 4),
                   t_macro=round(d["t_macro"], 4),
                   t_micro=round(d["t_micro"], 4),
                   t_exchange=round(d["t_exchange"], 4),
                   sweeps_mean=round(float(np.mean(d["sweeps"])), 3))
        row.update(score(gt, lab, n=n, edges=edges))
        if gt is not None:
            row["gt_k"] = int(len(np.unique(gt)))
        sink.write(row)
        print(f"  {medium}: ami={row.get('ami')} k={row['k']} "
              f"t={row['t_total']:.1f}s", flush=True)


def main():
    import pymocd
    from _exp_real_net.networks import LOADERS
    pymocd.max_cores(int(os.environ.get("REV_THREADS", "48")))
    dec = Sink("similarity_decode.csv", DEC_FIELDS, DEC_KEY)
    sea = Sink("similarity_search.csv", SEA_FIELDS, SEA_KEY)
    betas = [float(b) for b in os.environ.get("REV_BETA", "0.05").split(",")]

    nets = os.environ.get(
        "REV_SIM_NETS",
        "karate,dolphins,polbooks,football,lesmis,florentine,email_eu").split(",")
    for net in nets:
        edges, n, gt, eval_nodes = LOADERS[net]()
        sub = gt if gt is None else gt
        for beta in betas:
            print(f"{net} (n={n}) beta={beta}", flush=True)
            g_eval = None if gt is None else np.asarray(gt)
            full_gt = None
            if g_eval is not None and len(g_eval) == n:
                full_gt = g_eval
            decode_case(dec, pymocd, net, "", "", 0, n, edges, full_gt, beta)
            search_case(sea, pymocd, net, "", "", 0, n, edges, full_gt, beta)

    grid = json.loads(os.environ.get(
        "REV_SIM_LFR", '[[300,0.3],[300,0.5],[600,0.3],[600,0.5],'
                       '[1000,0.3],[1000,0.5],[2000,0.3],[2000,0.5]]'))
    runs = int(os.environ.get("REV_SIM_RUNS", "5"))
    for n_cfg, mu in grid:
        for seed in range(runs):
            edges, gt = lfr(n_cfg, mu, seed)
            for beta in betas:
                print(f"lfr n={n_cfg} mu={mu} s={seed} beta={beta}", flush=True)
                decode_case(dec, pymocd, f"lfr{n_cfg}", n_cfg, mu, seed,
                            n_cfg, edges, gt, beta)
                search_case(sea, pymocd, f"lfr{n_cfg}", n_cfg, mu, seed,
                            n_cfg, edges, gt, beta)
    print(f"SIMILARITY_DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
