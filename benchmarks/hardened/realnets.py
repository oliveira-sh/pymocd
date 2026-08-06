"""Loaders return (edges uint32 [m,2] relabelled 0..n-1, n, gt labels or
None, eval_nodes or None). Dolphins ground truth (Lusseau's two groups) comes
from the vlivashkin/community-graphs `gt` attribute; com-* use the SNAP
top-5000 sets restricted to single-membership nodes. lesmis and florentine
have no accepted crisp ground truth in the literature — modularity only."""

import gzip
import os
import sys
import urllib.request
import zipfile

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
DATA = os.path.join(BENCH, "data")
sys.path.insert(0, BENCH)

DOLPHINS_GT_URL = ("https://raw.githubusercontent.com/vlivashkin/"
                   "community-graphs/master/gml_connected_subgraphs/"
                   "dolphins.gml")


def _fetch(url, fname=None):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, fname or os.path.basename(url))
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(path, "wb") as out:
            out.write(resp.read())
    return path


def _from_nx(G, gt_map=None):
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = np.array(
        [(idx[u], idx[v]) for u, v in G.edges() if u != v], dtype=np.uint32)
    if gt_map is None:
        return edges, len(nodes), None, None
    labels = sorted(set(gt_map.values()), key=str)
    lab_idx = {l: i for i, l in enumerate(labels)}
    eval_nodes = np.array([idx[v] for v in nodes if v in gt_map],
                          dtype=np.int64)
    gt = np.array([lab_idx[gt_map[nodes[i]]] for i in eval_nodes],
                  dtype=np.int64)
    return edges, len(nodes), gt, eval_nodes


def _newman(name):
    path = _fetch(f"https://websites.umich.edu/~mejn/netdata/{name}.zip")
    with zipfile.ZipFile(path) as z:
        gml = z.read(f"{name}.gml").decode()
    G = nx.Graph(nx.parse_gml(gml, label="id"))
    values = nx.get_node_attributes(G, "value")
    return G, (values if len(values) == len(G) else None)


def load_karate():
    G = nx.karate_club_graph()
    return _from_nx(G, {v: G.nodes[v]["club"] for v in G})


def load_dolphins():
    path = _fetch(DOLPHINS_GT_URL, "dolphins_gt.gml")
    G = nx.Graph(nx.read_gml(path, label="id"))
    return _from_nx(G, nx.get_node_attributes(G, "gt"))


def load_lesmis():
    return _from_nx(nx.Graph(nx.les_miserables_graph()))


def load_florentine():
    G = nx.florentine_families_graph()
    return _from_nx(G.subgraph([v for v, d in G.degree() if d > 0]).copy())


def load_polbooks():
    return _from_nx(*_newman("polbooks"))


def load_football():
    return _from_nx(*_newman("football"))


def load_email_eu():
    epath = _fetch("https://snap.stanford.edu/data/email-Eu-core.txt.gz")
    lpath = _fetch(
        "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz")
    with gzip.open(epath, "rt") as f:
        G = nx.Graph(nx.parse_edgelist(f, nodetype=int))
    G.remove_edges_from(nx.selfloop_edges(G))
    with gzip.open(lpath, "rt") as f:
        gt = {int(a): int(b) for a, b in (line.split() for line in f)}
    return _from_nx(G, gt)


SNAP_BASE = "https://snap.stanford.edu/data/bigdata/communities/"


def load_snap(name):
    import pandas as pd
    epath = _fetch(f"{SNAP_BASE}com-{name}.ungraph.txt.gz")
    e = pd.read_csv(epath, sep="\t", comment="#", header=None,
                    dtype=np.int64).values
    ids = np.unique(e)
    edges = np.searchsorted(ids, e).astype(np.uint32)

    cpath = _fetch(f"{SNAP_BASE}com-{name}.top5000.cmty.txt.gz")
    gt: dict[int, int] = {}
    multi: set[int] = set()
    with gzip.open(cpath, "rt") as f:
        for ci, line in enumerate(f):
            for tok in line.split():
                v = int(tok)
                if v in gt and gt[v] != ci:
                    multi.add(v)
                gt[v] = ci
    singles = np.array(sorted(set(gt) - multi), dtype=np.int64)
    pos = np.searchsorted(ids, singles)
    present = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == singles)
    eval_nodes = pos[present].astype(np.int64)
    eval_labels = np.array([gt[int(v)] for v in singles[present]],
                           dtype=np.int64)
    return edges, len(ids), eval_labels, eval_nodes


LOADERS = {
    "karate": load_karate,
    "dolphins": load_dolphins,
    "lesmis": load_lesmis,
    "florentine": load_florentine,
    "polbooks": load_polbooks,
    "football": load_football,
    "email_eu": load_email_eu,
    "dblp": lambda: load_snap("dblp"),
    "amazon": lambda: load_snap("amazon"),
    "youtube": lambda: load_snap("youtube"),
    "lj": lambda: load_snap("lj"),
    "orkut": lambda: load_snap("orkut"),
}
