import networkx as nx
import pymocd
import community as community_louvain
import igraph as ig
from networkx.algorithms.community import asyn_lpa_communities

from .registry import algorithm, _safe, _with_seed


def _ensure_int_nodes(G):
    if all(isinstance(n, int) and n >= 0 for n in G.nodes()):
        return G, None
    mapping = {old: i for i, old in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping, copy=True)
    inverse = {i: old for old, i in mapping.items()}
    return H, inverse


def _restore_labels(part, inverse):
    if inverse is None:
        return part
    return {inverse[n]: c for n, c in part.items()}


@algorithm("Scale", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def scale_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.scale(H), inverse)


@algorithm("HPMOCD", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def hpmocd_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.hpmocd(H), inverse)


# Dense n x n diffusion kernel + eigendecomposition, roughly O(n^3):
# 517 s at n=1458, so it cannot take part in the larger sweeps.
@algorithm("MMCoMO", needs_conversion=False, parallel=False, max_nodes=2000)
@_safe
@_with_seed
def mmcomo_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.mmcomo(H), inverse)


@algorithm("NSGA III CCM", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def ccm_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.ccm(H), inverse)


@algorithm("NSGA III KRM", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def krm_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.krm(H), inverse)


@algorithm("Shi-MOCD (Q)", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def mocd_q_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.mocd_q(H), inverse)


@algorithm("Shi-MOCD (D)", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def mocd_d_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.mocd_d(H), inverse)


@algorithm("MOGA-Net", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def moga_net_algorithm(G):
    H, inverse = _ensure_int_nodes(G)
    return _restore_labels(pymocd.moga_net(H), inverse)


@algorithm("Louvain", needs_conversion=False, parallel=True)
@_safe
@_with_seed
def louvain_algorithm(G):
    return community_louvain.best_partition(G)


@algorithm("Leiden", needs_conversion=False, parallel=True)
@_safe
@_with_seed
def leiden_algorithm(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ig_graph = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    partition = ig_graph.community_leiden(objective_function="modularity")
    return {nodes[i]: partition.membership[i] for i in range(ig_graph.vcount())}


@algorithm("ASYN-LPA", needs_conversion=True, parallel=True)
@_safe
@_with_seed
def asyn_lpa_algorithm(G):
    return list(asyn_lpa_communities(G))
