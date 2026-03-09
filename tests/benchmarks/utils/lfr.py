import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


def generate_lfr_benchmark(
    n=1000, tau1=2.5, tau2=1.5, mu=0.3, average_degree=20, min_community=20, seed=0
):
    G = nx.generators.community.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        average_degree=average_degree,
        min_community=min_community,
        max_degree=50,
        seed=seed,
        max_community=100,
    )
    communities = {node: frozenset(G.nodes[node]["community"]) for node in G}
    G = nx.Graph(G)
    return G, communities


def convert_communities_to_partition(communities):
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition


def evaluate_communities(
    G, detected_communities, ground_truth_communities, convert=True
):
    if convert:
        detected_partition = convert_communities_to_partition(detected_communities)
    else:
        detected_partition = detected_communities

    if not detected_partition:
        return {"modularity": 0.0, "nmi": 0.0, "ami": 0.0}

    ground_truth_partition = {}
    for node, comms in ground_truth_communities.items():
        ground_truth_partition[node] = (
            list(comms)[0] if isinstance(comms, frozenset) else comms
        )

    communities_as_list = []
    max_community = max(detected_partition.values())
    for i in range(max_community + 1):
        community = {node for node, comm in detected_partition.items() if comm == i}
        if community:
            communities_as_list.append(community)

    mod = modularity(G, communities_as_list)
    n_nodes = len(G.nodes())
    gt_labels = np.zeros(n_nodes, dtype=np.int32)
    detected_labels = np.zeros(n_nodes, dtype=np.int32)

    for i, node in enumerate(sorted(G.nodes())):
        gt_labels[i] = ground_truth_partition[node]
        detected_labels[i] = detected_partition[node]

    nmi = normalized_mutual_info_score(gt_labels, detected_labels)
    ami = adjusted_mutual_info_score(gt_labels, detected_labels)

    return {"modularity": mod, "nmi": nmi, "ami": ami}
