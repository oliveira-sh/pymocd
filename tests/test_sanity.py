import unittest

import networkx as nx
import pymocd
from networkx.algorithms.community import modularity


def as_communities(partition):
    groups = {}
    for node, community in partition.items():
        groups.setdefault(community, set()).add(node)
    return list(groups.values())


def cut_and_pair(graph, partition):
    active = [node for node in graph if graph.degree(node) > 0]
    internal = sum(1 for u, v in graph.edges if partition[u] == partition[v])
    sizes = {}
    for node in active:
        sizes[partition[node]] = sizes.get(partition[node], 0) + 1
    co_clustered = sum(size * (size - 1) // 2 for size in sizes.values())
    all_pairs = len(active) * (len(active) - 1) // 2
    m = graph.number_of_edges()
    cut = 1.0 - internal / m if m else 0.0
    pair = co_clustered / all_pairs if all_pairs else 0.0
    return cut, pair


class TestSanity(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_mr_mocd_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.mr_mocd(self.graph)))
        self.assertGreater(q, 0.25)

    def test_hpmocd_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.hpmocd(self.graph)))
        self.assertGreater(q, 0.25)

    def test_metrics_perfect_self_agreement(self):
        partition = pymocd.mr_mocd(self.graph)
        for value in pymocd.gt_metrics(partition, partition):
            self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(pymocd.nmi(partition, partition), 1.0)

    def test_isolated_node_gets_minus_one(self):
        self.graph.add_node(99)
        self.assertEqual(pymocd.mr_mocd(self.graph)[99], -1)

    def test_mr_mocd_front_points_match_their_partitions(self):
        self.graph.add_nodes_from([99, 100])
        front, points, _selected = pymocd.mr_mocd_fronts(
            self.graph, pop_size=8, num_gens=2
        )
        self.assertTrue(front)
        for partition, (cut, pair) in zip(front, points):
            self.assertEqual(partition[99], -1)
            self.assertEqual(partition[100], -1)
            self.assertEqual((cut, pair), cut_and_pair(self.graph, partition))

    def test_single_community_graph(self):
        partition = pymocd.mr_mocd(nx.complete_graph(6))
        self.assertEqual(set(partition), set(range(6)))

    def test_empty_graph(self):
        self.assertEqual(pymocd.mr_mocd(nx.Graph()), {})


if __name__ == "__main__":
    unittest.main()
