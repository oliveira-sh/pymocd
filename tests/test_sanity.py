import unittest

import networkx as nx
import pymocd
from networkx.algorithms.community import modularity


def as_communities(partition):
    groups = {}
    for node, community in partition.items():
        groups.setdefault(community, set()).add(node)
    return list(groups.values())


class TestSanity(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_smocc_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.smocc(self.graph)))
        self.assertGreater(q, 0.25)

    def test_hpmocd_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.hpmocd(self.graph)))
        self.assertGreater(q, 0.25)

    def test_metrics_perfect_self_agreement(self):
        partition = pymocd.smocc(self.graph)
        for value in pymocd.gt_metrics(partition, partition):
            self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(pymocd.nmi(partition, partition), 1.0)

    def test_isolated_node_gets_minus_one(self):
        self.graph.add_node(99)
        self.assertEqual(pymocd.smocc(self.graph)[99], -1)

    def test_single_community_graph(self):
        partition = pymocd.smocc(nx.complete_graph(6))
        self.assertEqual(set(partition), set(range(6)))

    def test_empty_graph(self):
        self.assertEqual(pymocd.smocc(nx.Graph()), {})


if __name__ == "__main__":
    unittest.main()
