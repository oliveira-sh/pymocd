import unittest

import networkx as nx
import pymocd
from networkx.algorithms.community import modularity

from tests._gpu import requires_gpu


def as_communities(partition):
    groups = {}
    for node, community in partition.items():
        groups.setdefault(community, set()).add(node)
    return list(groups.values())


class TestSanity(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    @requires_gpu
    def test_gmocs_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.gmocs(self.graph)))
        self.assertGreater(q, 0.25)

    def test_hpmocd_modularity_floor_on_karate(self):
        q = modularity(self.graph, as_communities(pymocd.hpmocd(self.graph)))
        self.assertGreater(q, 0.25)

    @requires_gpu
    def test_metrics_perfect_self_agreement(self):
        partition = pymocd.gmocs(self.graph)
        for value in pymocd.gt_metrics(partition, partition):
            self.assertAlmostEqual(value, 1.0)
        self.assertAlmostEqual(pymocd.nmi(partition, partition), 1.0)

    @requires_gpu
    def test_isolated_node_gets_minus_one(self):
        self.graph.add_node(99)
        self.assertEqual(pymocd.gmocs(self.graph)[99], -1)

    @requires_gpu
    def test_single_community_graph(self):
        partition = pymocd.gmocs(nx.complete_graph(6))
        self.assertEqual(set(partition), set(range(6)))

    def test_empty_graph(self):
        self.assertEqual(pymocd.gmocs(nx.Graph()), {})


if __name__ == "__main__":
    unittest.main()
