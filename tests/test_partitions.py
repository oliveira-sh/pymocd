import unittest

import networkx as nx
import pymocd

try:
    from tests._gpu import requires_gpu
except ImportError:
    from _gpu import requires_gpu


def build_two_clique_graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 1), (0, 2), (1, 2)])
    graph.add_edges_from([(3, 4), (3, 5), (4, 5)])
    graph.add_edge(2, 3)
    return graph


class TestPartitions(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def assert_valid_partition(self, result):
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), set(self.graph.nodes))
        groups = {}
        for node, community in result.items():
            self.assertIsInstance(community, int)
            groups.setdefault(community, []).append(node)
        for community, members in groups.items():
            self.assertTrue(members, f"community {community} is empty")

    @requires_gpu
    def test_gmocs(self):
        self.assert_valid_partition(pymocd.gmocs(self.graph))

    def test_hpmocd(self):
        self.assert_valid_partition(pymocd.hpmocd(self.graph))

    def test_mocd_q(self):
        self.assert_valid_partition(pymocd.mocd_q(self.graph))

    def test_mocd_d(self):
        self.assert_valid_partition(pymocd.mocd_d(self.graph))

    def test_moga_net(self):
        self.assert_valid_partition(pymocd.moga_net(self.graph))

    def test_ccm(self):
        self.assert_valid_partition(pymocd.ccm(self.graph))

    def test_krm(self):
        self.assert_valid_partition(pymocd.krm(self.graph))

    def test_mmcomo(self):
        self.assert_valid_partition(pymocd.mmcomo(self.graph))

    @requires_gpu
    def test_gmocs_fronts(self):
        front = pymocd.gmocs_fronts(self.graph)
        self.assertIsInstance(front, list)
        self.assertTrue(front)
        for partition in front:
            self.assert_valid_partition(partition)

    def test_mmcomo_fronts(self):
        front = pymocd.mmcomo_fronts(self.graph)
        self.assertIsInstance(front, list)
        self.assertTrue(front)
        for partition in front:
            self.assert_valid_partition(partition)


class TestTwoCliquePartition(unittest.TestCase):
    def setUp(self):
        self.graph = build_two_clique_graph()

    def assert_exact_two_clique_split(self, result):
        groups = {}
        for node, community in result.items():
            groups.setdefault(community, set()).add(node)
        self.assertEqual(len(groups), 2)
        parts = tuple(sorted(tuple(sorted(nodes)) for nodes in groups.values()))
        self.assertEqual(parts, ((0, 1, 2), (3, 4, 5)))

    def test_hpmocd_recovers_exact_partition(self):
        self.assert_exact_two_clique_split(pymocd.hpmocd(self.graph))

    @requires_gpu
    def test_gmocs_recovers_exact_partition(self):
        self.assert_exact_two_clique_split(pymocd.gmocs(self.graph))


if __name__ == "__main__":
    unittest.main()
