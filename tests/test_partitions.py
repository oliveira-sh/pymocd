import unittest

import networkx as nx
import pymocd


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

    def test_mr_mocd(self):
        self.assert_valid_partition(pymocd.mr_mocd(self.graph))

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

    def test_mr_mocd_fronts(self):
        front, points, selected = pymocd.mr_mocd_fronts(self.graph)
        self.assertIsInstance(front, list)
        self.assertTrue(front)
        self.assertEqual(len(front), len(points))
        self.assertLess(selected, len(front))
        for partition in front:
            self.assert_valid_partition(partition)

    def test_mmcomo_fronts(self):
        front = pymocd.mmcomo_fronts(self.graph)
        self.assertIsInstance(front, list)
        self.assertTrue(front)
        for partition in front:
            self.assert_valid_partition(partition)

    def test_hpmocd_recovers_exact_partition(self):
        self.assert_exact_two_clique_split(pymocd.hpmocd(self.graph))

    def test_mr_mocd_recovers_exact_partition(self):
        self.assert_exact_two_clique_split(pymocd.mr_mocd(self.graph))


if __name__ == "__main__":
    unittest.main()
