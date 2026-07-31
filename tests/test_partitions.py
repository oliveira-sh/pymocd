import unittest

import networkx as nx
import pymocd


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

    def test_scale(self):
        self.assert_valid_partition(pymocd.scale(self.graph))

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

    def test_scale_fronts(self):
        front = pymocd.scale_fronts(self.graph)
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


if __name__ == "__main__":
    unittest.main()
