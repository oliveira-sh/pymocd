import unittest

import networkx as nx
import pymocd

from tests._gpu import requires_gpu


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    @requires_gpu
    def test_gmocs_repeated_runs_are_valid(self):
        nodes = set(self.graph.nodes)
        for _ in range(2):
            result = pymocd.gmocs(self.graph)
            self.assertEqual(set(result), nodes)
            self.assertTrue(all(isinstance(c, int) for c in result.values()))

    def test_hpmocd_repeated_runs_are_valid(self):

        nodes = set(self.graph.nodes)
        for _ in range(2):
            result = pymocd.hpmocd(self.graph)
            self.assertEqual(set(result), nodes)
            self.assertTrue(all(isinstance(c, int) for c in result.values()))


if __name__ == "__main__":
    unittest.main()
