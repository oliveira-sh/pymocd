import unittest

import networkx as nx
import pymocd


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_scale_is_deterministic(self):
        self.assertEqual(pymocd.scale(self.graph), pymocd.scale(self.graph))

    def test_scale_fronts_is_deterministic(self):
        self.assertEqual(
            pymocd.scale_fronts(self.graph), pymocd.scale_fronts(self.graph)
        )

    def test_hpmocd_repeated_runs_are_valid(self):

        nodes = set(self.graph.nodes)
        for _ in range(2):
            result = pymocd.hpmocd(self.graph)
            self.assertEqual(set(result), nodes)
            self.assertTrue(all(isinstance(c, int) for c in result.values()))


if __name__ == "__main__":
    unittest.main()
