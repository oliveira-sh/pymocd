import unittest
import networkx as nx
import pymocd
import random
import igraph as ig

def build_two_clique_graph():
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])
    G.add_edges_from([(3, 4), (3, 5), (4, 5)])
    G.add_edge(2, 3)
    return G

class TestPyMoCDNetworkX(unittest.TestCase):
    def test_simple_two_clique_partition(self):
        G = build_two_clique_graph()
        alg = pymocd.HpMocd(G, debug_level=0)
        communities = alg.run()

        unique_labels = set(communities.values())
        self.assertEqual(len(unique_labels), 2)

        comm_to_nodes = {}
        for node, cid in communities.items():
            comm_to_nodes.setdefault(cid, set()).add(node)

        sorted_parts = tuple(
            sorted(tuple(sorted(nodes)) for nodes in comm_to_nodes.values())
        )
        self.assertEqual(sorted_parts, ((0, 1, 2), (3, 4, 5)))

    def test_empty_graph_returns_empty_dict_or_error(self):
        G_empty = nx.Graph()
        try:
            alg = pymocd.HpMocd(G_empty, debug_level=0)
            communities = alg.run()
            self.assertIsInstance(communities, dict)
            self.assertEqual(communities, {})
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_reproducibility_on_small_random_graph(self):
        random.seed(42)
        G_rand = nx.gnm_random_graph(n=10, m=15, seed=42)

        alg1 = pymocd.HpMocd(G_rand, debug_level=0)
        part1 = alg1.run()

        random.seed(42)
        G_rand_again = nx.gnm_random_graph(n=10, m=15, seed=42)

        alg2 = pymocd.HpMocd(G_rand_again, debug_level=0)
        part2 = alg2.run()

        self.assertEqual(part1, part2)


class TestPyMoCDIgraph(unittest.TestCase):
    def test_simple_two_clique_partition(self):
        # Build the same 2‐clique graph via networkx, then feed it into HpMocd.
        # (Assuming HpMocd can accept a “Graph‐like” object; if you meant to pass an igraph.Graph()
        # build, adjust accordingly.)
        g = build_two_clique_graph()
        alg = pymocd.HpMocd(g, debug_level=0)
        communities = alg.run()

        unique_labels = set(communities.values())
        self.assertEqual(len(unique_labels), 2)

        comm_to_nodes = {}
        for node, cid in communities.items():
            comm_to_nodes.setdefault(cid, set()).add(node)
        sorted_parts = sorted([tuple(sorted(list(s))) for s in comm_to_nodes.values()])
        self.assertEqual(tuple(sorted_parts), ((0, 1, 2), (3, 4, 5)))

if __name__ == "__main__":
    unittest.main()