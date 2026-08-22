import json
import subprocess
import sys
import unittest

import networkx as nx
import pymocd

# child script: set the thread count, run mopots on karate, print the partition
CHILD = """
import json, sys
import networkx as nx
import pymocd

pymocd.max_cores(int(sys.argv[1]))
part = pymocd.mopots(nx.karate_club_graph())
print(json.dumps(sorted((int(k), int(v)) for k, v in part.items())))
"""


def mopots_under_threads(threads):
    # a fresh process per thread count, since max_cores only takes effect once
    proc = subprocess.run([sys.executable, "-c", CHILD, str(threads)],
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_smocc_is_deterministic(self):
        self.assertEqual(pymocd.smocc(self.graph), pymocd.smocc(self.graph))

    def test_smocc_fronts_is_deterministic(self):
        self.assertEqual(
            pymocd.smocc_fronts(self.graph), pymocd.smocc_fronts(self.graph)
        )

    def test_mopots_is_deterministic(self):
        self.assertEqual(pymocd.mopots(self.graph), pymocd.mopots(self.graph))

    def test_mopots_fronts_is_deterministic(self):
        self.assertEqual(
            pymocd.mopots_fronts(self.graph), pymocd.mopots_fronts(self.graph)
        )

    def test_mopots_ladder_is_deterministic(self):
        # (partition, cut, pair, gamma) tuples, floats compared exactly
        self.assertEqual(
            pymocd.mopots_ladder(self.graph), pymocd.mopots_ladder(self.graph)
        )

    def test_mopots_is_thread_count_independent(self):
        # max_cores builds rayon's global pool once per process, so the two
        # thread counts have to come from two child processes
        self.assertEqual(mopots_under_threads(1), mopots_under_threads(4))

    def test_hpmocd_repeated_runs_are_valid(self):

        nodes = set(self.graph.nodes)
        for _ in range(2):
            result = pymocd.hpmocd(self.graph)
            self.assertEqual(set(result), nodes)
            self.assertTrue(all(isinstance(c, int) for c in result.values()))


if __name__ == "__main__":
    unittest.main()
