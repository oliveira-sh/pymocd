import json
import subprocess
import sys
import unittest

import networkx as nx
import pymocd

# child script: set the thread count, run rimpso on karate, print the partition
CHILD = """
import json, sys
import networkx as nx
import pymocd

pymocd.max_cores(int(sys.argv[1]))
part = pymocd.rimpso(nx.karate_club_graph())
print(json.dumps(sorted((int(k), int(v)) for k, v in part.items())))
"""


def rimpso_under_threads(threads):
    # a fresh process per thread count, since max_cores only takes effect once
    proc = subprocess.run([sys.executable, "-c", CHILD, str(threads)],
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()

    def test_rimpso_is_deterministic(self):
        self.assertEqual(pymocd.rimpso(self.graph), pymocd.rimpso(self.graph))

    def test_rimpso_fronts_is_deterministic(self):
        # (partitions, (cut, pair) points, selected index), floats compared exactly
        self.assertEqual(
            pymocd.rimpso_fronts(self.graph), pymocd.rimpso_fronts(self.graph)
        )

    def test_rimpso_is_thread_count_independent(self):
        # max_cores builds rayon's global pool once per process, so the two
        # thread counts have to come from two child processes
        self.assertEqual(rimpso_under_threads(1), rimpso_under_threads(4))

    def test_hpmocd_repeated_runs_are_valid(self):

        nodes = set(self.graph.nodes)
        for _ in range(2):
            result = pymocd.hpmocd(self.graph)
            self.assertEqual(set(result), nodes)
            self.assertTrue(all(isinstance(c, int) for c in result.values()))


if __name__ == "__main__":
    unittest.main()
