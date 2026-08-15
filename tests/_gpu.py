import unittest

import networkx as nx
import pymocd


def _gpu_available():
    try:
        pymocd.gmocs(nx.complete_graph(4), pop_size=4, num_gens=2)
        return True
    except RuntimeError:
        return False


GPU_AVAILABLE = _gpu_available()
requires_gpu = unittest.skipUnless(GPU_AVAILABLE, "no usable CUDA device")
