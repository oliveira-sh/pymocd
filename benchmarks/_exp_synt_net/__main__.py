import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .runner import ExperimentRunner
from .constants import NUM_RUNS

if __name__ == "__main__":
    runner = ExperimentRunner(n_runs=NUM_RUNS)
    runner.run_mu_experiment(
        mus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_nodes_list=[10_000, 50_000],
    )
    runner.run_nodes_experiment(
        n_list=[10_000, 25_000, 50_000, 100_000],
        mus=[0.3, 0.5],
    )
