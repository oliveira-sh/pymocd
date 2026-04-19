import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfr_experiment.runner import ExperimentRunner
from .constants import NUM_RUNS

if __name__ == "__main__":
    runner = ExperimentRunner(n_runs=NUM_RUNS)
    # 2D sweep: mu on x-axis, faceted by graph size
    runner.run_mu_experiment(
        mus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_nodes_list=[10_000, 50_000],
    )
    # 2D sweep: nodes on x-axis, faceted by fixed mu.
    runner.run_nodes_experiment(
        n_list=[10_000, 25_000, 50_000, 100_000],
        mus=[0.3, 0.5],
    )
