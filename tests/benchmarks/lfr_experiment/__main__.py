import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfr_experiment.algorithms import NUM_RUNS, NUM_ND
from lfr_experiment.runner import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner(n_runs=NUM_RUNS)
    runner.run_mu_experiment(n_nodes=NUM_ND)
    runner.run_nodes_experiment()
