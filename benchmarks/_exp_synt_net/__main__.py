import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .runner import ExperimentRunner
from .constants import NUM_RUNS


def _env_list(name, default, cast):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [cast(item) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    runner = ExperimentRunner(n_runs=NUM_RUNS)
    # A sweep that varies both mu and nodes is plotted as facets; varying exactly
    # one axis gets the per-metric plots instead.
    runner.run_mu_experiment(
        mus=_env_list("PYMOCD_BENCH_MUS", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], float),
        n_nodes_list=_env_list("PYMOCD_BENCH_NODES", [10_000, 50_000], int),
    )
    if not os.environ.get("PYMOCD_BENCH_SKIP_NODES"):
        runner.run_nodes_experiment(
            n_list=_env_list(
                "PYMOCD_BENCH_NODES_SWEEP", [10_000, 25_000, 50_000, 100_000], int
            ),
            mus=_env_list("PYMOCD_BENCH_NODES_MU", [0.3, 0.5], float),
        )
