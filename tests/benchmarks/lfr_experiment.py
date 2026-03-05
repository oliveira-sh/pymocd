import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
from functools import wraps
from typing import Callable, Dict, Any, List

import community as community_louvain
import igraph as ig
import pymocd

from utils import (
    generate_lfr_benchmark,
    evaluate_communities,
    plot_results,
    read_results_from_csv,
    SAVE_PATH
)

CSV_FILE = os.path.join(SAVE_PATH, 'lfr.csv')
BACKUP_CSV_FILE = CSV_FILE.replace('.csv', '_bk.csv')
MIN_MU = 0.1
MAX_MU = 0.8
STEP_MU = 0.1
NUM_RUNS = 1
JUST_PLOT = False  # Set True to skip experiments and plot existing CSV
MU_EXPERIMENT = False  # True = vary mu, False = vary n

ALGORITHM_REGISTRY = {}

def algorithm(name: str, needs_conversion: bool = True, parallel: bool = True):
    """Decorator to register a community detection algorithm.

    Add a new algorithm by decorating a function that takes (G, seed=None)
    and returns a partition dict {node: community_id}.
    Set parallel=False for algorithms that use all CPU cores internally
    (they will run sequentially to avoid CPU oversubscription).
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        ALGORITHM_REGISTRY[name] = {
            'function': wrapper,
            'needs_conversion': needs_conversion,
            'parallel': parallel,
        }
        return wrapper
    return decorator

def _with_seed(func: Callable):
    """Set numpy seed before calling the function for reproducibility."""
    @wraps(func)
    def wrapper(G, seed=None, *args, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return func(G, *args, **kwargs)
    return wrapper

def _safe(func: Callable):
    """Return empty dict on exception instead of crashing the whole run."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return {}
    return wrapper


# ── Algorithm definitions ────────────────────────────────────────────────────
# To add a new algorithm: apply @algorithm(...), @_safe, @_with_seed in that
# order and implement (G) → partition dict.

@algorithm('Louvain', needs_conversion=False, parallel=True)
@_safe
@_with_seed
def louvain_algorithm(G):
    return community_louvain.best_partition(G)

@algorithm('Leiden', needs_conversion=False, parallel=True)
@_safe
@_with_seed
def leiden_algorithm(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ig_graph = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    partition = ig_graph.community_leiden(objective_function='modularity')
    return {nodes[i]: partition.membership[i] for i in range(ig_graph.vcount())}

@algorithm('HPMOCD', needs_conversion=False, parallel=False)
@_safe
@_with_seed
def hpmocd_algorithm(G):
    return pymocd.HpMocd(G, debug_level=0).run()


# ── Experiment runner ────────────────────────────────────────────────────────

class ExperimentRunner:
    def __init__(self, n_runs: int = NUM_RUNS):
        self.n_runs = n_runs

    def _run_single(self, args: tuple) -> Dict[str, Any]:
        alg_name, alg_func, needs_conversion, n_runs, param_name, param_value, fixed_param = args
        metrics = {'modularity': [], 'nmi': [], 'ami': [], 'time': []}

        for run_id in range(n_runs):
            if param_name == 'mu':
                G, ground_truth = generate_lfr_benchmark(n=fixed_param, mu=param_value, seed=run_id)
            else:
                G, ground_truth = generate_lfr_benchmark(n=param_value, mu=fixed_param, seed=run_id)

            start = time.time()
            communities = alg_func(G, seed=run_id)
            duration = time.time() - start

            eval_result = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
            metrics['modularity'].append(eval_result['modularity'])
            metrics['nmi'].append(eval_result['nmi'])
            metrics['ami'].append(eval_result['ami'])
            metrics['time'].append(duration)

        result = {'algorithm': alg_name, param_name: param_value}
        for m in metrics:
            result[f'{m}_mean'] = np.mean(metrics[m])
            result[f'{m}_std'] = np.std(metrics[m], ddof=min(1, len(metrics[m]) - 1))

        # Incremental backup after each run
        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
            row = pd.DataFrame([result]).rename(columns={
                'modularity_mean': 'modularity', 'nmi_mean': 'nmi',
                'ami_mean': 'ami', 'time_mean': 'time'
            })
            row.to_csv(BACKUP_CSV_FILE, mode='a', header=not os.path.exists(BACKUP_CSV_FILE), index=False)
        except Exception as e:
            print(f"[Backup ERROR] {e}")

        return result

    def _build_args(self, param_name, param_values, fixed_param) -> tuple:
        """Split args into parallel and sequential based on algorithm registry."""
        parallel_args, sequential_args = [], []
        for name, info in ALGORITHM_REGISTRY.items():
            for val in param_values:
                entry = (name, info['function'], info['needs_conversion'],
                         self.n_runs, param_name, val, fixed_param)
                (parallel_args if info['parallel'] else sequential_args).append(entry)
        return parallel_args, sequential_args

    def _run(self, param_name, param_values, fixed_param) -> pd.DataFrame:
        parallel_args, sequential_args = self._build_args(param_name, param_values, fixed_param)

        results = []
        if parallel_args:
            with Pool() as pool:
                results += pool.map(self._run_single, parallel_args)
        results += [self._run_single(a) for a in sequential_args]

        df = pd.DataFrame(results).rename(columns={
            'modularity_mean': 'modularity', 'nmi_mean': 'nmi',
            'ami_mean': 'ami', 'time_mean': 'time',
        })
        os.makedirs(SAVE_PATH, exist_ok=True)
        df.to_csv(CSV_FILE, index=False)
        return df

    def run_mu_experiment(self, mus=None, n_nodes=250) -> pd.DataFrame:
        if mus is None:
            mus = np.arange(MIN_MU, MAX_MU + STEP_MU, STEP_MU)
        return self._run('mu', mus, n_nodes)

    def run_nodes_experiment(self, n_list=None, mu=0.3) -> pd.DataFrame:
        if n_list is None:
            n_list = np.arange(500, 5500, 500)
        return self._run('nodes', n_list, mu)


def main():
    print(f"Registered algorithms: {list(ALGORITHM_REGISTRY.keys())}")

    if JUST_PLOT:
        results = read_results_from_csv(CSV_FILE)
    else:
        runner = ExperimentRunner(n_runs=NUM_RUNS)
        results = runner.run_mu_experiment() if MU_EXPERIMENT else runner.run_nodes_experiment()

    plot_results(results)

if __name__ == "__main__":
    main()
