import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool
from functools import wraps
from typing import Callable, Dict, Any, List, Set, Union

import community as community_louvain
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import girvan_newman
from nsga_iii import run_krm, run_ccm
from cdrme.gen import run_cdrme
import pymocd

from utils import (
    generate_lfr_benchmark,
    evaluate_communities,
    plot_results,
    read_results_from_csv,
    SAVE_PATH
)

CSV_FILE_PATH = 'lfr.csv'
MIN_MU = 0.1
MAX_MU = 0.8
STEP_MU = 0.1
NUM_RUNS = 10
JUST_PLOT_AVAILABLE_RESULTS = False
MU_EXPERIMENT = False
BACKUP_CSV_FILE_PATH = os.path.join(SAVE_PATH, CSV_FILE_PATH.replace('.csv', '_bk.csv'))

ALGORITHM_REGISTRY = {}

def algorithm(name: str, needs_conversion: bool = True, parallel: bool = True):
    """Decorator to register community detection algorithms"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        ALGORITHM_REGISTRY[name] = {
            'function': wrapper,
            'needs_conversion': needs_conversion,
            'parallel': parallel
        }
        print(f"Registered algorithm: {name} (parallel={parallel})")
        return wrapper
    return decorator

def seed_handler(func: Callable):
    """Decorator to handle seed parameter for reproducibility"""
    @wraps(func)
    def wrapper(G, seed=None, *args, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return func(G, *args, **kwargs)
    return wrapper

def timing_decorator(func: Callable):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrapper

def error_handler(func: Callable):
    """Decorator to handle exceptions in algorithm execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return []
    return wrapper

@algorithm('Louvain', needs_conversion=True, parallel=True)
@error_handler
@seed_handler
def louvain_algorithm(G):
    return louvain_communities(G)

@algorithm('Leiden', needs_conversion=True, parallel=True)
@error_handler
@seed_handler
def leiden_algorithm(G):
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph(edges=list(G.edges()), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    return [set(c) for c in partition]

@algorithm('ASYN-LPA', needs_conversion=True, parallel=True)
@error_handler
@seed_handler
def asynlpa_algorithm(G):
    return list(asyn_lpa_communities(G))

@algorithm('HPMOCD', needs_conversion=False, parallel=False)
@error_handler
@seed_handler
def hpmocd_algorithm(G):
    return pymocd.HpMocd(G, debug_level=0).run()

@algorithm('NSGA III KRM', needs_conversion=False, parallel=True)
@error_handler
@seed_handler
def krm_algorithm(G):
    return run_krm(G)

@algorithm('NSGA III CCM', needs_conversion=False, parallel=True)
@error_handler
@seed_handler
def ccm_algorithm(G):
    return run_ccm(G)

@algorithm('CDRME', needs_conversion=True, parallel=True)
@error_handler
@seed_handler
def cdrme_algorithm(G):
    return run_cdrme(G)

@algorithm('MOCD', needs_conversion=False, parallel=True)
@error_handler
@seed_handler
def mocd_algorithm(G):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mocd import mocd
    return mocd(G)

@algorithm('MogaNet', needs_conversion=False, parallel=True)
@error_handler
@seed_handler
def moganet_algorithm(G):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from moganet import detect_communities_ga
    return detect_communities_ga(G,
                                pop_size=100,
                                generations=100,
                                crossover_rate=0.9,
                                mutation_rate=0.1,
                                r=1.5,
                                elite_ratio=0.1)

class ExperimentRunner:
    def __init__(self, n_runs: int = NUM_RUNS):
        self.n_runs = n_runs
    
    def _run_single_experiment(self, args: tuple) -> Dict[str, Any]:
        alg_name, alg_func, needs_conversion, n_runs, param_name, param_value, fixed_param = args
        metrics = {'modularity': [], 'nmi': [], 'ami': [], 'time': []}
        
        for run_id in range(n_runs):
            if param_name == 'mu':
                G, ground_truth = generate_lfr_benchmark(n=fixed_param, mu=param_value, seed=run_id)
            else:  # nodes experiment
                G, ground_truth = generate_lfr_benchmark(n=param_value, mu=fixed_param, seed=run_id)
            
            start = time.time()
            communities = alg_func(G, seed=run_id)
            duration = time.time() - start
            
            eval_result = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
            
            metrics['modularity'].append(eval_result['modularity'])
            metrics['nmi'].append(eval_result['nmi'])
            metrics['ami'].append(eval_result['ami'])
            metrics['time'].append(duration)
        
        result = {
            'algorithm': alg_name,
            param_name: param_value,
        }
        
        for metric in metrics:
            result[f'{metric}_mean'] = np.mean(metrics[metric])
            result[f'{metric}_std'] = np.std(metrics[metric], ddof=1)
        
        # --- incremental backup: append this single-result row to *_bk.csv ---
        try:
            # only write header if file doesn't yet exist
            write_header = not os.path.exists(BACKUP_CSV_FILE_PATH)
            row_df = pd.DataFrame([result])
            # rename mean‐columns to match final output
            row_df.rename(columns={
                'modularity_mean': 'modularity',
                'nmi_mean'      : 'nmi',
                'ami_mean'      : 'ami',
                'time_mean'     : 'time'
            }, inplace=True)
            row_df.to_csv(BACKUP_CSV_FILE_PATH,
                          mode='a',
                          header=write_header,
                          index=False)
        except Exception as e:
            print(f"[Backup ERROR] could not write to {BACKUP_CSV_FILE_PATH}: {e}")

        return result
    
    def _prepare_experiment_args(self, algorithms: List[str], param_name: str, 
                               param_values: np.ndarray, fixed_param: float) -> List[tuple]:
        args_list = []
        for alg_name in algorithms:
            info = ALGORITHM_REGISTRY[alg_name]
            for param_value in param_values:
                args_list.append((
                    alg_name, info['function'], info['needs_conversion'],
                    self.n_runs, param_name, param_value, fixed_param
                ))
        return args_list
    
    def run_mu_experiment(self, algorithms: List[str] = None, 
                         mus: np.ndarray = None, n_nodes: int = 250) -> pd.DataFrame:
        if algorithms is None:
            algorithms = list(ALGORITHM_REGISTRY.keys())
        if mus is None:
            mus = np.arange(MIN_MU, MAX_MU + STEP_MU, STEP_MU)
        
        args_list = self._prepare_experiment_args(algorithms, 'mu', mus, n_nodes)
        
        with Pool() as pool:
            results = pool.map(self._run_single_experiment, args_list)
        
        return self._format_results(results)
    
    def run_nodes_experiment(self, algorithms: List[str] = None,
                           n_list: np.ndarray = None, mu: float = 0.3) -> pd.DataFrame:
        if algorithms is None:
            algorithms = list(ALGORITHM_REGISTRY.keys())
        if n_list is None:
            n_list = np.arange(500, 5500, 500)
        
        args_list = self._prepare_experiment_args(algorithms, 'nodes', n_list, mu)
        
        with Pool() as pool:
            results = pool.map(self._run_single_experiment, args_list)
        
        return self._format_results(results)
    
    def _format_results(self, results: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(results)
        df.rename(columns={
            'modularity_mean': 'modularity',
            'nmi_mean': 'nmi', 
            'ami_mean': 'ami',
            'time_mean': 'time',
            'modularity_std': 'modularity_std',
            'nmi_std': 'nmi_std',
            'ami_std': 'ami_std',
            'time_std': 'time_std'
        }, inplace=True)
        
        df.to_csv(f'{SAVE_PATH}{CSV_FILE_PATH}', index=False)
        return df

def main():
    print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    
    if JUST_PLOT_AVAILABLE_RESULTS:
        results = read_results_from_csv(SAVE_PATH + CSV_FILE_PATH)
    else:
        runner = ExperimentRunner(n_runs=NUM_RUNS)
        
        if MU_EXPERIMENT:
            results = runner.run_mu_experiment(
                mus=np.arange(MIN_MU, MAX_MU + STEP_MU, STEP_MU)
            )
        else:
            results = runner.run_nodes_experiment()
    
    plot_results(results)

if __name__ == "__main__":
    main()