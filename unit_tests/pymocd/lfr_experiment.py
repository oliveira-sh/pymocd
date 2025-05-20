from networkx.algorithms.community import louvain_communities
import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

from utils import (
    generate_lfr_benchmark,
    evaluate_communities,
    plot_results,
    SAVE_PATH
)

CSV_FILE_PATH = 'lfr_experiment.csv'
MIN_MU = 0.1
MAX_MU = 0.8
STEP_MU = 0.1
NUM_RUNS = 10
JUST_PLOT_AVAILABLE_RESULTS = False
MU_EXPERIMENT = True                    # false to run scalling experiment

# ======================================================================
# Helpers
# ======================================================================

ALGORITHM_REGISTRY = {}

def register_algorithm(name, func, needs_conversion=True):
    ALGORITHM_REGISTRY[name] = {
        'function': func,
        'needs_conversion': needs_conversion
    }
    print(f"Registered algorithm: {name}")

# ======================================================================
# Experiment
# ======================================================================

def run_experiment(algorithms=None, mus=np.arange(MIN_MU, MAX_MU, STEP_MU), n_runs=NUM_RUNS, n_nodes=100000):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    
    results = {
        'algorithm': [],
        'mu': [],
        'modularity': [],
        'nmi': [],
        'ami': [],
        'time': [],
        'modularity_std': [],
        'nmi_std': [],
        'ami_std': [],
        'time_std': []
    }
    
    for mu in tqdm(mus, desc="Processing mu values"):
        for alg_name in algorithms:
            alg_info = ALGORITHM_REGISTRY[alg_name]
            alg_func = alg_info['function']
            needs_conversion = alg_info['needs_conversion']
            
            mod_values = []
            nmi_values = []
            ami_values = []
            time_values = []
            
            for run in range(n_runs):
                seed = run
                G, ground_truth = generate_lfr_benchmark(n=n_nodes, mu=mu, seed=seed)
                start_time = time.time()
                communities = alg_func(G, seed=seed)
                end_time = time.time()
                eval_results = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
                
                mod_values.append(eval_results['modularity'])
                nmi_values.append(eval_results['nmi'])
                ami_values.append(eval_results['ami'])
                time_values.append(end_time - start_time)
                print(f"{alg_name} {mu}: Q = {eval_results['modularity']}, NMI/AMI: {eval_results['nmi']}/{eval_results['ami']}")
                
            results['algorithm'].append(alg_name)
            results['mu'].append(mu)
            results['modularity'].append(np.mean(mod_values))
            results['nmi'].append(np.mean(nmi_values))
            results['ami'].append(np.mean(ami_values))
            results['time'].append(np.mean(time_values))
            results['modularity_std'].append(np.std(mod_values, ddof=1))
            results['nmi_std'].append(np.std(nmi_values, ddof=1))
            results['ami_std'].append(np.std(ami_values, ddof=1))
            results['time_std'].append(np.std(time_values, ddof=1))        
        df = pd.DataFrame(results)
        df.to_csv(f'{SAVE_PATH}{CSV_FILE_PATH}', index=False)
    
    return results

def run_nodes_experiment(algorithms=None, n_list=np.arange(10000, 110000, 10000), n_runs=NUM_RUNS, mu=0.3):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    
    results = {
        'algorithm': [],
        'nodes': [],
        'modularity': [],
        'nmi': [],
        'ami': [],
        'time': [],
        'modularity_std': [],
        'nmi_std': [],
        'ami_std': [],
        'time_std': []
    }
    
    for n in tqdm(n_list, desc="Processing nodes values"):
        for alg_name in algorithms:
            alg_info = ALGORITHM_REGISTRY[alg_name]
            alg_func = alg_info['function']
            needs_conversion = alg_info['needs_conversion']
            
            mod_values = []
            nmi_values = []
            ami_values = []
            time_values = []
            
            for run in range(n_runs):
                seed = run
                G, ground_truth = generate_lfr_benchmark(n=n, mu=mu, seed=seed)
                start_time = time.time()
                communities = alg_func(G, seed=seed)
                end_time = time.time()
                eval_results = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
                
                mod_values.append(eval_results['modularity'])
                nmi_values.append(eval_results['nmi'])
                ami_values.append(eval_results['ami'])
                time_values.append(end_time - start_time)
                
            results['algorithm'].append(alg_name)
            results['nodes'].append(n)
            results['modularity'].append(np.mean(mod_values))
            results['nmi'].append(np.mean(nmi_values))
            results['ami'].append(np.mean(ami_values))
            results['time'].append(np.mean(time_values))
            results['modularity_std'].append(np.std(mod_values, ddof=1))
            results['nmi_std'].append(np.std(nmi_values, ddof=1))
            results['ami_std'].append(np.std(ami_values, ddof=1))
            results['time_std'].append(np.std(time_values, ddof=1))

            print(", ".join(str(results[key][-1]) for key in results))
        df = pd.DataFrame(results)
        df.to_csv(f'{SAVE_PATH}{CSV_FILE_PATH}', index=False)
    
    return results

# ======================================================================
# Algorithms registration
# ======================================================================

def mocd_wrapper(G, seed=None):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mocd import mocd
    return mocd(G)

def louvain_wrapper(G, seed=None):
    return louvain_communities(G, seed=seed)

def hpmocd_wrapper(G, seed=None):
    import pymocd
    if seed is not None:
        np.random.seed(seed)
    return pymocd.HpMocd(G, debug_level=1).run()

def leiden_wrapper(G, seed=None):
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph(edges=list(G.edges()), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    communities = [set(cluster) for cluster in partition]
    return communities

def girvan_newman_wrapper(G, seed=None):
    return nx.community.girvan_newman(G)

def moganet_wrapper(G, seed=None):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mogaNet import detect_communities_ga
    return detect_communities_ga(G,
                                  pop_size=50,
                                  generations=30,
                                  crossover_rate=0.8,
                                  mutation_rate=0.2,
                                  r=1.5,
                                  elite_ratio=0.1)

# ======================================================================
# Register
# ======================================================================

register_algorithm('HPMOCD', hpmocd_wrapper, needs_conversion=False)
register_algorithm('Louvain', louvain_wrapper, needs_conversion=True)
register_algorithm('Leiden', leiden_wrapper, needs_conversion=True)
#register_algorithm('MOCD', mocd_wrapper, needs_conversion=False)
#register_algorithm("MogaNet", moganet_wrapper, needs_conversion=False)

if __name__ == "__main__":
    print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    results = None

    if JUST_PLOT_AVAILABLE_RESULTS:
        results = read_results_from_csv('community_detection_results.csv')
    else: # Run experiments
        if MU_EXPERIMENT:
            results = run_experiment(mus=np.arange(MIN_MU, MAX_MU + 0.1, 0.1), n_runs=10)
        else:
            results = run_nodes_experiment(n_runs=20)

    plot_results(results)    