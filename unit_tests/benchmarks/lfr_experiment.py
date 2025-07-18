import sys, os

# ======================================================================
# ALGORITHMS
import community as community_louvain
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import girvan_newman
from nsga_iii import run_krm, run_ccm
from cdrme.gen import run_cdrme
# ======================================================================

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

from utils import (
    generate_lfr_benchmark,
    evaluate_communities,
    plot_results,
    read_results_from_csv,
    SAVE_PATH
)

CSV_FILE_PATH = 'lfr_experiment.csv'
MIN_MU = 0.1
MAX_MU = 0.3
STEP_MU = 0.1
NUM_RUNS = 4

# Set this to true if you already has the csv (with the name = CSV_FILE_PATH)
# and just want to plot the comparasions
JUST_PLOT_AVAILABLE_RESULTS = False

# true: growing mu parameter experiment
# false; network growing size experiment (nodes only)
MU_EXPERIMENT = True  

# ======================================================================
# Registry and Helpers
# ======================================================================
ALGORITHM_REGISTRY = {}

def register_algorithm(name, func, needs_conversion=True, parallel=False):
    ALGORITHM_REGISTRY[name] = {
        'function': func,
        'needs_conversion': needs_conversion,
        'parallel': parallel
    }
    print(f"Registered algorithm: {name} (parallel={parallel})")

# Top-level worker for mu experiments
def _process_mu(args):
    alg_name, alg_func, needs_conversion, n_runs, mu, n_nodes = args
    mod_vals, nmi_vals, ami_vals, time_vals = [], [], [], []
    for run_id in range(n_runs):
        G, ground_truth = generate_lfr_benchmark(n=n_nodes, mu=mu, seed=run_id)
        start = time.time()
        communities = alg_func(G, seed=run_id)
        duration = time.time() - start
        eval_r = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
        mod_vals.append(eval_r['modularity'])
        nmi_vals.append(eval_r['nmi'])
        ami_vals.append(eval_r['ami'])
        time_vals.append(duration)
    return {
        'algorithm': alg_name,
        'mu': mu,
        'mod_mean': np.mean(mod_vals),
        'nmi_mean': np.mean(nmi_vals),
        'ami_mean': np.mean(ami_vals),
        'time_mean': np.mean(time_vals),
        'mod_std': np.std(mod_vals, ddof=1),
        'nmi_std': np.std(nmi_vals, ddof=1),
        'ami_std': np.std(ami_vals, ddof=1),
        'time_std': np.std(time_vals, ddof=1)
    }

# Top-level worker for node-size experiments
def _process_n(args):
    alg_name, alg_func, needs_conversion, n_runs, nodes, mu = args
    mod_vals, nmi_vals, ami_vals, time_vals = [], [], [], []
    for run_id in range(n_runs):
        G, ground_truth = generate_lfr_benchmark(n=nodes, mu=mu, seed=run_id)
        start = time.time()
        communities = alg_func(G, seed=run_id)
        duration = time.time() - start
        eval_r = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
        mod_vals.append(eval_r['modularity'])
        nmi_vals.append(eval_r['nmi'])
        ami_vals.append(eval_r['ami'])
        time_vals.append(duration)
    return {
        'algorithm': alg_name,
        'nodes': nodes,
        'mod_mean': np.mean(mod_vals),
        'nmi_mean': np.mean(nmi_vals),
        'ami_mean': np.mean(ami_vals),
        'time_mean': np.mean(time_vals),
        'mod_std': np.std(mod_vals, ddof=1),
        'nmi_std': np.std(nmi_vals, ddof=1),
        'ami_std': np.std(ami_vals, ddof=1),
        'time_std': np.std(time_vals, ddof=1)
    }

# ======================================================================
# Experiments
# ======================================================================

def run_experiment(algorithms=None,
                   mus=np.arange(MIN_MU, MAX_MU + STEP_MU, STEP_MU),
                   n_runs=NUM_RUNS,
                   n_nodes=250):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    all_results = []

    for alg_name in algorithms:
        info = ALGORITHM_REGISTRY[alg_name]
        func = info['function']
        needs_conversion = info['needs_conversion']
        parallel_flag = info['parallel']

        if parallel_flag:
            args = [(alg_name, func, needs_conversion, n_runs, mu, n_nodes) for mu in mus]
            with Pool() as pool:
                summary = pool.map(_process_mu, args)
        else:
            summary = []
            for mu in tqdm(mus, desc=f"{alg_name} (sequential µ loop)"):
                summary.append(_process_mu((alg_name, func, needs_conversion, n_runs, mu, n_nodes)))

        for entry in summary:
            all_results.append(entry)
            print(f"{entry['algorithm']} µ={entry['mu']}: Q={entry['mod_mean']:.4f}, "
                  f"NMI={entry['nmi_mean']:.4f}, AMI={entry['ami_mean']:.4f}")

    df = pd.DataFrame(all_results)
    df.rename(columns={
        'mod_mean': 'modularity', 'nmi_mean': 'nmi', 'ami_mean': 'ami', 'time_mean': 'time',
        'mod_std': 'modularity_std', 'nmi_std': 'nmi_std', 'ami_std': 'ami_std', 'time_std': 'time_std'
    }, inplace=True)
    df.to_csv(f'{SAVE_PATH}{CSV_FILE_PATH}', index=False)
    return df


def run_nodes_experiment(algorithms=None,
                         n_list=np.arange(10000, 110000, 10000),
                         n_runs=NUM_RUNS,
                         mu=0.3):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    all_results = []

    for alg_name in algorithms:
        info = ALGORITHM_REGISTRY[alg_name]
        func = info['function']
        needs_conversion = info['needs_conversion']
        parallel_flag = info['parallel']

        if parallel_flag:
            args = [(alg_name, func, needs_conversion, n_runs, n, mu) for n in n_list]
            with Pool() as pool:
                summary = pool.map(_process_n, args)
        else:
            summary = []
            for n in tqdm(n_list, desc=f"{alg_name} (sequential n loop)"):
                summary.append(_process_n((alg_name, func, needs_conversion, n_runs, n, mu)))

        for entry in summary:
            all_results.append(entry)
            print(f"{entry['algorithm']} n={entry['nodes']}: Q={entry['mod_mean']:.4f}, "
                  f"NMI={entry['nmi_mean']:.4f}, AMI={entry['ami_mean']:.4f}")

    df = pd.DataFrame(all_results)
    df.rename(columns={
        'mod_mean': 'modularity', 'nmi_mean': 'nmi', 'ami_mean': 'ami', 'time_mean': 'time',
        'mod_std': 'modularity_std', 'nmi_std': 'nmi_std', 'ami_std': 'ami_std', 'time_std': 'time_std'
    }, inplace=True)
    df.to_csv(f'{SAVE_PATH}{CSV_FILE_PATH}', index=False)
    return df

# ======================================================================
# Algorithm Wrappers and Registration
# ======================================================================

def mocd_w(G, seed=None):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mocd import mocd
    return mocd(G)

def louvain_w(G, seed=None):
    return louvain_communities(G, seed=seed)

def hpmocd_w(G, seed=None):
    import pymocd
    if seed is not None:
        np.random.seed(seed)
    return pymocd.HpMocd(G, debug_level=0).run()

def leiden_w(G, seed=None):
    import igraph as ig, leidenalg
    G_ig = ig.Graph(edges=list(G.edges()), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    return [set(c) for c in partition]

def moganet_w(G, seed=None):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mogaNet import detect_communities_ga
    return detect_communities_ga(G,
                                  pop_size=100,
                                  generations=100,
                                  crossover_rate=0.9,
                                  mutation_rate=0.1,
                                  r=1.5,
                                  elite_ratio=0.1)

def newman_w(G, seed=None):
    # girvan_newman yields successive splits; take the first one
    comp_gen = girvan_newman(G)
    first_split = next(comp_gen)
    # wrap each community in a set and return as list
    return [set(c) for c in first_split]

def asynlpr_w(G, seed=None):
    lpa_communities = list(asyn_lpa_communities(G))
    return lpa_communities

def krm_w(G, seed=None):
    return run_krm(G)

def ccm_w(G, seed=None):
    return run_ccm(G)

def cdrme_w(G, seed=None):
    return run_cdrme(G)

# ======================================================================
# use parallel always as false, has built-in features. 
register_algorithm('NSGA III KRM', krm_w, needs_conversion=False, parallel=True )
register_algorithm('NSGA III CCM', ccm_w, needs_conversion=False, parallel=True )
register_algorithm('CDRME', cdrme_w, needs_conversion=True, parallel=True )
register_algorithm('HPMOCD', hpmocd_w, needs_conversion=False, parallel=False )
register_algorithm('Louvain', louvain_w, needs_conversion=True, parallel=True )
register_algorithm('Leiden', leiden_w, needs_conversion=True, parallel=True )
register_algorithm('ASYN-LPA', asynlpr_w, needs_conversion=True, parallel=True )


# ======================================================================
# We removed the MOCD, MogaNet and Girvan Newman due to the fast execution time being unfeasible,
# reaching over 25 hours for a single run. 
# These two algorithms will not be available in any form in the pymocd source code. 
# This is because, since the algorithms are from other authors, it is unethical to make them
# available without proper permission. 
# The wrappers are still here, in case you want to reimplement them from scratch.

#register_algorithm('MOCD', mocd_w, needs_conversion=False, parallel=True)
#register_algorithm('MogaNet', moganet_w, needs_conversion=False, parallel=True)
#register_algorithm('Girvan Newman', newman_w, needs_conversion=True, parallel=True)

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    if JUST_PLOT_AVAILABLE_RESULTS:
        results = read_results_from_csv(SAVE_PATH + CSV_FILE_PATH)
    else:
        if MU_EXPERIMENT:
            results = run_experiment(mus=np.arange(MIN_MU, MAX_MU + STEP_MU, STEP_MU), n_runs=NUM_RUNS)
        else:
            results = run_nodes_experiment(n_runs=NUM_RUNS)
    plot_results(results)