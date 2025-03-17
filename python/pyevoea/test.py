import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
import community as community_louvain
import time
from tqdm import tqdm
import pandas as pd

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

def generate_lfr_benchmark(n=1000, tau1=2.5, tau2=1.5, mu=0.1, average_degree=20, 
                           min_community=20, seed=0):
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree, 
            min_community=min_community, max_degree=50, seed=seed, max_community=100
        )        
        communities = {node: frozenset(G.nodes[node]['community']) for node in G}        
        G = nx.Graph(G)  # Convert to simple graph (remove metadata)
        return G, communities
        
    except AttributeError:
        print("NetworkX LFR implementation not available. Please install networkx extra packages.")
        raise

def convert_communities_to_partition(communities):
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition

def evaluate_communities(G, detected_communities, ground_truth_communities, convert=True):
    if convert:
        detected_partition = convert_communities_to_partition(detected_communities)
    else:
        detected_partition = detected_communities

    ground_truth_partition = {}
    for node, comms in ground_truth_communities.items():
        ground_truth_partition[node] = list(comms)[0] if isinstance(comms, frozenset) else comms
    
    communities_as_list = []
    max_community = max(detected_partition.values())
    for i in range(max_community + 1):
        community = {node for node, comm in detected_partition.items() if comm == i}
        if community:
            communities_as_list.append(community)
    
    mod = modularity(G, communities_as_list)    
    n_nodes = len(G.nodes())
    gt_labels = np.zeros(n_nodes, dtype=np.int32)
    detected_labels = np.zeros(n_nodes, dtype=np.int32)
    
    for i, node in enumerate(sorted(G.nodes())):
        gt_labels[i] = ground_truth_partition[node]
        detected_labels[i] = detected_partition[node]
    
    nmi = normalized_mutual_info_score(gt_labels, detected_labels)
    ami = adjusted_mutual_info_score(gt_labels, detected_labels)
    
    return {
        'modularity': mod,
        'nmi': nmi,
        'ami': ami
    }

# ======================================================================
# Experiment
# ======================================================================

def run_experiment(algorithms=None, mus=np.arange(0.1, 0.8, 0.1), n_runs=5, n_nodes=5000):
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
        
        # Save results incrementally
        df = pd.DataFrame(results)
        df.to_csv('community_detection_results.csv', index=False)
    
    return results

# https://grok.com/share/bGVnYWN5_8acc0ffd-6bd7-4570-b7e7-a7d5df652aaa // why mu = 0.3
def run_nodes_experiment(algorithms=None, n_list=np.arange(32000, 201000, 1000), n_runs=20, mu=0.3):
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
                print(f"{alg_name} {n}: Q = {eval_results['modularity']}, NMI/AMI: {eval_results['nmi']}/{eval_results['ami']}")
                
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
        
        # Save results incrementally
        df = pd.DataFrame(results)
        df.to_csv('community_detection_results.csv', index=False)
    
    return results

# ======================================================================
# Helpers
# ======================================================================

def plot_results(results):
    df = pd.DataFrame(results)
    algorithms = df['algorithm'].unique()
    if 'mu' in df.columns:
        x_var = 'mu'
        x_label = 'Mixing Parameter (μ)'
    elif 'nodes' in df.columns:
        x_var = 'nodes'
        x_label = 'Number of nodes (n)'
    else:
        raise ValueError("Neither 'mu' nor 'nodes' found in results")
    has_std_data = all(col in df.columns for col in ['nmi_std', 'ami_std', 'modularity_std', 'time_std'])
    metrics = [
        {'key': 'nmi', 'title': 'Normalized Mutual Information', 'ylabel': 'NMI'},
        {'key': 'ami', 'title': 'Adjusted Mutual Information', 'ylabel': 'AMI'},
        {'key': 'modularity', 'title': 'Modularity', 'ylabel': 'Modularity'},
        {'key': 'time', 'title': 'Computation Time', 'ylabel': 'Time (s)'}
    ]
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, start=1):
        plt.subplot(2, 2, i)
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg].sort_values(by=x_var)
            x_values = alg_data[x_var].values
            y_values = alg_data[metric['key']].values
            if has_std_data:
                std_key = metric['key'] + '_std'
                if std_key in alg_data.columns:
                    y_std = alg_data[std_key].values
                    plt.errorbar(x_values, y_values, yerr=y_std, fmt='o-', label=alg, capsize=5)
                else:
                    plt.plot(x_values, y_values, 'o-', label=alg)
            else:
                plt.plot(x_values, y_values, 'o-', label=alg)
        plt.xlabel(x_label)
        plt.ylabel(metric['ylabel'])
        plt.title(metric['title'])
        plt.legend()
        plt.grid(True)
    plt.suptitle("Community Detection Algorithms Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('community_detection_results.png', dpi=300)
    plt.show()

def save_results_to_csv(results, filename='community_detection_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def print_results_table(results):
    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df.to_string(index=False))

# ======================================================================
# Algorithms registration
# ======================================================================

def pymocd(G, seed=None):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from mocd import mocd
    return mocd(G)

def louvain_wrapper(G, seed=None):
    return louvain_communities(G, seed=seed)

def pyevoea_hpmocd_wrapper(G, seed=None):
    import pyevoea
    if seed is not None:
        np.random.seed(seed)
    return pyevoea.HpMocd(G).run()

def leiden_wrapper(G, seed=None):
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph(edges=list(G.edges()), directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    communities = [set(cluster) for cluster in partition]
    return communities

# ======================================================================
# Register
# ======================================================================

#register_algorithm('MOCD', pymocd, needs_conversion=False)
register_algorithm('Louvain', louvain_wrapper, needs_conversion=True)
register_algorithm('HpMocd', pyevoea_hpmocd_wrapper, needs_conversion=False)
register_algorithm('Leiden', leiden_wrapper, needs_conversion=True)

# ======================================================================
# Plotting and saving results
# ======================================================================

def read_results_from_csv(filename='community_detection_results.csv'):
    try:
        df = pd.read_csv(filename)
        # Base keys common to both experiments
        results = {
            'algorithm': [],
            'modularity': [],
            'nmi': [],
            'ami': [],
            'time': []
        }
        # Dynamically add 'mu' or 'nodes' based on CSV content
        if 'mu' in df.columns:
            results['mu'] = []
        elif 'nodes' in df.columns:
            results['nodes'] = []
        else:
            raise ValueError("Neither 'mu' nor 'nodes' found in CSV")
        # Add standard deviation columns if present
        std_columns = ['modularity_std', 'nmi_std', 'ami_std', 'time_std']
        for col in std_columns:
            if col in df.columns:
                results[col] = []
        # Populate lists only for keys present in both results and CSV
        for col in results.keys():
            if col in df.columns:
                results[col] = df[col].tolist()
        print(f"Successfully read results from {filename}")
        has_std = all(col in results for col in std_columns)
        if has_std:
            print("Standard deviation data found - confidence intervals will be shown")
        else:
            print("No standard deviation data found - confidence intervals will not be shown")
        return results
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None

if __name__ == "__main__":
    mus = np.arange(0.1, 0.7, 0.1)  # 0.1, ..., 0.7
    
    print(f"Running community detection algorithms on LFR benchmark graphs with 1000 nodes")
    print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")

    # Run experiments
    #results = read_results_from_csv('results.csv')
    #results = run_experiment(mus=mus, n_runs=10)
    results = run_nodes_experiment(n_runs=20)

    print("\nResults:")
    plot_results(results)    
    save_results_to_csv(results)