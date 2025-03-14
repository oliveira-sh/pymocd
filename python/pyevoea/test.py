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

def generate_lfr_benchmark(n=1000, tau1=3, tau2=1.5, mu=0.1, average_degree=20, 
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

    # Extract ground truth partition
    ground_truth_partition = {}
    for node, comms in ground_truth_communities.items():
        ground_truth_partition[node] = list(comms)[0] if isinstance(comms, frozenset) else comms
    
    # For modularity calculation, we need communities as a list of sets
    communities_as_list = []
    max_community = max(detected_partition.values())
    for i in range(max_community + 1):
        community = {node for node, comm in detected_partition.items() if comm == i}
        if community:  # Only add non-empty communities
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

def run_experiment(algorithms=None, mus=np.arange(0.1, 0.8, 0.1), n_runs=5, n_nodes=400):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    
    results = {
        'algorithm': [],
        'mu': [],
        'modularity': [],
        'nmi': [],
        'ami': [],
        'time': [],
        # Add standard deviation columns
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
                # Generate LFR benchmark graph
                seed = run  # Different seed for each run
                G, ground_truth = generate_lfr_benchmark(n=n_nodes, mu=mu, seed=seed)
                
                # Run algorithm
                start_time = time.time()
                communities = alg_func(G, seed=seed)
                end_time = time.time()
                
                # Evaluate communities
                eval_results = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
                
                mod_values.append(eval_results['modularity'])
                nmi_values.append(eval_results['nmi'])
                ami_values.append(eval_results['ami'])
                time_values.append(end_time - start_time)
                print(f"{alg_name} {mu}: Q = {eval_results['modularity']}, NMI/AMI: {eval_results['nmi']}/{eval_results['ami']}")
                
            # Store average results
            results['algorithm'].append(alg_name)
            results['mu'].append(mu)
            results['modularity'].append(np.mean(mod_values))
            results['nmi'].append(np.mean(nmi_values))
            results['ami'].append(np.mean(ami_values))
            results['time'].append(np.mean(time_values))
            
            # Store standard deviations
            results['modularity_std'].append(np.std(mod_values, ddof=1))
            results['nmi_std'].append(np.std(nmi_values, ddof=1))
            results['ami_std'].append(np.std(ami_values, ddof=1))
            results['time_std'].append(np.std(time_values, ddof=1))
            
    return results

def run_nodes_experiment(algorithms=None, n_list=np.arange(10000, 110000, 5000), n_runs=5, mu=0.3):
    if algorithms is None:
        algorithms = list(ALGORITHM_REGISTRY.keys())
    
    results = {
        'algorithm': [],
        'nodes': [],
        'modularity': [],
        'nmi': [],
        'ami': [],
        'time': [],
        # Add standard deviation columns
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
                # Generate LFR benchmark graph
                seed = run  # Different seed for each run
                G, ground_truth = generate_lfr_benchmark(n=n, mu=mu, seed=seed)
                
                # Run algorithm
                start_time = time.time()
                communities = alg_func(G, seed=seed)
                end_time = time.time()
                
                # Evaluate communities
                eval_results = evaluate_communities(G, communities, ground_truth, convert=needs_conversion)
                
                mod_values.append(eval_results['modularity'])
                nmi_values.append(eval_results['nmi'])
                ami_values.append(eval_results['ami'])
                time_values.append(end_time - start_time)
                print(f"{alg_name} {n}: Q = {eval_results['modularity']}, NMI/AMI: {eval_results['nmi']}/{eval_results['ami']}")
                
            # Store average results
            results['algorithm'].append(alg_name)
            results['nodes'].append(n)
            results['modularity'].append(np.mean(mod_values))
            results['nmi'].append(np.mean(nmi_values))
            results['ami'].append(np.mean(ami_values))
            results['time'].append(np.mean(time_values))
            
            # Store standard deviations
            results['modularity_std'].append(np.std(mod_values, ddof=1))
            results['nmi_std'].append(np.std(nmi_values, ddof=1))
            results['ami_std'].append(np.std(ami_values, ddof=1))
            results['time_std'].append(np.std(time_values, ddof=1))
            
    return results


# ======================================================================
# Helpers
# ======================================================================

def plot_results(results):
    """
    Plot community detection results with or without confidence intervals
    depending on available data.
    
    Parameters:
    -----------
    results : dict
        Dictionary with algorithm results data
    """
    df = pd.DataFrame(results)    
    algorithms = df['algorithm'].unique()
    
    # Check if we have standard deviation data
    has_std_data = ('nmi_std' in df.columns and 'ami_std' in df.columns and 
                   'modularity_std' in df.columns and 'time_std' in df.columns)
    
    plt.figure(figsize=(15, 10))
    
    # Plot NMI
    plt.subplot(2, 2, 1)
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        try:
            mu_values = alg_data['mu'].values
        except Exception as e:
            mu_values - alg_data['nodes'].values
        nmi_values = alg_data['nmi'].values
        
        if has_std_data:
            nmi_std = alg_data['nmi_std'].values
            plt.errorbar(mu_values, nmi_values, yerr=nmi_std, fmt='o-', label=alg, capsize=5)
        else:
            plt.plot(mu_values, nmi_values, 'o-', label=alg)
    
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('NMI')
    plt.title('Normalized Mutual Information')
    plt.legend()
    plt.grid(True)
    
    # Plot AMI
    plt.subplot(2, 2, 2)
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        mu_values = alg_data['mu'].values
        ami_values = alg_data['ami'].values
        
        if has_std_data:
            ami_std = alg_data['ami_std'].values
            plt.errorbar(mu_values, ami_values, yerr=ami_std, fmt='o-', label=alg, capsize=5)
        else:
            plt.plot(mu_values, ami_values, 'o-', label=alg)
    
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('AMI')
    plt.title('Adjusted Mutual Information')
    plt.legend()
    plt.grid(True)
    
    # Plot Modularity
    plt.subplot(2, 2, 3)
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        mu_values = alg_data['mu'].values
        mod_values = alg_data['modularity'].values
        
        if has_std_data:
            mod_std = alg_data['modularity_std'].values
            plt.errorbar(mu_values, mod_values, yerr=mod_std, fmt='o-', label=alg, capsize=5)
        else:
            plt.plot(mu_values, mod_values, 'o-', label=alg)
    
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('Modularity')
    plt.title('Modularity')
    plt.legend()
    plt.grid(True)
    
    # Plot Time
    plt.subplot(2, 2, 4)
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        mu_values = alg_data['mu'].values
        time_values = alg_data['time'].values
        
        if has_std_data:
            time_std = alg_data['time_std'].values
            plt.errorbar(mu_values, time_values, yerr=time_std, fmt='o-', label=alg, capsize=5)
        else:
            plt.plot(mu_values, time_values, 'o-', label=alg)
    
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('Time (s)')
    plt.title('Computation Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('community_detection_results.png')
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

register_algorithm('MOCD', pymocd, needs_conversion=False)

def louvain_wrapper(G, seed=None):
    return louvain_communities(G, seed=seed)

register_algorithm('Louvain', louvain_wrapper, needs_conversion=True)

def pyevoea_nsga_wrapper(G, seed=None):
    import pyevoea
    if seed is not None:
        np.random.seed(seed)
    return pyevoea.MocdNsgaII(G).max_q()

register_algorithm('NSGA-II', pyevoea_nsga_wrapper, needs_conversion=False)

def pyevoea_mocd_wrapper(G, seed=None):
    import pyevoea 
    if seed is not None:
        np.random.seed(seed)
    return pyevoea.MocdPesaII(G).min_max()

def pyevoea_mocdq_wrapper(G, seed=None):
    import pyevoea 
    if seed is not None:
        np.random.seed(seed)
    return pyevoea.MocdPesaII(G).max_q()

register_algorithm('MOCD-Q', pyevoea_mocdq_wrapper, needs_conversion=False)
register_algorithm('MOCD-M', pyevoea_mocd_wrapper, needs_conversion=False)

def read_results_from_csv(filename='community_detection_results.csv'):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        
        # Initialize the results dictionary with required columns
        results = {
            'algorithm': [],
            'mu': [],
            'modularity': [],
            'nmi': [],
            'ami': [],
            'time': []
        }
        
        # Add standard deviation columns if they exist in the CSV
        std_columns = ['modularity_std', 'nmi_std', 'ami_std', 'time_std']
        for col in std_columns:
            if col in df.columns:
                results[col] = []
        
        # Extract data from DataFrame into the results dictionary
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
    mus = np.arange(0.1, 0.5, 0.1)  # 0.1, ..., 0.7
    
    print(f"Running community detection algorithms on LFR benchmark graphs with 1000 nodes")
    print(f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")

    # Run experiments
    #results = read_results_from_csv('community_detection_results.csv')
    results = run_experiment(mus=mus, n_runs=2)
    #results = run_nodes_experiment(n_runs=2)

    # Print table of results
    print("\nResults:")
    print_results_table(results)    
    plot_results(results)    
    save_results_to_csv(results)