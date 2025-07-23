import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

SAVE_PATH = "tests/output"

def generate_lfr_benchmark(n=1000, tau1=2.5, tau2=1.5, mu=0.3, average_degree=20, 
                           min_community=20, seed=0):
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree, 
            min_community=min_community, max_degree=50, seed=seed, max_community=100
        )        
        communities = {node: frozenset(G.nodes[node]['community']) for node in G}        
        G = nx.Graph(G)  
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
# Plotting and saving results
# ======================================================================

def read_results_from_csv(filename='community_detection_results.csv'):
    try:
        df = pd.read_csv(filename)
        results = {
            'algorithm': [],
            'modularity': [],
            'nmi': [],
            'ami': [],
            'time': []
        }
        if 'mu' in df.columns:
            results['mu'] = []
        elif 'nodes' in df.columns:
            results['nodes'] = []
        else:
            raise ValueError("Neither 'mu' nor 'nodes' found in CSV")
        std_columns = ['modularity_std', 'nmi_std', 'ami_std', 'time_std']
        for col in std_columns:
            if col in df.columns:
                results[col] = []
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

def plot_results(results):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import rcParams
    
    plt.style.use('default') 
    df = pd.DataFrame(results)
    algorithms = df['algorithm'].unique()
    if 'mu' in df.columns:
        x_var = 'mu'
        x_label = 'μ'  # LaTeX format for mu
    elif 'nodes' in df.columns:
        x_var = 'nodes'
        x_label = 'n'  # LaTeX format for n
    else:
        raise ValueError("Neither 'mu' nor 'nodes' found in results")    
    has_std_data = all(col in df.columns for col in ['nmi_std', 'ami_std', 'time_std'])
    metrics = [
        {'key': 'nmi', 'ylabel': 'NMI'},
        {'key': 'ami', 'ylabel': 'AMI'},
        {'key': 'time', 'ylabel': 'Time (s)'}
    ]    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']    
    for metric in metrics:
        fig, ax = plt.subplots()        
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm'] == alg].sort_values(by=x_var)
            x_values = alg_data[x_var].values
            y_values = alg_data[metric['key']].values            
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            ax.plot(x_values, y_values, 
                   marker=marker, 
                   linestyle=linestyle,
                   label=alg)            
            if has_std_data:
                std_key = metric['key'] + '_std'
                if std_key in alg_data.columns:
                    y_std = alg_data[std_key].values
                    lower_bound = y_values - y_std
                    upper_bound = y_values + y_std
                    ax.fill_between(x_values, lower_bound, upper_bound, alpha=0.2)        
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric['ylabel'])
        
        # if the metric is 'time', use log in y axis .
        if metric['key'] == 'time':
            ax.set_yscale("log")        
        ax.legend(loc='best', frameon=False, handlelength=1.5, handletextpad=0.5)        
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        plt.tight_layout(pad=0.3)        
        plt.savefig(f"{SAVE_PATH}{metric['key']}_plot.pdf", format='pdf', bbox_inches='tight')
        #plt.savefig(f"{metric['key']}_plot.png", dpi=600, bbox_inches='tight')
        plt.close()