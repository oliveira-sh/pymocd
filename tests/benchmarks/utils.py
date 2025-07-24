import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configure matplotlib for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

SAVE_PATH = "tests/output"

# Color palette for algorithms (colorblind-friendly)
ALGORITHM_COLORS = {
    'Louvain': '#1f77b4',
    'Leiden': '#ff7f0e',
    'ASYN-LPA': '#2ca02c',
    'HPMOCD': '#d62728',
    'NSGA-III KRM': '#9467bd',
    'NSGA-III CCM': '#8c564b',
    'CDRME': '#e377c2',
    'MOCD': '#7f7f7f',
    'MogaNet': '#bcbd22',
    'Default': '#17becf'
}

# Markers for different algorithms
ALGORITHM_MARKERS = {
    'Louvain': 'o',
    'Leiden': 's',
    'ASYN-LPA': '^',
    'HPMOCD': 'D',
    'NSGA-III KRM': 'v',
    'NSGA-III CCM': '<',
    'CDRME': '>',
    'MOCD': 'p',
    'MogaNet': 'h',
    'Default': 'x'
}

def generate_lfr_benchmark(n=1000, tau1=2.5, tau2=1.5, mu=0.3, average_degree=20, 
                           min_community=20, seed=0):
    """Generate LFR benchmark graph with ground truth communities."""
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
    """Convert list of communities to partition dictionary."""
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i
    return partition

def evaluate_communities(G, detected_communities, ground_truth_communities, convert=True):
    """Evaluate detected communities against ground truth."""
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

def read_results_from_csv(filename='community_detection_results.csv'):
    """Read experimental results from CSV file."""
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

def create_publication_plot(figsize=(10, 8), subplot_layout=None):
    """Create a publication-ready matplotlib figure."""
    if subplot_layout is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    else:
        fig, axes = plt.subplots(*subplot_layout, figsize=figsize)
        return fig, axes

def style_axis(ax, x_label, y_label, title=None, log_scale=None):
    """Apply consistent styling to an axis."""
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    
    if title:
        ax.set_title(title, fontweight='bold', pad=20)
    
    if log_scale == 'y':
        ax.set_yscale('log')
    elif log_scale == 'x':
        ax.set_xscale('log')
    elif log_scale == 'both':
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Style ticks
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=4, width=1.0)

def plot_single_metric(df: pd.DataFrame, metric: str, x_var: str, save_path: str = SAVE_PATH) -> None:
    """Plot a single metric with publication quality."""
    fig, ax = create_publication_plot(figsize=(10, 8))
    
    algorithms = df['algorithm'].unique()
    
    # Determine labels
    x_label = 'Mixing parameter (μ)' if x_var == 'mu' else 'Number of nodes (n)'
    y_labels = {
        'nmi': 'NMI',
        'ami': 'AMI',
        'modularity': 'Modularity',
        'time': 'Execution time (s)'
    }
    y_label = y_labels.get(metric, metric.upper())
    
    # Plot each algorithm
    for i, alg in enumerate(algorithms):
        alg_data = df[df['algorithm'] == alg].sort_values(by=x_var)
        x_values = alg_data[x_var].values
        y_values = alg_data[metric].values
        
        # Get colors and markers
        color = ALGORITHM_COLORS.get(alg, ALGORITHM_COLORS['Default'])
        marker = ALGORITHM_MARKERS.get(alg, ALGORITHM_MARKERS['Default'])
        
        # Plot main line
        ax.plot(x_values, y_values, 
               color=color, marker=marker, 
               label=alg, linewidth=2.5, markersize=8,
               markerfacecolor=color, markeredgecolor='white', 
               markeredgewidth=1.5)
        
        # Add confidence intervals if available
        std_key = f'{metric}_std'
        if std_key in alg_data.columns and not alg_data[std_key].isna().all():
            y_std = alg_data[std_key].values
            y_lower = y_values - y_std
            y_upper = y_values + y_std
            ax.fill_between(x_values, y_lower, y_upper, 
                           color=color, alpha=0.2, interpolate=True)
    
    # Style the plot
    log_scale = 'y' if metric == 'time' else None
    style_axis(ax, x_label, y_label, log_scale=log_scale)
    
    # Customize legend
    legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                      ncol=2 if len(algorithms) > 6 else 1,
                      handlelength=2.5, handletextpad=0.8,
                      columnspacing=1.0, borderaxespad=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.0)
    
    # Adjust layout
    plt.tight_layout(pad=1.0)
    
    # Save the plot
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / f"{metric}_plot.pdf", format='pdf', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(output_path / f"{metric}_plot.png", format='png', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def plot_comparison_matrix(df: pd.DataFrame, x_var: str, save_path: str = SAVE_PATH) -> None:
    """Create a comparison matrix of all metrics."""
    metrics = ['nmi', 'ami', 'modularity', 'time']
    fig, axes = create_publication_plot(figsize=(16, 12), subplot_layout=(2, 2))
    axes = axes.flatten()
    
    algorithms = df['algorithm'].unique()
    x_label = 'Mixing parameter (μ)' if x_var == 'mu' else 'Number of nodes (n)'
    
    y_labels = {
        'nmi': 'NMI',
        'ami': 'AMI', 
        'modularity': 'Modularity (Q)',
        'time': 'Time (s)'
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg].sort_values(by=x_var)
            x_values = alg_data[x_var].values
            y_values = alg_data[metric].values
            
            color = ALGORITHM_COLORS.get(alg, ALGORITHM_COLORS['Default'])
            marker = ALGORITHM_MARKERS.get(alg, ALGORITHM_MARKERS['Default'])
            
            ax.plot(x_values, y_values, 
                   color=color, marker=marker, 
                   label=alg if idx == 0 else "", 
                   linewidth=2.0, markersize=6,
                   markerfacecolor=color, markeredgecolor='white', 
                   markeredgewidth=1.0)
            
            # Add confidence intervals
            std_key = f'{metric}_std'
            if std_key in alg_data.columns and not alg_data[std_key].isna().all():
                y_std = alg_data[std_key].values
                y_lower = y_values - y_std
                y_upper = y_values + y_std
                ax.fill_between(x_values, y_lower, y_upper, 
                               color=color, alpha=0.15)
        
        # Style each subplot
        log_scale = 'y' if metric == 'time' else None
        style_axis(ax, x_label, y_labels[metric], log_scale=log_scale)
        
        # Add subplot labels
        ax.text(0.02, 0.98, f'({chr(97+idx)})', transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=min(len(algorithms), 5), frameon=True, fancybox=True, shadow=True,
              handlelength=2.0, handletextpad=0.8, columnspacing=1.5)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save the plot
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / "comparison_matrix.pdf", format='pdf', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(output_path / "comparison_matrix.png", format='png', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def plot_performance_radar(df: pd.DataFrame, save_path: str = SAVE_PATH) -> None:
    """Create a radar plot showing algorithm performance across metrics."""
    # Calculate mean performance for each algorithm
    metrics = ['nmi', 'ami', 'modularity']
    algorithms = df['algorithm'].unique()
    
    # Normalize metrics to 0-1 scale for radar plot
    normalized_data = {}
    for metric in metrics:
        max_val = df[metric].max()
        min_val = df[metric].min()
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            mean_val = alg_data[metric].mean()
            normalized_val = (mean_val - min_val) / (max_val - min_val)
            
            if alg not in normalized_data:
                normalized_data[alg] = {}
            normalized_data[alg][metric] = normalized_val
    
    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for alg in algorithms:
        values = [normalized_data[alg][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        color = ALGORITHM_COLORS.get(alg, ALGORITHM_COLORS['Default'])
        ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Customize radar plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['NMI', 'AMI', 'Modularity'], fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Save the plot
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path / "performance_radar.pdf", format='pdf', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(output_path / "performance_radar.png", format='png', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def plot_results(results: Union[Dict, pd.DataFrame], save_path: str = SAVE_PATH) -> None:
    """Plot all results with publication quality."""
    if isinstance(results, dict):
        df = pd.DataFrame(results)
    else:
        df = results
    
    # Determine x variable
    if 'mu' in df.columns:
        x_var = 'mu'
    elif 'nodes' in df.columns:
        x_var = 'nodes'
    else:
        raise ValueError("Neither 'mu' nor 'nodes' found in results")
    
    # Create individual metric plots
    metrics = ['nmi', 'ami', 'modularity', 'time']
    for metric in metrics:
        if metric in df.columns:
            plot_single_metric(df, metric, x_var, save_path)
    
    # Create comparison matrix
    plot_comparison_matrix(df, x_var, save_path)
    
    # Create radar plot (only if we have quality metrics)
    if all(metric in df.columns for metric in ['nmi', 'ami', 'modularity']):
        plot_performance_radar(df, save_path)
    
    print(f"All plots saved to {save_path}")

def create_summary_table(df: pd.DataFrame, save_path: str = SAVE_PATH) -> pd.DataFrame:
    """Create a publication-ready summary table."""
    # Calculate summary statistics
    summary = df.groupby('algorithm').agg({
        'modularity': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'ami': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Format for publication (mean ± std)
    formatted_summary = pd.DataFrame(index=summary.index)
    for metric in ['modularity', 'nmi', 'ami', 'time']:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        formatted_summary[metric.upper()] = summary.apply(
            lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}", axis=1
        )
    
    # Save to LaTeX table
    latex_table = formatted_summary.to_latex(
        caption="Performance comparison of community detection algorithms",
        label="tab:algorithm_comparison",
        column_format='l' + 'c' * len(formatted_summary.columns),
        escape=False
    )
    
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "summary_table.tex", 'w') as f:
        f.write(latex_table)
    
    formatted_summary.to_csv(output_path / "summary_table.csv")
    
    print(f"Summary table saved to {save_path}")
    return formatted_summary