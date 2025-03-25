import networkx as nx

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

G, _ = generate_lfr_benchmark(n=10000, mu=0.3, seed=23)
nx.write_adjlist(G, "graph.adjlist")