import numpy as np
import networkx as nx
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.population import Population
from collections import defaultdict

class MOCDProblem(ElementwiseProblem):
    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = len(graph.nodes())
        self.node_list = list(graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.idx_to_node = {i: node for i, node in enumerate(self.node_list)}
        self.m = len(graph.edges())  # Total number of edges
        
        # For each node, get list of adjacent nodes (including itself)
        self.adjacency_list = {}
        for node in self.node_list:
            neighbors = list(graph.neighbors(node))
            if node not in neighbors:  # Include the node itself as a neighbor
                neighbors.append(node)
            self.adjacency_list[node] = neighbors
        
        # For the mutation operator, need to know valid values for each variable
        n_vars = self.n_nodes
        
        # Calculate degree for each node (used in objective calculation)
        self.degrees = {node: graph.degree(node) for node in self.node_list}
        
        super().__init__(n_var=n_vars, 
                         n_obj=2,
                         n_ieq_constr=0, 
                         xl=0,
                         xu=1)  # We'll use float encoding and map to neighbors
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Convert continuous values to actual neighbors
        communities = self._decode_solution(x)
        
        # Calculate objectives
        intra_obj = self._calc_intra_objective(communities)
        inter_obj = self._calc_inter_objective(communities)
        
        out["F"] = np.array([intra_obj, inter_obj])
    
    def _decode_solution(self, x):
        """Decode a solution to get communities"""
        # Convert continuous values to neighbor indices
        genotype = {}
        for i, val in enumerate(x):
            node = self.idx_to_node[i]
            neighbors = self.adjacency_list[node]
            # Map continuous value to a neighbor index
            neighbor_idx = int(val * len(neighbors)) % len(neighbors)
            genotype[node] = neighbors[neighbor_idx]
        
        # Find connected components to identify communities
        temp_graph = nx.Graph()
        for node, neighbor in genotype.items():
            temp_graph.add_edge(node, neighbor)
        
        # Get communities as connected components
        communities = list(nx.connected_components(temp_graph))
        return communities
    
    def _calc_intra_objective(self, communities):
        """Calculate intra-community objective: 1 - sum(|E(c)|/m for c in C)"""
        intra_sum = 0
        for community in communities:
            community_edges = self.graph.subgraph(community).number_of_edges()
            intra_sum += community_edges / self.m
        
        return 1 - intra_sum
    
    def _calc_inter_objective(self, communities):
        """Calculate inter-community objective: sum((sum(deg(v))/2m)^2 for c in C)"""
        inter_sum = 0
        for community in communities:
            total_degree = sum(self.degrees[node] for node in community)
            inter_sum += (total_degree / (2 * self.m)) ** 2
        
        return inter_sum

def calculate_modularity(graph, communities):
    """Calculate modularity Q for a given community partition"""
    m = len(graph.edges())
    q = 0
    for community in communities:
        community_edges = graph.subgraph(community).number_of_edges()
        total_degree = sum(graph.degree(node) for node in community)
        q += (community_edges / m) - ((total_degree / (2 * m)) ** 2)
    
    return q

def communities_to_dict(communities, node_list):
    """Convert communities list to node_id:community_id dict"""
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    
    # Make sure all nodes are included
    for node in node_list:
        if node not in community_dict:
            community_dict[node] = -1  # Isolated nodes
    
    return community_dict

def generate_random_network(original_graph):
    """Generate a random graph with same number of nodes and edges"""
    random_graph = nx.configuration_model([original_graph.degree(n) for n in original_graph.nodes()])
    random_graph = nx.Graph(random_graph)  # Remove parallel edges
    random_graph.remove_edges_from(nx.selfloop_edges(random_graph))  # Remove self-loops
    return random_graph

def mocd(graph, pop_size=100, generations=100, random_runs=3):
    """Main MOCD function that detects communities using multi-objective optimization"""
    # Create problem instance
    problem = MOCDProblem(graph)
    
    # Configure algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.6, eta=15),
        mutation=PM(prob=0.4, eta=20),
        eliminate_duplicates=True
    )
    
    # Optimize
    res = minimize(problem,
                algorithm,
                ('n_gen', generations),
                seed=1,
                verbose=False)
    
    # Get Pareto front solutions
    pareto_front = res.F
    pareto_solutions = res.X
    
    # Run on random networks for Max-Min Distance model selection
    random_pareto_fronts = []
    for _ in range(random_runs):
        random_graph = generate_random_network(graph)
        random_problem = MOCDProblem(random_graph)
        random_res = minimize(random_problem,
                        algorithm,
                        ('n_gen', generations),
                        seed=1,
                        verbose=False)
        random_pareto_fronts.append(random_res.F)
    
    # Decode solutions and calculate modularity
    decoded_solutions = []
    modularities = []
    
    for solution in pareto_solutions:
        communities = problem._decode_solution(solution)
        decoded_solutions.append(communities)
        modularities.append(calculate_modularity(graph, communities))
    
    # Max Q model selection
    max_q_idx = np.argmax(modularities)
    max_q_solution = decoded_solutions[max_q_idx]
    
    # Max-Min Distance model selection
    max_min_dists = []
    for i, solution_f in enumerate(pareto_front):
        min_dist = float('inf')
        for random_front in random_pareto_fronts:
            for random_f in random_front:
                dist = np.sqrt(np.sum((solution_f - random_f) ** 2))
                min_dist = min(min_dist, dist)
        max_min_dists.append(min_dist)
    
    max_min_idx = np.argmax(max_min_dists)
    max_min_solution = decoded_solutions[max_min_idx]
    
    # Choose Max-Min solution as final result (can be changed to Max Q if preferred)
    selected_communities = max_min_solution
    
    # Convert to node:community dict
    result = communities_to_dict(selected_communities, graph.nodes())
    
    return result

def detect_communities(graph):
    """Interface function that receives a networkx graph and returns community assignments"""
    return mocd(graph)

# Example usage
if __name__ == "__main__":
    # Create a simple test graph
    G = nx.karate_club_graph()
    communities = detect_communities(G)
    print(communities)