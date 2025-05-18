import pymocd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_lfr_benchmark(n=1000, tau1=2.5, tau2=1.5, mu=0.3, average_degree=20, 
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

# Create the graph
G, _ = generate_lfr_benchmark()
G = nx.karate_club_graph()
alg = pymocd.MOCD(G, num_gens=300)
pareto_front = alg.generate_pareto_front()

# Extract the objective values from the pareto front
intra_values = []
inter_values = []
score_values = []  # To store 1 - intra - inter scores
solutions = []  # To store all solutions

for solution in pareto_front:
    # Pareto front structure is (community_assignment_dict, [objective1, objective2])
    community_dict = solution[0]
    objectives = solution[1]
    intra = objectives[0]
    inter = objectives[1]
    
    intra_values.append(intra)
    inter_values.append(inter)
    score = 1 - intra - inter
    score_values.append(score)
    solutions.append((community_dict, [intra, inter], score))

# Find the best solution according to max(1 - intra - inter)
best_idx = np.argmax(score_values)
best_intra = intra_values[best_idx]
best_inter = inter_values[best_idx]
best_solution = solutions[best_idx]
best_score = score_values[best_idx]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(intra_values, inter_values, c='blue', s=50, alpha=0.7, edgecolors='black')

# Highlight the best solution
plt.scatter(best_intra, best_inter, c='red', s=100, edgecolors='black', zorder=5)

# Create an arrow pointing to the best solution
arrow_start_x = best_intra - 0.03
arrow_start_y = best_inter - 0.03

# Add title and labels
plt.title('Pareto Front for Community Detection in Karate Club Graph')
plt.xlabel('Intra-community Density')
plt.ylabel('Inter-community Sparsity')

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Determine appropriate axis limits with padding for the arrow and annotation
x_min = min(min(intra_values), arrow_start_x) - 0.05
x_max = max(intra_values) + 0.01
y_min = min(min(inter_values), arrow_start_y) - 0.05
y_max = max(inter_values) + 0.01

plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])

# Annotate the number of unique solutions
unique_solutions = set([(round(x, 6), round(y, 6)) for x, y in zip(intra_values, inter_values)])
plt.annotate(f'Number of unique solutions: {len(unique_solutions)}', 
             xy=(0.02, 0.02), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Print the best solution details
print("Best solution based on max(1 - intra - inter):")
print(f"Community assignment: {best_solution[0]}")
print(f"Intra value: {best_intra}")
print(f"Inter value: {best_inter}")
print(f"Score (1 - intra - inter): {best_score}")

# Save the figure
plt.savefig("pareto_front_plot.png")

# Show the plot
plt.show()

# Print out all unique objective value pairs for reference
print("\nAll unique objective value pairs found:")
for pair in unique_solutions:
    score = 1 - pair[0] - pair[1]
    print(f"Intra: {pair[0]}, Inter: {pair[1]}, Score (1-intra-inter): {score}")

# ----------------------------------------
# Plot Q-value (1 - intra - inter) vs. number of communities
# ----------------------------------------

# Compute number of communities and Q-values for each solution
num_communities = [len(set(comm_dict.values())) for comm_dict, _, _ in solutions]
q_values       = [score for _, _, score in solutions]

# If you want to highlight the best solution on this plot:
best_num_com = len(set(best_solution[0].values()))

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    num_communities,
    q_values,
    c='green',
    s=50,
    alpha=0.7,
    edgecolors='black',
    label='All solutions'
)

# Highlight the best solution in red
plt.scatter(
    best_num_com,
    best_score,
    c='red',
    s=100,
    edgecolors='black',
    zorder=5,
    label=f'Best Q = {best_score:.4f}'
)


# Labels, title, grid, legend
plt.title('Q vs. Number of Communities')
plt.xlabel('Number of Communities')
plt.ylabel('Q')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Tight layout, save and show
plt.tight_layout()
plt.savefig("q_vs_communities_plot.png")
plt.show()
