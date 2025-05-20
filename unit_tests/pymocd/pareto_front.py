import pymocd
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_lfr_benchmark

G, _ = generate_lfr_benchmark()
pareto_front = pymocd.HpMocd(G).generate_pareto_front()

# Q = 1 - intra - inter
solutions = [
    (comm, intra, inter, 1 - intra - inter)
    for comm, (intra, inter) in pareto_front
]
intra_vals = [s[1] for s in solutions]
inter_vals = [s[2] for s in solutions]
scores = [s[3] for s in solutions]

best_comm, best_intra, best_inter, best_score = max(solutions, key=lambda s: s[3])
best_num_com = len(set(best_comm.values()))

unique_pairs = {(round(i,6), round(e,6)) for i, e in zip(intra_vals, inter_vals)}

plt.figure(figsize=(10, 6))
plt.scatter(intra_vals, inter_vals, s=50, alpha=0.7, edgecolors='black')
plt.scatter(best_intra, best_inter, s=100, color='red', edgecolors='black', zorder=5)
plt.title('Pareto Front for Community Detection')
plt.xlabel('Intra-community Density')
plt.ylabel('Inter-community Sparsity')
plt.grid(linestyle='--', alpha=0.7)
plt.annotate(f'Unique solutions: {len(unique_pairs)}',
             xy=(0.02, 0.02), xycoords='axes fraction',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
plt.savefig("pareto_front_plot.png")
plt.show()

print(f"Best solution (Intra={best_intra:.4f}, Inter={best_inter:.4f}, Score={best_score:.4f}):")
print(best_comm)

print("\nAll unique objective pairs:")
for i, e in unique_pairs:
    print(f"Intra: {i:.4f}, Inter: {e:.4f}, Score: {1 - i - e:.4f}")

# Plot Q vs. number of communities
num_comms = [len(set(comm.values())) for comm, *_ in solutions]
q_vals = scores

plt.figure(figsize=(10, 6))
plt.scatter(num_comms, q_vals, s=50, alpha=0.7, edgecolors='black', label='All solutions')
plt.scatter(best_num_com, best_score, s=100, color='red', edgecolors='black', zorder=5,
            label=f'Best Q = {best_score:.4f}')
plt.title('Q vs. Number of Communities')
plt.xlabel('Number of Communities')
plt.ylabel('Q')
plt.grid(linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("q_vs_communities_plot.png")
plt.show()
