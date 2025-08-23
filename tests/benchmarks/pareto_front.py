import pymocd
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_lfr_benchmark, evaluate_communities, SAVE_PATH

G, ground_truth = generate_lfr_benchmark()
pareto_front = pymocd.HpMocd(G).generate_pareto_front()
solutions = [
    (comm, intra, inter, 1 - intra - inter)
    for comm, (intra, inter) in pareto_front
]
intra_vals = [s[1] for s in solutions]
inter_vals = [s[2] for s in solutions]
q_vals     = [s[3] for s in solutions]

# Identify best Q
best_comm, best_intra, best_inter, best_q = max(solutions, key=lambda s: s[3])
best_num_com = len(set(best_comm.values()))
metrics = [evaluate_communities(G, comm, ground_truth, convert=False) for comm, *_ in solutions]
mod_vals = [m['modularity'] for m in metrics]
nmi_vals = [m['nmi']        for m in metrics]
ami_vals = [m['ami']        for m in metrics]
best_metrics = evaluate_communities(G, best_comm, ground_truth, False)
best_mod, best_nmi, best_ami = best_metrics['modularity'], best_metrics['nmi'], best_metrics['ami']

# ---- PLOT 1: Pareto front ----
plt.figure(figsize=(4,3))
plt.scatter(intra_vals, inter_vals, s=50, alpha=0.7, edgecolors='black')
plt.scatter(best_intra, best_inter, s=120, color='red', edgecolors='black', zorder=5)
plt.title('Pareto Frontier')
plt.xlabel('Intra')
plt.ylabel('Inter')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}pareto_front_plot.pdf")     # <- PDF now

# ---- PLOT 2: Q vs. Number of Communities ----
num_coms = [len(set(comm.values())) for comm, *_ in solutions]

plt.figure(figsize=(4,3))
plt.scatter(num_coms, q_vals, s=50, alpha=0.7, edgecolors='black', label='All solutions')
plt.scatter(best_num_com, best_q, s=120, color='red', edgecolors='black', zorder=5,
            label=f'Best Q = {best_q:.4f}')
plt.title('Q vs. Number of Communities')
plt.xlabel('Number of Communities')
plt.ylabel('Q')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}q_vs_communities_plot.pdf")  # <- PDF now


# ---- PLOT 3A: Q vs. Modularity ----
plt.figure(figsize=(4,3))
plt.scatter(mod_vals, q_vals, s=50, alpha=0.7, edgecolors='black')
plt.scatter(best_mod, best_q, s=120, color='red', edgecolors='black', zorder=5)
plt.title('Q vs. Modularity')
plt.xlabel('Modularity')
plt.ylabel('Q')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}q_vs_modularity_plot.pdf")


# ---- PLOT 3B: Q vs. NMI ----
plt.figure(figsize=(4,3))
plt.scatter(nmi_vals, q_vals, s=50, alpha=0.7, edgecolors='black')
plt.scatter(best_nmi, best_q, s=120, color='red', edgecolors='black', zorder=5)
plt.title('Q vs. NMI')
plt.xlabel('NMI')
plt.ylabel('Q')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}q_vs_nmi_plot.pdf")

# ---- PLOT 3C: Q vs. AMI ----
plt.figure(figsize=(4,3))
plt.scatter(ami_vals, q_vals, s=50, alpha=0.7, edgecolors='black')
plt.scatter(best_ami, best_q, s=120, color='red', edgecolors='black', zorder=5)
plt.title('Q vs. AMI')
plt.xlabel('AMI')
plt.ylabel('Q')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}q_vs_ami_plot.pdf")
