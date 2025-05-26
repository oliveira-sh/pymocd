# plot the algorithm evolution based on some parameters

import os
import pymocd
import matplotlib.pyplot as plt
import networkx as nx
from utils import generate_lfr_benchmark, SAVE_PATH

G, _ = generate_lfr_benchmark(n=250, mu=0.1)
GENERATIONS = [10, 30, 50, 80, 100, 110]
os.makedirs(SAVE_PATH, exist_ok=True)
pos = nx.spring_layout(G, seed=42)

for gen in GENERATIONS:
    model = pymocd.HpMocd(G, num_gens=gen)
    partition = model.run()
    plt.figure(figsize=(4, 3))
    nx.draw(
        G, pos,
        node_color=[partition[n] for n in G.nodes()],
        with_labels=False,
        node_size=40,
        width=0.001,
        cmap="tab20"
    )
    out_path = os.path.join(SAVE_PATH, f"communities_gen_{gen}.pdf")
    plt.savefig(out_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.close()
