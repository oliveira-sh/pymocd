# Plotting communities

## A reusable plotting helper

Draws a graph with nodes colored by community (matplotlib `tab20`); works with any partition returned by pymocd (a `dict` mapping node to community id).

```python
import matplotlib.pyplot as plt
import networkx as nx

def plot_communities(G, labels, title="Community assignment"):
    pos = nx.spring_layout(G, seed=42)
    communities = sorted(set(labels.values()))
    cmap = plt.colormaps["tab20"].resampled(len(communities))

    for idx, c in enumerate(communities):
        nodes = [n for n in G.nodes if labels[n] == c]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=[cmap(idx)],
            label=f"Community {c}",
        )

    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()
```

!!! note "Isolated nodes"
    pymocd assigns isolated nodes the community id `-1`. They show up as their own color group; filter them out of `labels` beforehand if unwanted.

## Example: karate club with `smocc`

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
labels = pymocd.smocc(G)
plot_communities(G, labels)
```

All module-level detectors (`smocc`, `hpmocd`, `mmcomo`, ...) return the same `dict[node, community]` shape.

## Plotting the Pareto front

The `*_fronts` functions return the candidate set as a `list[dict]` of partitions. A quick way to see the trade-off the front spans is to scatter each member's community count against its modularity:

```python
import matplotlib.pyplot as plt
import networkx as nx
import pymocd

G = nx.karate_club_graph()
frontier = pymocd.hpmocd_fronts(G)

def modularity(G, partition):
    comms = {}
    for node, c in partition.items():
        comms.setdefault(c, set()).add(node)
    return nx.community.modularity(G, comms.values())

k = [len(set(p.values())) for p in frontier]
q = [modularity(G, p) for p in frontier]

plt.scatter(k, q)
plt.xlabel("Communities")
plt.ylabel("Modularity")
plt.title("Pareto front")
plt.grid(True)
plt.show()
```

## Picking a solution off the front

Pick the front member that maximizes modularity and plot it:

```python
best = max(frontier, key=lambda p: modularity(G, p))
plot_communities(G, best, title="Max-modularity front member")
```

This is the same criterion `hpmocd` uses internally, but iterating over the front yourself lets you swap in any selection rule.

See [Pareto fronts](pareto-fronts.md) for more on working with the front.
