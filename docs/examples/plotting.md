# Plotting communities

Visualizing a partition — and the Pareto front it came from — is the quickest way to sanity-check what a detector actually found.

## A reusable plotting helper

The helper below draws a graph with nodes colored by community, using matplotlib's `tab20` colormap. It works with any partition returned by pymocd (a `dict` mapping node to community id).

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
    pymocd assigns isolated nodes the community id `-1`. They show up as their own color group, which is usually what you want — but you can filter them out of `labels` beforehand if not.

## Example: karate club with `scale`

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
labels = pymocd.scale(G)
plot_communities(G, labels)
```

Any of the module-level detectors (`scale`, `hpmocd`, `mmcomo`, ...) or `HpMocd(...).run()` return the same `dict[node, community]` shape, so the helper works unchanged.

## Plotting the Pareto front

`HpMocd.generate_pareto_front()` returns a list of `(partition, [objective_values])` pairs — the full set of non-dominated trade-offs found by the search. Scattering the two objective values shows the shape of that trade-off.

Both built-in objectives (intra and inter) are **minimized**, so on both axes lower is better and the ideal region is the bottom-left corner.

```python
import matplotlib.pyplot as plt
import networkx as nx
import pymocd

G = nx.karate_club_graph()
alg = pymocd.HpMocd(graph=G)
frontier = alg.generate_pareto_front()

intra = [objs[0] for _, objs in frontier]
inter = [objs[1] for _, objs in frontier]

plt.scatter(inter, intra)
plt.xlabel("Inter (lower is better)")
plt.ylabel("Intra (lower is better)")
plt.title("Pareto front")
plt.grid(True)
plt.show()
```

## Picking a solution off the front

Rather than eyeballing an index, pick the front member that maximizes modularity and plot it:

```python
best, _ = max(
    frontier,
    key=lambda entry: nx.community.modularity(
        G, [{n for n, c in entry[0].items() if c == k} for k in set(entry[0].values())]
    ),
)
plot_communities(G, best, title="Max-modularity front member")
```

This is the same criterion `HpMocd.run()` uses internally, but iterating over the front yourself lets you swap in any selection rule you like.

For a deeper look at what the front contains and how to work with it, see [Pareto fronts](pareto-fronts.md).
