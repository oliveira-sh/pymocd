# Pareto fronts

Multi-objective community detection does not produce one answer. The evolutionary search optimizes two (or more) competing objectives at once, so it ends with a **Pareto front**: a set of partitions where no member is better than another on every objective. Coarse partitions trade against fine ones, tight communities against well-separated ones.

Every detector in pymocd already resolves this for you — each applies the selection rule published with its algorithm (max-modularity, MDL, etc.; see [Algorithms](../algorithms.md)) and returns a single partition. The front accessors exist for when you want to make that choice yourself: you have ground truth, domain knowledge about the expected number of communities, or your own quality metric.

## `HpMocd.generate_pareto_front()`

Returns every non-dominated solution as a list of `(partition, objective_values)` tuples:

- `partition`: `dict[node, community]`
- `objective_values`: `list[float]`, one value per objective, **all minimized** — the built-in intra and inter objectives included. With custom objectives, the list matches the objectives you passed, in order.

```python
import networkx as nx
from pymocd import HpMocd

G = nx.karate_club_graph()
front = HpMocd(G).generate_pareto_front()

for partition, objectives in front:
    k = len(set(partition.values()))
    print(f"{k} communities, objectives = {objectives}")
```

!!! warning "Both objectives are minimized"
    Older documentation described intra as "higher is better". It is not: lower is better for **every** value in `objective_values`.

## Picking a member yourself

### Max modularity

Score each partition with NetworkX and keep the best:

```python
def modularity(G, partition):
    comms = {}
    for node, c in partition.items():
        comms.setdefault(c, set()).add(node)
    return nx.community.modularity(G, comms.values())

best, _ = max(front, key=lambda sol: modularity(G, sol[0]))
```

### Against ground truth

If you have labels, score every member with `pymocd.gt_metrics` (or `ari`, `nmi`, `ami`, `f1` individually) and pick the best. Karate club, using the `club` attribute as ground truth:

```python
import pymocd

gt = {v: int(G.nodes[v]['club'] != 'Mr. Hi') for v in G}

best, _ = max(front, key=lambda sol: pymocd.ari(sol[0], gt))
nmi, ami, ari, f1 = pymocd.gt_metrics(best, gt)
print(f"NMI={nmi:.3f} AMI={ami:.3f} ARI={ari:.3f} F1={f1:.3f}")
```

!!! note "The oracle may not be on the front"
    Even the best front member can fall short of ARI = 1.0: the ground-truth partition may be *dominated* under the search objectives and never survive to the final front. The front bounds what selection can recover.

### Target community count

```python
target = 2
best, _ = min(front, key=lambda sol: abs(len(set(sol[0].values())) - target))
```

## `scale_fronts` and `mmcomo_fronts`

`scale` and `mmcomo` expose their candidate sets too, but as a plain `list[dict]` of partitions — no objective values attached. Each list is exactly what the corresponding detector selects from:

```python
candidates = pymocd.scale_fronts(G)
best = max(candidates, key=lambda p: pymocd.ari(p, gt))
```

`scale_fronts` takes the same kwargs as `scale`, plus two of its own: `refine` (apply union-refinement to the merged front, on by default) and `topo_mode` (topology-handling mode, integer).

## See also

- [Plotting](plotting.md) — visualize the front in objective space and draw the selected partition.
- [Fronts API](../api/fronts.md) — full signatures for `generate_pareto_front`, `scale_fronts`, and `mmcomo_fronts`.
