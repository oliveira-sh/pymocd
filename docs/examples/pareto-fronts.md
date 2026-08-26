# Pareto fronts

The evolutionary search optimizes competing objectives at once, so it ends with a **Pareto front**: a set of partitions where no member is better than another on every objective — coarse trades against fine, tight communities against well-separated ones.

Every detector already resolves this — each applies the selection rule published with its algorithm (max-modularity for most; a label-free map-equation / resolution-plateau chain for `mr_mocd`; Shi's max-min distance to random-graph control fronts for `mocd_d`; see [Algorithms](../algorithms.md)) and returns a single partition. The front accessors exist for making that choice yourself: ground truth, a known expected community count, or your own quality metric.

## The `*_fronts` functions

Six of the ten detector entry points pair with a `*_fronts` function exposing the candidate set as a plain `list[dict]` of partitions: `mr_mocd`, `hpmocd`, `mmcomo`, `ccm`, `krm` and `moga_net`. Each list is exactly what the corresponding detector selects from:

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
front, points, selected = pymocd.mr_mocd_fronts(G)

for partition in front:
    k = len(set(partition.values()))
    print(f"{k} communities")
```

## Picking a member yourself

### Max modularity

```python
def modularity(G, partition):
    comms = {}
    for node, c in partition.items():
        comms.setdefault(c, set()).add(node)
    return nx.community.modularity(G, comms.values())

best = max(front, key=lambda p: modularity(G, p))
```

### Against ground truth

Score every member with `pymocd.gt_metrics` (or `ari`, `nmi`, `ami`, `f1` individually). Karate club, using the `club` attribute as ground truth:

```python
gt = {v: int(G.nodes[v]['club'] != 'Mr. Hi') for v in G}

best = max(front, key=lambda p: pymocd.ari(p, gt))
nmi, ami, ari, f1 = pymocd.gt_metrics(best, gt)
print(f"NMI={nmi:.3f} AMI={ami:.3f} ARI={ari:.3f} F1={f1:.3f}")
```

!!! note "The oracle may not be on the front"
    Even the best front member can fall short of ARI = 1.0: the ground-truth partition may be *dominated* under the search objectives and never survive to the final front. The front bounds what selection can recover.

### Target community count

```python
target = 2
best = min(front, key=lambda p: abs(len(set(p.values())) - target))
```

`gdpso` and `cdrme` optimize a single scalar — Newman-Girvan modularity and the CDRME paper's Eq. (12) linkage sum respectively — so they have no Pareto front and no `gdpso_fronts` / `cdrme_fronts`. `mocd_q` and `mocd_d` are multi-objective but expose no front accessor.

Each `*_fronts` function takes the same kwargs as its detector. `mr_mocd_fronts` returns `(partitions, points, selected)` rather than a bare list: every member, its `(cut, pair)` point, and the index the selector picked.

`mr_mocd` also offers [`mr_mocd_select`](../api/fronts.md#pymocd.mr_mocd_select), which runs its label-free selection chain over a candidate set this library did not produce — the control that separates the search's contribution from the selector's:

```python
pick, points = pymocd.mr_mocd_select(G, candidates)
print(f"selected k={len(set(candidates[pick].values()))}")
```

The baseline fronts exist because the original papers report the best-NMI solution *of the front*, not the max-modularity one their detectors return — reproducing those tables needs the full candidate set:

```python
front = pymocd.moga_net_fronts(G, r=2.0)   # Pizzuti Table 1 protocol (TEVC 2012 Sec. VI-C)
best_nmi = max(pymocd.nmi(p, gt) for p in front)
```

## See also

- [Plotting](plotting.md) — visualize the trade-off the front spans and draw the selected partition.
- [Fronts API](../api/fronts.md) — full signatures for the `*_fronts` functions.
