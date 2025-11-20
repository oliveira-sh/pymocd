---
weight: 230
title: "Generate Pareto Front"
icon: dashboard
description: "Step-by-step guide to extract and interpret the Pareto front of community-detection solutions."
lead: "Learn how to generate the Pareto front from HpMocd and choose the best trade-off solution for your needs."
date: 2022-10-19T21:49:38+01:00
lastmod: 2023-08-24T16:34:38+01:00
draft: false
images: []
toc: true
aliases:
  - /docs/hpmocd/pareto/
  - /hpmocd/pareto/
---

## Pareto Front Generation

The Pareto front lets you explore the trade-off between **intra-community density** (how tight each community is) and **inter-community separation** (how distinct communities are). Instead of a single “best” solution, you get a spectrum of optimal solutions—so you can pick the one that balances density and separation to your liking.

### Instantiate the Algorithm

First, set up **HpMocd** exactly as in the Basic Usage section:

```python
import networkx as nx
from pymocd import HpMocd

G = nx.karate_club_graph()
alg = HpMocd(
    graph=G,
    debug_level=1,
    pop_size=100,
    num_gens=100,
    cross_rate=0.8,
    mut_rate=0.2
)
```

### Generate the Pareto Front

Call the `generate_pareto_front() method:

```python
frontier = alg.generate_pareto_front()
```

* **Return value**:
  A list of tuples `[(assignment, metrics), …]`

  * `assignment` (`dict[int, int]`): maps each node → community ID
  * `metrics` (`tuple[float, float]`):

    * `[0]` = inter-community score 
    * `[1]` = intra-community score

### Inspect a Solution

Suppose you want the solution at index 23:

```python
labels, (intra_score, inter_score) = frontier[23]

print("Node → Community:", labels)
print(f"Inter: {intra_score:.4f}, Intra: {inter_score:.4f}")
```

### Selecting Your Preferred Solution

Now that you have the full Pareto front, you can choose:

* **Maximum intra-community density**
  Pick the tuple with the highest `intra_score`.
* **Minimum inter-community connectivity**
  Pick the tuple with the lowest `inter_score`.
* **Combined metric or domain-specific measure**
  Compute modularity, conductance, cut ratio, etc., on each `labels` dict and choose the best.
* **Manual or visual selection**
  Plot `(inter_score, intra_score)` to see the frontier and pick by eye.

```python
# Example: find index of highest intra score
best_intra_idx = max(range(len(frontier)), key=lambda i: frontier[i][1][1])
best_labels, best_metrics = frontier[best_intra_idx]

print(best_labels)
print(best_metrics)
```