# Custom objectives

Replace the built-in objectives with custom Python functions to optimize for domain-specific community quality metrics.

## Overview

By default, `HpMocd` optimizes two built-in objectives implemented in Rust: **intra-community** and **inter-community** cost (both minimized). These are evaluated in parallel and cover the most common use case.

You can **replace** these defaults with your own Python objective functions via the `objectives` parameter. This lets you optimize for any community quality metric — modularity, conductance, motif-based measures, or anything else you can express as a function. You can pass **any number** of objectives; the algorithm uses NSGA-II to find Pareto-optimal trade-offs across all of them.

## Writing an objective function

Each objective must follow this signature:

```python
def objective(graph, partition: dict[int, int]) -> float:
    ...
```

| Argument      | Type                       | Description                                               |
|---------------|----------------------------|-----------------------------------------------------------|
| **graph**     | Your original graph object | The same graph you passed to `HpMocd`                     |
| **partition** | `dict[int, int]`           | Maps each node ID to its community ID                     |
| **return**    | `float`                    | The objective value — **minimized** by the algorithm      |

!!! warning
    All objectives are **minimized**. If you want to maximize a metric (e.g. modularity), return its negation or `1.0 - value`.

## Simple example

A single objective that maximizes Newman modularity:

```python
import networkx as nx
from pymocd import HpMocd

def negative_modularity(G, partition):
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    return -nx.community.modularity(G, communities.values())

G = nx.karate_club_graph()
alg = HpMocd(G, objectives=[negative_modularity], pop_size=50, num_gens=50)
solution = alg.run()
```

Multiple objectives work the same way:

```python
alg = HpMocd(G, objectives=[obj_1, obj_2, obj_3])
```

## Factory pattern for performance

Each objective is called once **per individual, per generation** — that is `pop_size x num_gens` times. Recomputing graph properties inside the function body every time is wasteful.

The factory pattern solves this: a function receives the graph once, precomputes constants, and returns a fast closure:

```python
import numpy as np
import scipy.sparse as sp

def make_conductance(G):
    # Precompute once (called at construction time)
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    src = [idx[u] for u, v in G.edges()] + [idx[v] for u, v in G.edges()]
    dst = [idx[v] for u, v in G.edges()] + [idx[u] for u, v in G.edges()]
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    degrees = np.asarray(A.sum(axis=1)).ravel()
    total_vol = degrees.sum()
    rows, cols = A.nonzero()

    # Fast closure (called pop_size x num_gens times)
    def _obj(_G, partition):
        if total_vol == 0:
            return 0.0
        labels = np.array([partition[v] for v in nodes], dtype=np.int32)
        n_comms = labels.max() + 1
        is_cut = (labels[rows] != labels[cols]).astype(np.float64)
        cut_comm = np.bincount(labels[rows], weights=is_cut, minlength=n_comms)
        vol_comm = np.bincount(labels, weights=degrees, minlength=n_comms)
        vol_comp = total_vol - vol_comm
        denom = np.minimum(vol_comm, vol_comp)
        mask = denom > 0
        return float((cut_comm[mask] / denom[mask]).mean()) if mask.any() else 0.0

    return _obj
```

The split matters: everything above the closure runs **once**, when you call `make_conductance(G)` at construction time — building the sparse adjacency matrix, degree vector, and edge index arrays. The returned `_obj` closure is what `HpMocd` calls thousands of times, and it only does cheap vectorized work on the precomputed arrays.

```python
alg = HpMocd(
    G,
    objectives=[make_conductance(G)],
    pop_size=50,
    num_gens=50,
)
solution = alg.run()
```

## Changing objectives after construction

Use `set_objectives()` to swap objectives on an existing instance. Pass an empty list to revert to the built-in Rust objectives:

```python
alg = HpMocd(G)

# Switch to custom objectives
alg.set_objectives([make_conductance(G)])

# Revert to built-in Rust objectives
alg.set_objectives([])
```

## Progress tracking

Register a callback with `set_on_generation()` to monitor the evolutionary process. The callback is invoked after every generation:

```python
from tqdm import tqdm

alg = HpMocd(G, objectives=[make_conductance(G)], pop_size=50, num_gens=50)

bar = tqdm(total=alg.num_gens, desc="HpMocd", unit="gen")

def on_gen(generation, num_gens, front_size):
    bar.set_postfix(front=front_size)
    bar.update(1)
    if generation == num_gens - 1:
        bar.close()

alg.set_on_generation(on_gen)
solution = alg.run()
```

Pass `None` to `set_on_generation()` to clear the callback.

| Argument        | Type  | Description                                    |
|-----------------|-------|------------------------------------------------|
| **generation**  | `int` | Current generation (0-indexed)                 |
| **num_gens**    | `int` | Total number of generations                    |
| **front_size**  | `int` | Number of solutions in the first Pareto front  |

## Performance considerations

!!! danger "Python objectives are slow"
    Python objectives are evaluated **sequentially under the GIL** — unlike the built-in Rust objectives, which run in parallel. Expect significantly longer runtimes.

Recommendations:

- **Reduce the evolutionary budget**: use a smaller `pop_size` and `num_gens` (e.g. `50/50` instead of the default `100/100`).
- **Precompute graph constants**: use the [factory pattern](#factory-pattern-for-performance) to avoid redundant work inside the objective closure.
- **Use vectorized operations**: leverage `numpy` and `scipy.sparse` instead of Python loops — sparse matmul and `np.bincount` run in C-level code.
- **Prefer the built-ins when they suffice**: only use custom objectives when you need a metric the default objectives do not capture.
