# Getting started

## Installation

pymocd requires Python **3.10 or newer**. Prebuilt wheels are published for Linux, macOS, and Windows:

```bash
pip install pymocd
```

To build from source you need a Rust toolchain and [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/oliveira-sh/pymocd
cd pymocd
make build
```

## First detection

`pymocd.smocc` is the recommended entry point:

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
communities = pymocd.smocc(G)
```

!!! important "Graph format"
    Every detector accepts a **NetworkX** or **igraph** graph with **integer node ids** and returns a crisp `dict[node, community]`. Isolated nodes are always assigned community `-1`.

## Tuning

`smocc` and `mmcomo` share the same evolutionary knobs:

```python
communities = pymocd.smocc(
    G,
    pop_size=100,
    num_gens=50,
    cross_rate=0.1,
    mut_rate=0.1,
    gap=10,
    beta=0.05,
)
```

`gap` is the macro/micro co-evolution interval and `beta` the micro-stage perturbation strength. `num_gens` is the generation count: the search always runs all of them.

For tuned HP-MOCD runs, use the `HpMocd` class — it exposes custom objectives, a per-generation callback, and the raw Pareto front:

```python
alg = pymocd.HpMocd(G, pop_size=100, num_gens=100, cross_rate=0.7, mut_rate=0.5)
alg.set_on_generation(lambda gen, total, front_size: print(gen, total, front_size))

best = alg.run()
front = alg.generate_pareto_front()
```

`run` returns the max-Q partition from the front; `generate_pareto_front` returns `[(partition, objectives), ...]`.

Custom objectives are callables `(graph, partition) -> float` to minimize, passed via `objectives=` or `set_objectives`; an empty list reverts to the built-in intra/inter pair.

See [Algorithms](algorithms.md) for what each detector optimizes, and the [detector API reference](api/detectors.md) for every signature.

## Threads

All detectors run on a shared Rayon thread pool. To cap it:

```python
pymocd.max_cores(4)
```

!!! note
    The Rayon pool is global and initialized once, so call `max_cores` before the first detection; repeat calls are ignored.

## Evaluating results

When you have ground-truth labels, `gt_metrics` computes four scores at once over the shared nodes of two `{node: community}` dicts:

```python
gt = {node: (0 if G.nodes[node]["club"] == "Mr. Hi" else 1) for node in G}

nmi, ami, ari, f1 = pymocd.gt_metrics(communities, gt)
```

Each metric is also available on its own: `pymocd.nmi`, `pymocd.ami`, `pymocd.ari`, and `pymocd.f1`, all with the same `(partition, gt)` signature. Details in the [metrics API reference](api/metrics.md).

## Inspecting Pareto fronts

`smocc`, `mmcomo`, `ccm`, `krm` and `moga_net` each pick one partition from a Pareto front of candidates. To see the whole candidate set, use `smocc_fronts`, `mmcomo_fronts`, `ccm_fronts`, `krm_fronts` or `moga_net_fronts`, which accept the same evolutionary kwargs as their detector (`smocc_fronts` adds `refine` and `topo_mode`) and return a `list[dict[node, community]]`:

```python
front = pymocd.smocc_fronts(G)
best = max(front, key=lambda p: pymocd.ari(p, gt))
```

See the [fronts API reference](api/fronts.md) for details.
