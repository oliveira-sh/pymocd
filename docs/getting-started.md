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

`pymocd.gmocs` is the recommended entry point on machines with an NVIDIA GPU; `pymocd.hpmocd` is the CPU-only recommendation:

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
communities = pymocd.gmocs(G)
```

!!! important "Graph format"
    Every detector accepts a **NetworkX** or **igraph** graph with **integer node ids** and returns a crisp `dict[node, community]`. Isolated nodes are always assigned community `-1`.

!!! important "GPU requirement"
    `gmocs` runs its search as CUDA kernels and requires an NVIDIA GPU (Pascal or newer); it raises `RuntimeError` when no usable CUDA device is present. It is also stochastic by design: repeated runs return different partitions.

## Tuning

`gmocs` exposes the swarm knobs directly:

```python
communities = pymocd.gmocs(
    G,
    pop_size=100,
    num_gens=50,
    gap=10,
    turb=0.1,
    macro_cap=1.0,
)
```

`gap` is the macro/micro co-evolution interval, `turb` the first-generation per-node turbulence probability (it decays with the inertia weight), and `macro_cap` a multiplier on the macro swarm's community-centre ceiling. `num_gens` is the generation count: the search always runs all of them.

`hpmocd` runs with its published defaults and returns the max-*Q* partition from its Pareto front; the front itself is available via [`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts).

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

`gmocs`, `mmcomo`, `ccm`, `krm` and `moga_net` each pick one partition from a Pareto front of candidates. To see the whole candidate set, use `gmocs_fronts`, `mmcomo_fronts`, `ccm_fronts`, `krm_fronts` or `moga_net_fronts`, which accept the same evolutionary kwargs as their detector (`gmocs_fronts` adds `refine` and `obj_mode`) and return a `list[dict[node, community]]`:

```python
front = pymocd.gmocs_fronts(G)
best = max(front, key=lambda p: pymocd.ari(p, gt))
```

See the [fronts API reference](api/fronts.md) for details.
