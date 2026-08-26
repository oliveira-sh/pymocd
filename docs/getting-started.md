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

`pymocd.mr_mocd` is the recommended entry point:

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
communities = pymocd.mr_mocd(G)
```

!!! important "Graph format"
    Every detector accepts a **NetworkX** or **igraph** graph with **integer node ids** and returns a crisp `dict[node, community]`. Isolated nodes are always assigned community `-1`.

## Tuning

`mr_mocd` takes its budget as keyword arguments, shown here at its defaults:

```python
communities = pymocd.mr_mocd(
    G,
    pop_size=100,
    num_gens=100,
    inertia=0.4,
    cognitive=0.7,
    social=0.7,
    local_rate=0.35,
    archive=100,
    ls_period=10,
)
```

`num_gens` is the generation count: the search always runs all of them.
`inertia`, `cognitive` and `social` are the swarm's three velocity terms;
`local_rate` is the per-node rate of the resolution-directed local move;
`archive` is the capacity of the external Pareto archive, one slot per
particle so it holds the whole profile; and `ls_period` is how often the full
local search runs. **Resolution is not a parameter** — a single run covers the
whole ladder.

`mmcomo` takes a different four knobs plus `gap` and `beta`, at its own
paper's defaults (`pop_size=100`, `num_gens=50`, `cross_rate=0.1`,
`mut_rate=0.1`, `gap=10`).

Every other detector takes its own paper's parameters as keyword arguments —
`r` and `alpha` for `moga_net` and `ccm`, `divisions` for `ccm` and `krm`, `w`
/ `c1` / `c2` / `lpa_sweeps` for `gdpso`, `n_walk` / `alpha_mut` /
`mut_sweeps` for `cdrme`, `rand_networks` for `mocd_d`. The
[detector API reference](api/detectors.md) lists every signature with its
default.

`hpmocd` is the exception: it takes the graph and nothing else, running at its
published configuration and returning the max-*Q* partition from its Pareto
front (the front itself is available via
[`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts)). To vary its budget, or
to plug in your own Python objective functions, use the `pymocd.HpMocd` class:

```python
detector = pymocd.HpMocd(G, pop_size=200, num_gens=150)
communities = detector.run()
front = detector.generate_pareto_front()   # [(partition, objectives), ...]
```

See [Algorithms](algorithms.md) for what each detector optimizes, which paper
it comes from, and whether its original authors released code.

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

Six detectors pick one partition from a Pareto front of candidates:
`mr_mocd`, `hpmocd`, `mmcomo`, `ccm`, `krm` and `moga_net`. To see the whole
candidate set, use `mr_mocd_fronts`, `hpmocd_fronts`, `mmcomo_fronts`,
`ccm_fronts`, `krm_fronts` or `moga_net_fronts`, which accept the same kwargs
as their detector and return a `list[dict[node, community]]`:

```python
front, points, selected = pymocd.mr_mocd_fronts(G)
best = max(front, key=lambda p: pymocd.ari(p, gt))
```

`gdpso` and `cdrme` optimize a single scalar, so they have no front;
`mocd_q` and `mocd_d` do not expose theirs.

`mr_mocd_fronts` returns `(partitions, points, selected)`: every member, its
`(cut, pair)` point, and the index the selector picked.
[`mr_mocd_select`](api/fronts.md#pymocd.mr_mocd_select) runs that selection
chain alone over partitions produced elsewhere.

See the [fronts API reference](api/fronts.md) for details.
