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

`smocc` takes the evolutionary budget as keyword arguments, shown here at its
defaults:

```python
communities = pymocd.smocc(
    G,
    pop_size=100,
    num_gens=100,
    cross_rate=0.7,
    mut_rate=0.5,
    gap=10,
)
```

`gap` is the macro/micro co-evolution interval. `num_gens` is the generation
count: the search always runs all of them. `smocc` also takes `macro_cap` (a
multiplier on the macro population's `ceil(sqrt(n))` community ceiling) and
`micro_mut`.

`mmcomo` shares the same four knobs plus `gap`, at its own paper's defaults
(`pop_size=100`, `num_gens=50`, `cross_rate=0.1`, `mut_rate=0.1`, `gap=10`),
and additionally takes `beta`, the exponent of its dense diffusion-kernel
similarity. `smocc` has no such parameter: it replaces that kernel with a
sparse edge similarity reinforced from the elite consensus.

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

Seven detectors pick one partition from a Pareto front of candidates:
`smocc`, `mopots`, `hpmocd`, `mmcomo`, `ccm`, `krm` and `moga_net`. To see the
whole candidate set, use `smocc_fronts`, `mopots_fronts`, `hpmocd_fronts`,
`mmcomo_fronts`, `ccm_fronts`, `krm_fronts` or `moga_net_fronts`, which accept
the same kwargs as their detector (`smocc_fronts` adds `refine`, `topo_mode`
and `obj_mode`) and return a `list[dict[node, community]]`:

```python
front = pymocd.smocc_fronts(G)
best = max(front, key=lambda p: pymocd.ari(p, gt))
```

`gdpso` and `cdrme` optimize a single scalar, so they have no front;
`mocd_q` and `mocd_d` do not expose theirs.

`mopots` adds [`mopots_ladder`](api/fronts.md#pymocd.mopots_ladder), which
returns the front's convex hull as `(partition, cut, pair, gamma)` tuples in
increasing `gamma` — each partition paired with the Constant Potts resolution
at which it becomes optimal.

See the [fronts API reference](api/fronts.md) for details.
