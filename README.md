<div align="center">
  <img src="res/logo.svg" alt="pymocd logo" width="75%">  
</div>

<div align="center">

[![PyPI Publish](https://github.com/oliveira-sh/pymocd/actions/workflows/release.yml/badge.svg)](https://github.com/oliveira-sh/pymocd/actions/workflows/release.yml)![Rust Compilation](https://img.shields.io/github/actions/workflow/status/oliveira-sh/pymocd/rust.yml)
![PyPI - Version](https://img.shields.io/pypi/v/pymocd)
![PyPI - License](https://img.shields.io/pypi/l/pymocd)

</div>

**pymocd** is a Python library, powered by a Rust backend, for multi-objective
evolutionary community detection in complex networks. The evolutionary core is
written in Rust and exposed through [PyO3](https://pyo3.rs), giving it a large
speed advantage over pure-Python implementations while staying a drop-in for
the **NetworkX** / **igraph** ecosystem, making it well-suited to large-scale
graphs.

**Read the [Documentation](https://pymocd.guiolvr.com/) for detailed
guidance and usage instructions.**

---

### Getting started

```bash
pip install pymocd
```

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()          # any NetworkX / igraph graph, integer node ids
communities = pymocd.rimpso(G)      # -> dict[node, community]
```

> [!IMPORTANT]
> Graphs must be in **NetworkX** or **igraph** compatible format with integer
> node ids. Isolated nodes are assigned community `-1`.

Every detector returns a single crisp partition as `dict[node, community]`.

### Algorithms

`pymocd` exposes **nine community-detection algorithms** through **ten
detector entry points** (Shi-MOCD ships under two selection rules).

| API | Source | Objectives | Engine | Decision Making | Code |
|---|---|---|---|---|---|
| `rimpso` | Santos, *in prep.* (2026) | Constant Potts split: cut fraction + pair coverage | memetic MOPSO over a resolution ladder | best-fit assortative block model | this library |
| `hpmocd` | [Santos et al., *SNAM* 2025](https://doi.org/10.1007/s13278-025-01519-7) | decomposed modularity (intra, inter) | NSGA-II | max *Q* | this library |
| `cdrme` | [Dabaghi-Zarandi et al., *JNCA* 2025](https://doi.org/10.1016/j.jnca.2024.104070) | Eq. 12 linkage scalar (single) | random walks + agglomerative merge | max *Q* | [private, vendored here](res/original_algs/cdrme) |
| `mmcomo` | [Zhang et al., *IEEE CIM* 2023](https://ieeexplore.ieee.org/document/10188453) | kernel *k*-means + ratio cut | macro/micro co-evolutionary NSGA-II | max *Q* | — |
| `ccm` | [Shaik et al., *SN Comp. Sci.* 2021](https://doi.org/10.1007/s42979-020-00382-x) | community score + fitness + modularity | NSGA-III | max *Q* | — |
| `krm` | [Shaik et al., *SN Comp. Sci.* 2021](https://doi.org/10.1007/s42979-020-00382-x) | kernel *k*-means + ratio cut + modularity | NSGA-III | max *Q* | — |
| `gdpso` | [Cai et al., *Inf. Sciences* 2015](https://doi.org/10.1016/j.ins.2014.09.041) | Newman–Girvan modularity (single) | greedy discrete PSO | swarm best | [doctor-cai/GDPSO](https://github.com/doctor-cai/GDPSO) |
| `mocd_q` | [Shi et al., *Applied Soft Computing* 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity | PESA-II | max *Q* | — |
| `mocd_d` | [Shi et al., *Applied Soft Computing* 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity | PESA-II | max–min random-graph distance | — |
| `moga_net` | [Pizzuti, *IEEE TEC* 2012](https://doi.org/10.1109/TEVC.2011.2161090) | community score + fitness | NSGA-II | max *Q* | [Moganet2016.zip](https://staff.icar.cnr.it/pizzuti/codice/Moganet2016.zip) |

Each detector has a module README with its full derivation, its parameter
table and the list of every deliberate divergence from its paper:
[`rimpso`](src/core/algorithms/rimpso/README.md) ·
[`hpmocd`](src/core/algorithms/hpmocd/README.md) ·
[`cdrme`](src/core/algorithms/cdrme/README.md) ·
[`mmcomo`](src/core/algorithms/mmcomo/README.md) ·
[`ccm`](src/core/algorithms/ccm/README.md) ·
[`krm`](src/core/algorithms/krm/README.md) ·
[`gdpso`](src/core/algorithms/gdpso/README.md) ·
[`mocd`](src/core/algorithms/mocd/README.md) (Shi-MOCD) ·
[`moganet`](src/core/algorithms/moganet/README.md) —
[index](src/core/algorithms/README.md).

### Usage

```python
import pymocd

# This library's own detectors
part = pymocd.rimpso(G)           # RIMPSO         (recommended default)
part = pymocd.hpmocd(G)           # HP-MOCD

# Re-implemented baselines
part = pymocd.cdrme(G)            # CDRME     (Dabaghi-Zarandi et al.)
part = pymocd.mmcomo(G)           # MMCoMO    (Zhang et al.)
part = pymocd.ccm(G)              # CCM       (Shaik et al., NSGA-III)
part = pymocd.krm(G)              # KRM       (Shaik et al., NSGA-III)
part = pymocd.gdpso(G)            # GDPSO     (Cai et al., particle swarm)
part = pymocd.mocd_q(G)           # Shi-MOCD, max-modularity selection
part = pymocd.mocd_d(G)           # Shi-MOCD, max-min-distance selection
part = pymocd.moga_net(G)         # MOGA-Net  (Pizzuti)

# All return dict[node, community]; isolated nodes -> -1
```

Every detector except `hpmocd` takes its budget as keyword arguments. The
values below are the shipped defaults, which follow each paper wherever the
paper states them:

```python
pymocd.rimpso(G, pop_size=100, num_gens=100, inertia=0.4, cognitive=0.7, social=0.7,
              local_rate=0.35, archive=100, ls_period=10, seed=0)
pymocd.mmcomo(G, pop_size=100, num_gens=50, cross_rate=0.1, mut_rate=0.1, gap=10, beta=0.05)
pymocd.ccm(G,    pop_size=200, num_gens=100, cross_rate=0.8, mut_rate=1/68, r=1.0, alpha=1.0, divisions=12)
pymocd.krm(G,    pop_size=100, num_gens=100, cross_rate=0.8, mut_rate=1/34, divisions=12)
pymocd.moga_net(G, pop_size=300, num_gens=30, cross_rate=0.8, mut_rate=0.2, r=2.0, alpha=1.0)
pymocd.gdpso(G,  pop_size=100, num_gens=250, w=0.7298, c1=1.4961, c2=1.4961,
             mut_rate=0.1, mut_frac=0.1, lpa_sweeps=5)
pymocd.cdrme(G,  alpha_walk=1.0, n_walk=50, pop_size=300, elite_size=300,
             alpha_mut=0.5, mut_sweeps=10)
pymocd.mocd_q(G, pop_size=100, num_gens=100, cross_rate=0.9, mut_rate=0.1)
pymocd.mocd_d(G, pop_size=100, num_gens=100, cross_rate=0.9, mut_rate=0.1, rand_networks=3)
```

The one exception is Shi-MOCD: `mocd_q` and `mocd_d` default to this repo's
HP-MOCD-parity benchmark budget, **not** Shi's published configuration, which
is `cross_rate=0.6`, `mut_rate=0.4` with per-graph population and generation
counts from the paper's Table 1. Pass those explicitly to reproduce the paper.

`hpmocd(G)` takes the graph only and runs at its published configuration; for
a tunable HP-MOCD, or to plug in your own Python objective functions, use the
`pymocd.HpMocd` class instead.

#### Pareto fronts

Seven of the eleven entry points expose the candidate set their selection
rule picks from:

```python
fronts = pymocd.rimpso_fronts(G)    # list[dict[node, community]]
fronts = pymocd.hpmocd_fronts(G)
fronts = pymocd.mmcomo_fronts(G)
fronts = pymocd.ccm_fronts(G)
fronts = pymocd.krm_fronts(G)
fronts = pymocd.moga_net_fronts(G)

# RIMPSO only: run its label-free selection rule over partitions produced
# elsewhere, which separates the search's contribution from the selector's
pick, points = pymocd.rimpso_select(G, candidates)
```

Those are *exactly* the detectors with a front accessor. `gdpso` and `cdrme`
are single-objective, and `mocd_q` / `mocd_d` do not expose theirs, so there is
no `gdpso_fronts`, `cdrme_fronts` or `mocd_fronts`.

`pymocd.mr_mocd`, `pymocd.mr_mocd_fronts` and `pymocd.mr_mocd_select` are the
names this detector carried before it was renamed to RIMPSO; `pymocd.scale` and
`pymocd.scale_fronts` are older still. All five are deprecated aliases for the
same function objects as `rimpso`, `rimpso_fronts` and `rimpso_select`, kept so
pinned callers keep working. They emit no warning and do not appear in the type
stubs. Use the new names.

### Helpers

```python
pymocd.max_cores(8)                  # set Rayon thread pool (first call wins)

# Fast native ground-truth agreement metrics between two {node: community}
# dicts, computed over their shared nodes
nmi, ami, ari, f1 = pymocd.gt_metrics(partition, gt)
pymocd.nmi(partition, gt)            # or each metric individually
pymocd.ami(partition, gt)
pymocd.ari(partition, gt)
pymocd.f1(partition, gt)             # pairwise F1
```

### Contributing

Contributions are welcome, open an issue or a pull request for features, bug
fixes, or improvements. This project is licensed under **GPL-3.0 or later**.

---

### Citation

If you use any algorithm in your research, please cite:

```bibtex
@article{Santos2025,
  author    = {Santos, Guilherme O. and Vieira, Lucas S. and Rossetti, Giulio and Ferreira, Carlos H. G. and Moreira, Gladston J. P.},
  title     = {A high-performance evolutionary multiobjective community detection algorithm},
  journal   = {Social Network Analysis and Mining},
  year      = {2025},
  volume    = {15},
  number    = {1},
  pages     = {110},
  doi       = {10.1007/s13278-025-01519-7},
  url       = {https://doi.org/10.1007/s13278-025-01519-7},
  issn      = {1869-5469},
  date      = {2025-11-18}
}
```
