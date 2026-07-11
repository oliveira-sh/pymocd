
<div align="center">
  <img src="res/logo.png" alt="pymocd logo" width="50%">  
  
  <strong>Multi-Objective Community Detection Algorithms</strong>  

[![PyPI Publish](https://github.com/oliveira-sh/pymocd/actions/workflows/release.yml/badge.svg)](https://github.com/oliveira-sh/pymocd/actions/workflows/release.yml)
![Rust Compilation](https://img.shields.io/github/actions/workflow/status/oliveira-sh/pymocd/rust.yml)
![PyPI - Version](https://img.shields.io/pypi/v/pymocd)
![PyPI - License](https://img.shields.io/pypi/l/pymocd)

</div>

**pymocd** is a Python library, powered by a Rust backend, for multi-objective
evolutionary community detection in complex networks. The evolutionary core is
written in Rust and exposed through [PyO3](https://pyo3.rs), giving it a large
speed advantage over pure-Python implementations while staying a drop-in for
the **NetworkX** / **igraph** ecosystem, making it well-suited to large-scale
graphs.

**Read the [Documentation](https://oliveira-sh.github.io/dpymocd/) for detailed
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
communities = pymocd.scale(G)       # -> dict[node, community]
```

> [!IMPORTANT]
> Graphs must be in **NetworkX** or **igraph** compatible format with integer
> node ids. Isolated nodes are assigned community `-1`.

Every detector returns a single crisp partition as `dict[node, community]`.

### Algorithms

`pymocd` ships eight detectors. **SCALE** and **HP-MOCD** are the library's
own contributions; the remaining six are faithful re-implementations of
published baselines (the original authors released no code).

| API | Algorithm | Objectives & engine | Solution selection | Year |
|---|---|---|---|---|
| `scale` | **SCALE** (Santos, in prep.) | KKM / ratio-cut bi-objective, sparse macro–micro co-evolutionary NSGA-II (near-linear, no dense kernel) | label-free **SBM/MDL** description length | 2026 |
| `hpmocd` | **HP-MOCD** ([Santos et al.](https://doi.org/10.1007/s13278-025-01519-7)) | decomposed modularity, parallel NSGA-II | max modularity *Q* | 2025 |
| `mmcomo` | **MMCoMO** ([Zhang et al.](https://ieeexplore.ieee.org/document/10188453)) | kernel *k*-means + ratio cut, macro/micro co-evolutionary NSGA-II | max *Q* (front via `mmcomo_fronts`) | 2023 |
| `ccm` | **CCM** ([Shaik et al.](https://doi.org/10.1007/s42979-020-00382-x)) | score + fitness + modularity, NSGA-III | max *Q* | 2021 |
| `krm` | **KRM** ([Shaik et al.](https://doi.org/10.1007/s42979-020-00382-x)) | kernel *k*-means + ratio cut + modularity, NSGA-III | max *Q* | 2021 |
| `mocd_q` | **Shi-MOCD** ([Shi et al.](https://doi.org/10.1016/j.asoc.2011.10.005)) | decomposed modularity, PESA-II | max *Q* | 2012 |
| `mocd_d` | **Shi-MOCD** ([Shi et al.](https://doi.org/10.1016/j.asoc.2011.10.005)) | decomposed modularity, PESA-II | max-min distance to random nets | 2012 |
| `moga_net` | **MOGA-Net** ([Pizzuti](https://doi.org/10.1109/ICTAI.2009.58)) | community score + fitness, NSGA-II | max *Q* | 2009 |

**SCALE** is the recommended crisp detector: it co-evolves a macro population of
medoid community centres with a micro population of per-node labels over the
kernel *k*-means / ratio-cut bi-objective, bridged by a sparse similarity carried
on the graph's edges rather than a dense *n*×*n* kernel — so memory is *O(n+m)*
and it scales to graphs the dense macro–micro baseline cannot build. The merged
rank-1 front is enriched by a union refinement, and one partition is returned with
no ground truth by minimising a label-free
[microcanonical SBM](https://doi.org/10.1103/PhysRevX.4.011047) description length.

### Usage

```python
import pymocd

# Recommended detectors (defaults work out of the box)
part = pymocd.scale(G)            # SCALE, sparse co-evolution + SBM/MDL selection
part = pymocd.hpmocd(G)           # HP-MOCD

# Baselines (sensible defaults; pop_size / num_gens / rates are tunable kwargs)
part = pymocd.mocd_q(G)           # Shi-MOCD, max-modularity selection
part = pymocd.mocd_d(G)           # Shi-MOCD, max-min-distance selection
part = pymocd.moga_net(G)         # MOGA-Net (Pizzuti)
part = pymocd.ccm(G)              # NSGA-III CCM (Shaik et al.)
part = pymocd.krm(G)              # NSGA-III KRM (Shaik et al.)
part = pymocd.mmcomo(G)           # MMCoMO (Zhang et al.), macro/micro co-evolution

# All return dict[node, community]; isolated nodes -> -1
```

`scale` accepts the same evolutionary kwargs as `mmcomo` (`pop_size=100`,
`num_gens=50`, `cross_rate=0.1`, `mut_rate=0.1`, `gap=10`, `beta=0.05`) plus
`adaptive_stop=False` / `conv_pval=0.1` — with `adaptive_stop=True` the search
self-terminates once a Welch t-test detects a convergence plateau and
`num_gens` becomes only a safety ceiling.

The Pareto frontier of some algorithms is exposed for inspection:

```python
fronts = pymocd.scale_fronts(G)      # list[dict[node, community]]
fronts = pymocd.mmcomo_fronts(G)     # list[dict[node, community]]
```

Helpers:

```python
pymocd.max_cores(8)                  # set Rayon thread pool (first call wins)

# Fast native ground-truth agreement metrics between two {node: community}
# dicts, computed over their shared nodes (exact AMI, matches scikit-learn):
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
