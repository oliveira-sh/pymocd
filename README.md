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
the **NetworkX** / **igraph** ecosystem — making it well-suited to large-scale
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
communities = pymocd.ariadne(G)     # -> dict[node, community]
```

> [!IMPORTANT]
> Graphs must be in **NetworkX** or **igraph** compatible format with integer
> node ids. Isolated nodes are assigned community `-1`.

Every detector returns a single crisp partition as `dict[node, community]`.

### Algorithms

`pymocd` ships seven detectors. **Ariadne** and **HP-MOCD** are the library's
own contributions; the remaining four are faithful re-implementations of
published baselines on a shared NSGA-II / NSGA-III backbone (the original
authors released no code).

| API | Algorithm | Objectives — engine | Solution selection | Params |
|---|---|---|---|---|
| `ariadne` | **Ariadne** — Santos, 2026 *(in preparation)* | bi-objective CPM — NSGA-II, per-graph auto-γ islands | label-free **SBM/MDL** description length | none |
| `hpmocd` | **HP-MOCD** — [Santos et al., 2025](https://doi.org/10.1007/s13278-025-01519-7) | decomposed modularity — parallel NSGA-II | max modularity *Q* | none |
| `mocd_q` | **Shi-MOCD** — [Shi et al., 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity — PESA-II | max *Q* | optional |
| `mocd_d` | **Shi-MOCD** — [Shi et al., 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity — PESA-II | max–min distance to random nets | optional |
| `moga_net` | **MOGA-Net** — [Pizzuti, 2009](https://doi.org/10.1109/ICTAI.2009.58) | community score + fitness — NSGA-II | max *Q* | optional |
| `ccm` | **CCM** — [Shaik et al., 2021](https://doi.org/10.1007/s42979-020-00382-x) | score + fitness + modularity — NSGA-III | max *Q* | optional |
| `krm` | **KRM** — [Shaik et al., 2021](https://doi.org/10.1007/s42979-020-00382-x) | kernel *k*-means + ratio cut + modularity — NSGA-III | max *Q* | optional |

**Ariadne** is the recommended crisp detector: parameter-free and
self-terminating. It probes graph density to bracket five resolution values
(γ), evolves one NSGA-II island per γ, pools the rank-1 fronts, and returns the
single partition of minimum [microcanonical SBM](https://doi.org/10.1103/PhysRevX.4.011047)
description length — a label-free model-selection step that recovers the
front's best partition without ground truth.

### Usage

```python
import pymocd

# Parameter-free detectors
part = pymocd.ariadne(G)          # Ariadne  — auto-γ + SBM/MDL selection
part = pymocd.hpmocd(G)           # HP-MOCD

# Baselines (sensible defaults; pop_size / num_gens / rates are tunable kwargs)
part = pymocd.mocd_q(G)           # Shi-MOCD, max-modularity selection
part = pymocd.mocd_d(G)           # Shi-MOCD, max–min-distance selection
part = pymocd.moga_net(G)         # MOGA-Net (Pizzuti)
part = pymocd.ccm(G)              # NSGA-III CCM (Shaik et al.)
part = pymocd.krm(G)              # NSGA-III KRM (Shaik et al.)

# All return dict[node, community]; isolated nodes -> -1
```

The frontier Ariadne selects *from* is exposed for inspection:

```python
fronts = pymocd.ariadne_fronts(G)   # list[dict[node, community]]
```

HP-MOCD is also available as a class when you want the full Pareto front or
explicit control:

```python
alg   = pymocd.HpMocd(G)
part  = alg.run()                   # selected partition
front = alg.generate_pareto_front()
```

Helpers:

```python
q = pymocd.fitness(G, part)         # modularity Q (Shi 2012)
pymocd.set_thread_count(8)          # set Rayon thread pool (first call wins)
```

### Contributing

Contributions are welcome — open an issue or a pull request for features, bug
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
