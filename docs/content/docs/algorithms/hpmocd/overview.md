---
weight: 210
title: "Overview"
description: "A High-Performance Evolutionary Multiobjective Community Detection Algorithm for Large Graphs"
icon: "light"
lead: ""
date: 2025-05-15T02:21:15+00:00
lastmod: 2025-10-15T02:21:15+00:00
draft: false
images: []
aliases:
  - /docs/hpmocd/overview/
  - /hpmocd/overview/
---

The High-Performance Multi-Objective Community Detection (HP-MOCD) algorithm is a scalable evolutionary method designed to efficiently identify high-quality community partitions in large complex networks. HP-MOCD combines the NSGA-II optimization framework with a parallel architecture and topology-aware genetic operators tailored to the structure of real-world graphs. In addition to detailing its core components, we describe the algorithm’s design choices, solution representation, and multi-objective selection strategy. The implementation is written in [Rust](https://www.rust-lang.org/) for performance and exposed to Python via [PyO3](https://pyo3.rs/v0.24.0/). The full source code is publicly available on [GitHub](https://oliveira-sh.github.io/pymocd/).

{{% alert context="success" text="You can read the full pre-print clicking [here](http://arxiv.org/abs/2506.01752)" /%}}

### Overview and Design Rationale

Let us consider a graph `G = (V, E)`, where `V` is the set of nodes and `E` the set of edges. The objective of the **HP-MOCD** is to uncover meaningful community structures by **simultaneously optimizing multiple, often conflicting, structural criteria**.

{{% alert context="info" text="**Note**: The library has only **networkx** or **igraph** support. Compatibility for another libraries should be a great contribution!" /%}}

To achieve this, HP-MOCD is built upon the **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) framework, a well-established method in multi-objective optimization. NSGA-II was chosen due to its strong ability to produce diverse, high-quality Pareto fronts, especially when compared to older algorithms like **PESA-II**, which often struggle with diversity maintenance or selection pressure.

---

### Optimization Strategy

The HP-MOCD algorithm proceeds in two main phases:

1. **Initialization Phase**:
   A population of potential community partitions (called *individuals*) is randomly generated. Each individual is a possible assignment of nodes to communities.

2. **Evolutionary Phase**:
   The population evolves through a number of generations using genetic operators—**selection**, **crossover**, and **mutation**. At each generation, individuals are evaluated, ranked, and filtered to maintain only the most promising solutions.

---


### Objectives and Representation

The optimization targets two structural objectives:

* **Intra-Community Connectivity**
  Measures how densely connected nodes are within each community. This objective is maximized (or its penalty minimized) to encourage cohesive clusters.

* **Inter-Community Separation**
  Measures the extent of connections between different communities. This is minimized to promote structural separation and distinct boundaries.

Together, these form a **multi-objective problem**, where each solution represents a trade-off between internal density and external separation.

---

### Internal Graph Representation

Internally, the graph `G` is stored using a **hash map** (via Rust’s high-performance `rustc-hash`) mapping each node to its neighbor list. This ensures:

* Fast access/modification during evolution
* Efficient computation of objective functions
* Scalability for large graphs

Each individual (solution) is encoded as a mapping from node IDs to community IDs:

```python
{ node_1: community_3, node_2: community_1, ... }
```

This compact representation supports fast mutations and evaluations during the evolutionary cycle.

---

[^1]: HP-MOCD uses [`rustc-hash`](https://github.com/rust-lang/rustc-hash), a fast, deterministic hash function used internally by Rust. It ensures performance without cryptographic guarantees.
