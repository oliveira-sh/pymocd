---
weight: 1000
title: "HPMOCD"
description: "A High-Performance Evolutionary Multiobjective Community Detection Algorithm for Large Graphs"
icon: "light"
lead: ""
date: 2025-05-15T02:21:15+00:00
lastmod: 2025-10-15T02:21:15+00:00
draft: false
images: []
---

The High-Performance Multi-Objective Community Detection (HP-MOCD) algorithm is a scalable evolutionary method designed to efficiently identify high-quality community partitions in large complex networks. HP-MOCD combines the NSGA-II optimization framework with a parallel architecture and topology-aware genetic operators tailored to the structure of real-world graphs. In addition to detailing its core components, we describe the algorithm’s design choices, solution representation, and multi-objective selection strategy. The implementation is written in [Rust](https://www.rust-lang.org/) for performance and exposed to Python via [PyO3](https://pyo3.rs/v0.24.0/). The full source code is publicly available on [GitHub](https://oliveira-sh.github.io/pymocd/).