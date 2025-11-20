---
weight: 100
draft: false
title: "Documentation"
icon: "book"
toc: true
description: "Complete documentation for PyMOCD - Multi-Objective Community Detection in Python"
---

# PyMOCD Documentation

Welcome to the **PyMOCD** documentation. PyMOCD is a high-performance Python library for multi-objective community detection in complex networks, implemented in Rust for maximum efficiency.

---

## Quick Start

New to PyMOCD? Start here:

1. **[Installation & Setup](/docs/quickstart/)** - Install PyMOCD and set up your environment
2. **[Choose an Algorithm](/docs/algorithms/)** - Understand HP-MOCD vs PSO and select the right one
3. **[Basic Usage](/docs/algorithms/hpmocd/basic-usage/)** - Run your first community detection
4. **[Generate Pareto Fronts](/docs/algorithms/hpmocd/pareto/)** - Explore multi-objective solutions

---

## Documentation Structure

### 🚀 Getting Started

- **[Quickstart](/docs/quickstart/)** - System requirements, installation (pip/source), and basic setup

### 🧬 Algorithms

Multi-objective community detection algorithms with different optimization approaches:

- **[Algorithm Overview & Comparison](/docs/algorithms/)** - Decision framework and feature comparison
- **[HP-MOCD](/docs/algorithms/hpmocd/)** - Evolutionary algorithm (NSGA-II) - **Available Now**
- **[PSO](/docs/algorithms/pso/)** - Particle Swarm Optimization - **Coming Soon**

### 📚 Guides

Cross-algorithm guides and best practices:

- **[Choosing an Algorithm](/docs/guides/)** - Decision framework for algorithm selection
- **Performance Tuning** - Optimization for large networks
- **Benchmarking** - Evaluation methods and metrics
- **Understanding Pareto Fronts** - Multi-objective optimization theory

### 📖 API Reference

Complete technical reference:

- **[HpMocd Class](/docs/api/)** - Full API documentation for HP-MOCD
- **PSO Class** - Coming soon
- **Utilities** - Helper functions and configuration

### 💡 Examples

Practical tutorials and use cases:

- **[Karate Club Network](/docs/examples/)** - Classic introductory example
- **Comparing Algorithms** - Side-by-side HP-MOCD vs PSO
- **LFR Benchmark** - Synthetic network analysis

### ⚙️ Library Configuration

- **[Threading Control](/docs/optional-options/limiting-usage/)** - Manage CPU utilization
- **Logging & Debug Levels** - Control output verbosity

### 🤝 Contributing

- **[How to Contribute](/docs/contributing/how-to-contribute/)** - Development workflow
- **[Code of Conduct](/docs/contributing/code-of-conduct/)** - Community guidelines
- **[Financial Support](/docs/contributing/financial-contributions/)** - Donate to the project

---

## Key Features

- **🚄 High Performance**: Rust implementation with full parallelization
- **📊 Multi-Objective**: Simultaneous optimization of multiple community quality metrics
- **🎯 Pareto Fronts**: Explore trade-offs between competing objectives
- **🔬 Academic Rigor**: Published algorithms with peer-reviewed foundations
- **🐍 Pythonic API**: Simple, intuitive interface for Python developers
- **📈 Scalable**: Handles networks with millions of nodes and edges

---

## Example Usage

```python
import networkx as nx
from pymocd import HpMocd

# Load your network
G = nx.karate_club_graph()

# Run community detection
alg = HpMocd(
    graph=G,
    pop_size=100,
    num_gens=100,
    cross_rate=0.8,
    mut_rate=0.2
)

# Get best solution
solution = alg.run()

# Or explore Pareto front
frontier = alg.generate_pareto_front()
```

---

## Need Help?

- **📖 Read the Docs**: You're in the right place! Use the navigation on the left
- **🐛 Report Issues**: [GitHub Issues](https://github.com/oliveira-sh/pymocd/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/oliveira-sh/pymocd/discussions)
- **📧 Contact**: Check the repository for contact information

---

## Citation

If PyMOCD helps your research, please cite:

**HP-MOCD**: Santos et al. (2025) - [View BibTeX](/docs/algorithms/hpmocd/citation/)

{{% alert context="success" text="PyMOCD is **open source** and **actively maintained**. Contributions are welcome!" /%}}
