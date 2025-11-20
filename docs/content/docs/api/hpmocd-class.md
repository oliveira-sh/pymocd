---
weight: 510
title: "HpMocd Class"
description: "Complete API reference for the HpMocd class"
icon: "api"
lead: "Detailed documentation for HP-MOCD algorithm implementation"
date: 2025-11-20T00:00:00+00:00
lastmod: 2025-11-20T00:00:00+00:00
draft: false
images: []
toc: true
---

## Class: `HpMocd`

High-Performance Multi-Objective Community Detection algorithm based on NSGA-II evolutionary optimization.

### Import

```python
from pymocd import HpMocd
```

---

## Constructor

```python
HpMocd(graph, 
      debug_level=0, 
      pop_size=100, 
      num_gens=100, 
      cross_rate=0.8, 
      mut_rate=0.2
)
```

Creates a new HP-MOCD algorithm instance.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **graph** | `networkx.Graph` or `igraph.Graph` | *required* | Unweighted, undirected input graph |
| **debug_level** | `int` (u8) | `0` | Logging verbosity: `0` (silent), `1` (basic), `2` (detailed), `3` (verbose) |
| **pop_size** | `int` (usize) | `100` | Population size for evolutionary algorithm |
| **num_gens** | `int` (usize) | `100` | Number of generations (iterations) to evolve |
| **cross_rate** | `float` (f64) | `0.8` | Crossover probability ∈ [0.0, 1.0] |
| **mut_rate** | `float` (f64) | `0.2` | Mutation probability ∈ [0.0, 1.0] |

#### Returns

`HpMocd` instance ready to execute community detection.

#### Raises

- **`ValueError`**: If graph is weighted or directed
- **`ValueError`**: If rates are outside [0.0, 1.0]
- **`TypeError`**: If graph is not NetworkX or igraph compatible

---

## Methods

#### `run()`

Executes the HP-MOCD algorithm and returns the best community partition found.

#### Signature

```python
def run(self) -> dict[int, int]
```

#### Returns

`dict[int, int]`: Node-to-community assignment mapping.
- **Keys**: Node IDs (int)
- **Values**: Community IDs (int)

---

#### `generate_pareto_front()`

Generates and returns the complete Pareto front of non-dominated solutions.

#### Signature

```python
def generate_pareto_front(self) -> list[tuple[dict[int, int], tuple[float, float]]]
```

#### Returns

`list[tuple[dict[int, int], tuple[float, float]]]`: List of Pareto-optimal solutions.

Each tuple contains:
1. **Assignment** (`dict[int, int]`): Node → Community mapping
2. **Metrics** (`tuple[float, float]`): Objective values
   - `[0]` = Inter-community connectivity (minimize)
   - `[1]` = Intra-community density (maximize)

#### Notes

- All solutions are **non-dominated** (Pareto-optimal)
- Frontier size varies (typically 10-50 solutions)
- Solutions represent trade-offs between objectives
- Use selection strategies to choose optimal solution

---

## Attributes

#### `graph`

**Type**: Internal Rust representation

The input graph converted to Rust's hash map structure. Not directly accessible from Python.

#### `debug_level`

**Type**: `int` (u8)

Current logging verbosity level.

#### `pop_size`

**Type**: `int` (usize)

Population size used in evolutionary algorithm.

#### `num_gens`

**Type**: `int` (usize)

Number of generations to evolve.

#### `cross_rate`

**Type**: `float` (f64)

Crossover probability.

#### `mut_rate`

**Type**: `float` (f64)

Mutation probability.

---

### Parallelization

HP-MOCD uses Rayon's work-stealing thread pool for parallel fitness evaluation:
- Evaluates multiple individuals simultaneously
- Scales efficiently with CPU cores
- Configure threads via `pymocd.set_thread_count(n)`

---

## Related

- [Basic Usage Guide](/docs/algorithms/hpmocd/basic-usage/)
- [Pareto Front Generation](/docs/algorithms/hpmocd/pareto/)
- [Selection Strategies](/docs/algorithms/hpmocd/selection-strategies/)
- [Performance Tuning](/docs/guides/performance-tuning/)
