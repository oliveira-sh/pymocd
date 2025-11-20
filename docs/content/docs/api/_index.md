---
weight: 500
title: "API Reference"
description: "Complete API documentation for PyMOCD classes and functions"
icon: "code"
lead: "Technical reference for all PyMOCD components"
date: 2025-11-20T00:00:00+00:00
lastmod: 2025-11-20T00:00:00+00:00
draft: false
images: []
---

Complete technical documentation for all PyMOCD classes, methods, and utilities. This reference is organized by component and includes detailed type signatures, parameter descriptions, return values, and usage examples.

---

## Available APIs

### Algorithms

- **[HpMocd Class](/docs/api/hpmocd-class/)** - Complete API for HP-MOCD algorithm
- **PSO Class** - Coming soon

### Utilities

- **[set_thread_count()](/docs/api/utilities/)** - Control thread pool size
- **Graph Converters** - NetworkX/igraph compatibility helpers

---

## Import Structure

```python
# Main algorithm classes
from pymocd import HpMocd  # Available now
from pymocd import PSO     # Coming soon

# Utility functions
from pymocd import set_thread_count
```

---

## Type Conventions

PyMOCD uses Rust types internally, exposed to Python via PyO3:

| Python Type | Rust Type | Description |
|-------------|-----------|-------------|
| `int` | `usize` | Unsigned platform-sized integer |
| `int` | `u8`, `u16`, `u32`, `u64` | Unsigned integers (8-64 bits) |
| `float` | `f64` | 64-bit floating point |
| `dict[int, int]` | `HashMap<usize, usize>` | Node → Community mapping |
| `tuple[float, float]` | `(f64, f64)` | Objective values |
| `networkx.Graph` | Graph representation | Converted internally |

---

Navigate to specific API pages for detailed documentation.
