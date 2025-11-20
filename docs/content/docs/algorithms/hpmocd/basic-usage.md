---
weight: 220
title: "Basic Usage"
icon: menu_book
description: "Learn how to run the HpMocd default community detection function"
lead: "A concise guide to setting up and executing HpMocd on your network."
date: 2022-10-19T21:49:38+01:00
lastmod: 2023-08-24T16:34:38+01:00
draft: false
images: []
toc: true
aliases:
  - /docs/hpmocd/basic-usage/
  - /hpmocd/basic-usage/
---

{{% alert context="warning" text="**Note**: Here it is assumed that you have read the quickstarting section and have already installed the library on your operating system, whatever it may be." /%}}

# Basic Usage

To detect communities in a graph using the **HpMocd** algorithm from **pymocd**, follow the steps below.

{{% alert context="danger" text="**Note**: The input graph **must** be unweighted and undirected. It can be an object from `networkx.Graph` or `igraph.Graph`" /%}}

First, prepare your graph:

```python
import networkx as nx       # You can use igraph too
from pymocd import HpMocd   # Our library

G = nx.karate_club_graph()
```

Then, instantiate and run the algorithm:

```python
alg = HpMocd(
    graph=G,
    debug_level=1,
    pop_size=100,
    num_gens=100,
    cross_rate=0.8,
    mut_rate=0.2
)

solution = alg.run()
print(solution)
```

## Parameters

| Parameter     | Type               | Description                                                                    |
|---------------|--------------------|--------------------------------------------------------------------------------|
| **graph**     | `networkx.Graph`   | Your unweighted, undirected graph. Only NetworkX graphs are supported.         |
| **debug_level** | `u8`             | Verbosity of internal logging (0 = no debug output; 1–3 = increasing detail).  |
| **pop_size**  | `usize`            | Population size for the evolutionary algorithm.                                |
| **num_gens**  | `usize`            | Number of generations (iterations) to run.                                     |
| **cross_rate** | `f64`             | Crossover rate, a floating-point value in `[0.0, 1.0]`.                         |
| **mut_rate**  | `f64`              | Mutation rate, a floating-point value in `[0.0, 1.0]`.                          |


---

# Further reading

## Rust Parameter Types

* **`usize`**: An unsigned integer whose size matches the platform’s pointer width (e.g. 64 bits on a 64-bit system).

* **`u8`**: An 8-bit unsigned integer (values 0 to 255).s.

* **`f64`**: A 64-bit (double-precision) floating-point number. Provides high precision for real-valued parameters.
