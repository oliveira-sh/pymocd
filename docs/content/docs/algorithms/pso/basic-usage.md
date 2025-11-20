---
weight: 320
title: "Basic Usage"
icon: menu_book
description: "Learn how to run the PSO community detection algorithm"
lead: "A guide to using PSO for community detection (Coming Soon)"
date: 2025-11-20T00:00:00+00:00
lastmod: 2025-11-20T00:00:00+00:00
draft: false
images: []
toc: true
---

{{% alert context="warning" text="**Coming Soon**: The PSO algorithm is currently under development." /%}}

## Planned Basic Usage

Once implemented, the PSO algorithm will follow a similar API to HP-MOCD:

```python
import networkx as nx
from pymocd import PSO  # Coming soon

G = nx.karate_club_graph()

# PSO instantiation (planned)
alg = PSO(
    graph=G,
    debug_level=1,
    swarm_size=100,
    max_iter=100,
    inertia=0.7,
    cognitive=1.5,
    social=1.5
)

solution = alg.run()
print(solution)
```

## Planned Parameters

| Parameter     | Type               | Description                                                                    |
|---------------|--------------------|--------------------------------------------------------------------------------|
| **graph**     | `networkx.Graph`   | Your unweighted, undirected graph                                              |
| **debug_level** | `u8`             | Verbosity of logging (0-3)                                                     |
| **swarm_size**| `usize`            | Number of particles in the swarm                                               |
| **max_iter**  | `usize`            | Maximum number of iterations                                                   |
| **inertia**   | `f64`              | Inertia weight (controls exploration vs exploitation)                          |
| **cognitive** | `f64`              | Cognitive coefficient (particle's own experience weight)                       |
| **social**    | `f64`              | Social coefficient (swarm's collective knowledge weight)                       |

---

**Status**: 🚧 Under Development

For a working algorithm, see [HP-MOCD Basic Usage](/docs/algorithms/hpmocd/basic-usage/).
