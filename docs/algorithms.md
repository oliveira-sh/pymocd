# Algorithms

`pymocd` ships nine detectors. **MOPSO**, **SMOCC** and **HP-MOCD** are the
library's own contributions; the remaining six re-implement published baselines
whose authors released no code.

## Overview

| API | Algorithm | Objectives & engine | Solution selection | Year |
|---|---|---|---|---|
| [`mopso`](api/detectors.md#pymocd.mopso) | **MOPSO** (Santos, in prep.) | decomposed Constant Potts Model, particle swarm over an external Pareto archive | resolution plateau of the front | 2026 |
| [`smocc`](api/detectors.md#pymocd.smocc) | **SMOCC** (Santos, in prep.) | heterogeneous intra/inter + KKM/ratio-cut objectives, sparse macro–micro co-evolutionary NSGA-II (near-linear, no dense kernel) | label-free normalised scalarisation | 2026 |
| [`hpmocd`](api/detectors.md#pymocd.hpmocd) | **HP-MOCD** ([Santos et al.](https://doi.org/10.1007/s13278-025-01519-7)) | decomposed modularity, parallel NSGA-II | max modularity *Q* | 2025 |
| [`mmcomo`](api/detectors.md#pymocd.mmcomo) | **MMCoMO** ([Zhang et al.](https://ieeexplore.ieee.org/document/10188453)) | kernel *k*-means + ratio cut, macro/micro co-evolutionary NSGA-II | max *Q* (front via [`mmcomo_fronts`](api/fronts.md#pymocd.mmcomo_fronts)) | 2023 |
| [`ccm`](api/detectors.md#pymocd.ccm) | **CCM** ([Shaik et al.](https://doi.org/10.1007/s42979-020-00382-x)) | score + fitness + modularity, NSGA-III | max *Q* (front via [`ccm_fronts`](api/fronts.md#pymocd.ccm_fronts)) | 2021 |
| [`krm`](api/detectors.md#pymocd.krm) | **KRM** ([Shaik et al.](https://doi.org/10.1007/s42979-020-00382-x)) | kernel *k*-means + ratio cut + modularity, NSGA-III | max *Q* (front via [`krm_fronts`](api/fronts.md#pymocd.krm_fronts)) | 2021 |
| [`mocd_q`](api/detectors.md#pymocd.mocd_q) | **Shi-MOCD** ([Shi et al.](https://doi.org/10.1016/j.asoc.2011.10.005)) | decomposed modularity, PESA-II | max *Q* | 2012 |
| [`mocd_d`](api/detectors.md#pymocd.mocd_d) | **Shi-MOCD** ([Shi et al.](https://doi.org/10.1016/j.asoc.2011.10.005)) | decomposed modularity, PESA-II | max-min distance to random nets | 2012 |
| [`moga_net`](api/detectors.md#pymocd.moga_net) | **MOGA-Net** ([Pizzuti](https://doi.org/10.1109/TEVC.2011.2161090)) | community score + fitness, NSGA-II | max *Q* (front via [`moga_net_fronts`](api/fronts.md#pymocd.moga_net_fronts)) | 2012 |

All detectors return a single crisp partition as `dict[node, community]`;
isolated nodes are assigned community `-1`.

## Which one should I use?

- **[`mopso`](api/detectors.md#pymocd.mopso)** — the fastest of the three and
  deterministic: the same graph always gives the same partition, on any number
  of threads. Returns the whole resolution profile if you ask for it.
- **[`smocc`](api/detectors.md#pymocd.smocc)** — label-free automatic
  selection, near-linear time and memory.
- **[`hpmocd`](api/detectors.md#pymocd.hpmocd)** — the published HP-MOCD
  behaviour with max-modularity selection.
- **The other six** — baselines for papers and benchmarks; `pop_size`,
  `num_gens` and rates are tunable kwargs.

## MOPSO

MOPSO optimises the **Constant Potts Model**, decomposed the way HP-MOCD
decomposes modularity. CPM is

$$H(\gamma) = \sum_c \left[ e_c - \gamma \binom{n_c}{2} \right],$$

and dividing by *m* splits it into two minimised terms,

$$\mathrm{cut} = 1 - \frac{\sum_c e_c}{m}, \qquad
  \mathrm{pair} = \frac{\sum_c \binom{n_c}{2}}{\binom{n}{2}},$$

so that $H(\gamma)/m = 1 - \mathrm{cut} - (\gamma/\gamma_d)\,\mathrm{pair}$
with $\gamma_d$ the graph's density. Every resolution is therefore a weighted
sum of the *same two numbers*, which means the Pareto front of
(`cut`, `pair`) is the graph's complete **resolution profile** — and CPM's
resolution parameter stops being something the caller has to choose. `cut` is
also exactly the realised mixing parameter of the partition.

The search is a discrete multi-objective particle swarm. Each particle carries
a position (one community label per vertex), a velocity (the per-vertex
probability of being unstable this iteration), a personal best, and a niche on
a geometric ladder of resolutions spanning $[1/n^2, 1]$ — the whole range over
which $\gamma$ can change the answer. Particles fly toward their own best and
toward a leader drawn from a bounded external Pareto archive, and refine
themselves with a resolution-directed CPM local move. Because both CPM terms
are integer counts, they are maintained through every single vertex move rather
than recomputed, so an iteration never rescans the graph.

One partition is chosen from the archive with no ground truth by reading it as
the resolution profile it is and keeping the granularity that survives the
widest span of $\gamma$ — the profile's **plateau**. The profile itself is
exposed via [`mopso_fronts`](api/fronts.md#pymocd.mopso_fronts), which also
returns each member's (`cut`, `pair`) point, so you can apply your own rule.

MOPSO is **deterministic**: one independent random stream per (iteration,
particle) and no shared state inside an iteration, so the same graph and
parameters give byte-identical output on any thread count.

## SMOCC

SMOCC (Sparse Multi-Objective Co-evolutionary Community detection) co-evolves
a macro population of medoid community centres optimising kernel *k*-means /
ratio cut with a micro population of per-node labels optimising the
intra/inter modularity decomposition, bridged by a sparse similarity carried
on the graph's edges rather than a dense *n*×*n* kernel — so memory is
*O(n+m)* and it scales to graphs the dense macro–micro baseline cannot build.
The merged rank-1 front is enriched by a union refinement, and one partition
is returned with no ground truth by minimising a min–max-normalised
scalarisation of all four objectives across the front.

The frontier is exposed for inspection via
[`smocc_fronts`](api/fronts.md#pymocd.smocc_fronts).

## HP-MOCD

HP-MOCD optimises decomposed modularity with a parallel NSGA-II and returns
the max-*Q* solution from the Pareto front. The front itself is exposed for
inspection via [`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts).
Published in
[Social Network Analysis and Mining (2025)](https://doi.org/10.1007/s13278-025-01519-7).

## Citation

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
