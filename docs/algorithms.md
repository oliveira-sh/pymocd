# Algorithms

`pymocd` ships eight detectors. **GMOCS** and **HP-MOCD** are the library's
own contributions; the remaining six re-implement published baselines whose
authors released no code.

## Overview

| API | Algorithm | Objectives & engine | Solution selection | Year |
|---|---|---|---|---|
| [`gmocs`](api/detectors.md#pymocd.gmocs) | **GMOCS** (Santos, in prep.) | heterogeneous intra/inter + KKM/ratio-cut objectives, GPU-accelerated macro–micro co-evolutionary multi-objective particle swarms (CUDA, requires an NVIDIA GPU) | label-free normalised scalarisation | 2026 |
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

- **[`gmocs`](api/detectors.md#pymocd.gmocs)** — the recommended default on
  machines with an NVIDIA GPU: label-free automatic selection, near-linear
  memory, CUDA-accelerated search.
- **[`hpmocd`](api/detectors.md#pymocd.hpmocd)** — the published HP-MOCD
  behaviour with max-modularity selection.
- **The other six** — baselines for papers and benchmarks; `pop_size`,
  `num_gens` and rates are tunable kwargs.

## GMOCS

GMOCS (GPU-accelerated Multiobjective Co-evolutionary Swarm particle
optimization) co-evolves two multi-objective particle swarms: a macro swarm
of community-centre genomes optimising kernel *k*-means / ratio cut and a
micro swarm of per-node label vectors optimising the intra/inter modularity
decomposition. The swarms exchange guidance and influence through
crowding-pruned Pareto archives and an elite-consensus edge weighting carried
on the graph's edges rather than a dense *n*×*n* kernel, so memory is
*O(n+m)*. Decoding, swarm updates and objective evaluations run as CUDA
kernels, and the search is stochastic by design: repeated runs return
different partitions. The merged rank-1 front is enriched by a union
refinement, and one partition is returned with no ground truth by minimising
a min–max-normalised scalarisation of all four objectives across the front.

GMOCS requires an NVIDIA GPU (Pascal or newer) and raises `RuntimeError`
when no usable CUDA device is present.

The frontier is exposed for inspection via
[`gmocs_fronts`](api/fronts.md#pymocd.gmocs_fronts).

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
