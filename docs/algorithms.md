# Algorithms

`pymocd` ships eight detectors. **SCALE** and **HP-MOCD** are the library's
own contributions; the remaining six re-implement published baselines whose
authors released no code.

## Overview

| API | Algorithm | Objectives & engine | Solution selection | Year |
|---|---|---|---|---|
| [`scale`](api/detectors.md#pymocd.scale) | **SCALE** (Santos, in prep.) | KKM / ratio-cut bi-objective, sparse macro–micro co-evolutionary NSGA-II (near-linear, no dense kernel) | label-free **SBM/MDL** description length | 2026 |
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

- **[`scale`](api/detectors.md#pymocd.scale)** — the recommended default:
  label-free SBM/MDL selection, near-linear time and memory.
- **[`hpmocd`](api/detectors.md#pymocd.hpmocd)** — the published HP-MOCD
  behaviour with max-modularity selection.
- **The other six** — baselines for papers and benchmarks; `pop_size`,
  `num_gens` and rates are tunable kwargs.

## SCALE

SCALE co-evolves a macro population of medoid community centres with a micro
population of per-node labels over the kernel *k*-means / ratio-cut
bi-objective, bridged by a sparse similarity carried on the graph's edges
rather than a dense *n*×*n* kernel — so memory is *O(n+m)* and it scales to
graphs the dense macro–micro baseline cannot build. The merged rank-1 front is
enriched by a union refinement, and one partition is returned with no ground
truth by minimising a label-free
[microcanonical SBM](https://doi.org/10.1103/PhysRevX.4.011047) description
length.

The frontier is exposed for inspection via
[`scale_fronts`](api/fronts.md#pymocd.scale_fronts).

## HP-MOCD

HP-MOCD optimises decomposed modularity with a parallel NSGA-II and returns
the max-*Q* solution from the Pareto front. The
[`HpMocd`](api/detectors.md#pymocd.HpMocd) class exposes the front itself for
inspection. Published in
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
