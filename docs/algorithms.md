# Algorithms

`pymocd` exposes **ten community-detection algorithms** through **eleven
detector entry points** — Shi-MOCD ships under two selection rules, `mocd_q`
and `mocd_d`.

Three of them are this library's own work: **SMOCC**, **MO-POTS** and
**HP-MOCD**. *Every other detector is a re-implementation of someone else's
published method*, written from the paper in this repository. The last column
records whether the original authors released code — three of the seven did,
four did not.

## Overview

| API | Algorithm | Objectives & engine | Selection rule | Original implementation |
|---|---|---|---|---|
| [`smocc`](api/detectors.md#pymocd.smocc) | **SMOCC** — Santos, *in prep.* (2026) | sparse macro–micro co-evolutionary NSGA-II: micro kernel *k*-means/ratio-cut, macro Constant Potts; similarity carried on the graph's edges, so memory is *O(n+m)* | label-free min–max-normalised scalarisation over the merged rank-1 front | **this library** |
| [`mopots`](api/detectors.md#pymocd.mopots) | **MO-POTS** — Santos, *in prep.* (2026) | exact Constant Potts split into cut fraction + pair fraction, parallel NSGA-II — the front *is* the resolution ladder | max modularity *Q* (front via [`mopots_fronts`](api/fronts.md#pymocd.mopots_fronts)) | **this library** |
| [`hpmocd`](api/detectors.md#pymocd.hpmocd) | **HP-MOCD** — [Santos et al., *SNAM* 2025](https://doi.org/10.1007/s13278-025-01519-7) | decomposed modularity (intra, inter), parallel NSGA-II | max modularity *Q* (front via [`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts)) | **this library** |
| [`cdrme`](api/detectors.md#pymocd.cdrme) | **CDRME** — [Dabaghi-Zarandi et al., *JNCA* 2025](https://doi.org/10.1016/j.jnca.2024.104070) | softmax-weighted random walks build a primary community set; stochastic agglomerative merge chains optimise the paper's Eq. (12) linkage scalar — a single objective, so there is no front | max modularity *Q* | a private Python notebook supplied by the authors — **no public repository exists**, so there is no URL to cite |
| [`mmcomo`](api/detectors.md#pymocd.mmcomo) | **MMCoMO** — [Zhang et al., *IEEE CIM* 2023](https://ieeexplore.ieee.org/document/10188453) | kernel *k*-means + ratio cut, macro/micro co-evolutionary NSGA-II over a dense diffusion kernel | max *Q* (front via [`mmcomo_fronts`](api/fronts.md#pymocd.mmcomo_fronts)) | — |
| [`ccm`](api/detectors.md#pymocd.ccm) | **CCM** — [Shaik et al., *SN Computer Science* 2021](https://doi.org/10.1007/s42979-020-00382-x) | community score + community fitness + modularity, NSGA-III | max *Q* (front via [`ccm_fronts`](api/fronts.md#pymocd.ccm_fronts)) | — |
| [`krm`](api/detectors.md#pymocd.krm) | **KRM** — [Shaik et al., *SN Computer Science* 2021](https://doi.org/10.1007/s42979-020-00382-x) | kernel *k*-means + ratio cut + modularity, NSGA-III | max *Q* (front via [`krm_fronts`](api/fronts.md#pymocd.krm_fronts)) | — |
| [`gdpso`](api/detectors.md#pymocd.gdpso) | **GDPSO** — [Cai et al., *Information Sciences* 2015](https://doi.org/10.1016/j.ins.2014.09.041) | Newman–Girvan modularity — a single objective, so there is no front — maximised by a greedy discrete particle swarm | best position the swarm ever held | [doctor-cai/GDPSO](https://github.com/doctor-cai/GDPSO) — C++, no licence file |
| [`mocd_q`](api/detectors.md#pymocd.mocd_q) | **Shi-MOCD** — [Shi et al., *Applied Soft Computing* 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity, PESA-II | max *Q* (Shi Eq. 3.8) | — |
| [`mocd_d`](api/detectors.md#pymocd.mocd_d) | **Shi-MOCD** — [Shi et al., *Applied Soft Computing* 2012](https://doi.org/10.1016/j.asoc.2011.10.005) | decomposed modularity, PESA-II | max–min distance to Erdős–Rényi control fronts (Shi Eqs. 3.9–3.11) | — |
| [`moga_net`](api/detectors.md#pymocd.moga_net) | **MOGA-Net** — [Pizzuti, *IEEE TEC* 2012](https://doi.org/10.1109/TEVC.2011.2161090) | community score + community fitness, NSGA-II | max *Q*, Pizzuti Sec. V-E (front via [`moga_net_fronts`](api/fronts.md#pymocd.moga_net_fronts)) | [Moganet2016.zip](https://staff.icar.cnr.it/pizzuti/codice/Moganet2016.zip) — MATLAB, shipped as obfuscated P-code |

All detectors return a single crisp partition as `dict[node, community]`;
isolated nodes are assigned community `-1`.

Each one has a module README carrying its full derivation, its parameter table
and every deliberate divergence from its paper:
[`smocc`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/smocc/README.md) ·
[`mopots`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/mopots/README.md) ·
[`hpmocd`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/hpmocd/README.md) ·
[`cdrme`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/cdrme/README.md) ·
[`mmcomo`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/mmcomo/README.md) ·
[`ccm`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/ccm/README.md) ·
[`krm`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/krm/README.md) ·
[`gdpso`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/gdpso/README.md) ·
[`mocd`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/mocd/README.md) (Shi-MOCD) ·
[`moganet`](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/moganet/README.md) ·
[index](https://github.com/oliveira-sh/pymocd/blob/master/src/core/algorithms/README.md).

## Which one should I use?

- **[`smocc`](api/detectors.md#pymocd.smocc)** — the recommended default:
  label-free automatic selection, near-linear time and memory.
- **[`mopots`](api/detectors.md#pymocd.mopots)** — one run, every resolution:
  its Pareto front *is* the Constant Potts resolution ladder, read off with
  [`mopots_ladder`](api/fronts.md#pymocd.mopots_ladder).
- **[`hpmocd`](api/detectors.md#pymocd.hpmocd)** — the published HP-MOCD
  behaviour with max-modularity selection.
- **The other seven** — re-implemented baselines, for papers and benchmarks.
  Each takes its budget as keyword arguments at the defaults its own paper
  states; see the [detector API reference](api/detectors.md) for every
  signature.

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

## MO-POTS

MO-POTS (Multi-Objective Potts) minimises the exact affine decomposition of
the Constant Potts Model into two conflicting objectives with a parallel
NSGA-II:

```text
cut(C)  = 1 - sum_c |E(c)| / m        fraction of edges leaving their community
pair(C) = sum_c C(n_c, 2) / C(n, 2)   fraction of node pairs put together
```

with `n` the number of **non-isolated** nodes, the only ones both objectives
range over — isolated nodes never move and are returned as community `-1`.

One giant community gives `cut = 0, pair = 1`; all singletons give
`cut = 1, pair = 0`. Because `H_g(C)/m = 1 - cut(C) - (g/g_d)*pair(C)`, with
`g_d = 2m/(n(n-1))` the edge density over those same `n` non-isolated nodes,
minimising any positive weighting `a*cut + b*pair` is exactly maximising CPM
at resolution `g = g_d*b/a`. The resolution is therefore not a search
parameter but the exchange rate between the two objectives, and the Pareto
front *is* the resolution profile — a single run sweeps the whole ladder.

[`mopots`](api/detectors.md#pymocd.mopots) returns the max-*Q* member of that
front, [`mopots_fronts`](api/fronts.md#pymocd.mopots_fronts) returns every
member, and [`mopots_ladder`](api/fronts.md#pymocd.mopots_ladder) returns the
front's lower convex hull as `(partition, cut, pair, gamma)` in increasing
`gamma` — each partition paired with the resolution at which it becomes
CPM-optimal. All three take `pop_size=100`, `num_gens=100`, `cross_rate=0.7`
and `mut_rate=0.5` as kwargs.

## HP-MOCD

HP-MOCD optimises decomposed modularity with a parallel NSGA-II and returns
the max-*Q* solution from the Pareto front. The front itself is exposed for
inspection via [`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts).
Published in
[Social Network Analysis and Mining (2025)](https://doi.org/10.1007/s13278-025-01519-7).

[`hpmocd`](api/detectors.md#pymocd.hpmocd) and
[`hpmocd_fronts`](api/fronts.md#pymocd.hpmocd_fronts) take the graph and
nothing else: they run at the published configuration (`pop_size=100`,
`num_gens=100`, `cross_rate=0.7`, `mut_rate=0.5`). For a tunable HP-MOCD, or
to supply your own Python objective functions, use the `pymocd.HpMocd` class.

## The re-implemented baselines

The seven detectors below are other people's algorithms, ported from their
papers so that benchmarks in this repository compare against a faithful
implementation rather than a paraphrase. Each module README lists what was
read, what the paper leaves open, and every place this port deliberately
diverges from it.

**Original code released by the authors — three of the seven.**

- **MOGA-Net** ([Moganet2016.zip](https://staff.icar.cnr.it/pizzuti/codice/Moganet2016.zip))
  is MATLAB, shipped as obfuscated P-code: the search loop cannot be read, but
  its result files can, and they were used to settle the objectives.
- **GDPSO** ([doctor-cai/GDPSO](https://github.com/doctor-cai/GDPSO)) is C++
  with no licence file, so this port was written clean-room from a description
  of it; no reference source was copied.
- **CDRME**'s reference is a private Python notebook supplied by the authors.
  It is **not published at any public URL**, so none is cited here. It was
  consulted only to disambiguate what the paper leaves open, and four of its
  choices were deliberately not reproduced.

**No original code exists** for MMCoMO, CCM, KRM or Shi-MOCD. The authors'
sites and GitHub were searched; only the papers' text and tables constrain
those four ports.

Two of the seven are single-objective and therefore have no Pareto front and
no `*_fronts` accessor: `gdpso` maximises Newman–Girvan modularity directly,
and `cdrme` maximises the sum its paper's Eq. (12) forms out of inner and
outer linkage. `mocd_q` and `mocd_d` are multi-objective but expose no front
accessor either; the two names are the same PESA-II search under Shi's two
published model-selection rules.

!!! warning "Shi-MOCD defaults are not Shi's"
    [`mocd_q`](api/detectors.md#pymocd.mocd_q) and
    [`mocd_d`](api/detectors.md#pymocd.mocd_d) default to this repository's
    HP-MOCD-parity benchmark budget (`cross_rate=0.9`, `mut_rate=0.1`), not to
    the paper's `pc=0.6`, `pm=0.4` with per-graph population and generation
    counts from its Table 1. Pass those explicitly to reproduce the paper.

## Deprecated aliases

`pymocd.scale` and `pymocd.scale_fronts` are kept from before SMOCC was
renamed. They are the same objects as
[`smocc`](api/detectors.md#pymocd.smocc) and
[`smocc_fronts`](api/fronts.md#pymocd.smocc_fronts); use the new names.

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
