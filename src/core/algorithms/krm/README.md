# KRM — NSGA-III with Kernel-K-Means, Ratio-Cut and Modularity

**Paper**
Tanveerulhuq Shaik, Vadlamani Ravi & Kalyanmoy Deb, "Evolutionary Multi-Objective
Optimization Algorithm for Community Detection in Complex Social Networks",
*SN Computer Science* 2:13, 2021. DOI [10.1007/s42979-020-00382-x](https://doi.org/10.1007/s42979-020-00382-x).
(The same paper defines this project's `ccm` module; the three letters of each
name are its three objectives — **K**KM / **R**atio-Cut / **M**odularity here,
Community Score / Community Fitness / Modularity there.)

The `(KKM, RC)` pair the paper adopts is the objective pair of MODPSO; the
genome is Clara Pizzuti's locus encoding, "A Multiobjective Genetic Algorithm to
Find Communities in Complex Networks", *IEEE TEC* 16(3):418–430, 2012.
DOI 10.1109/TEVC.2011.2161090, itself after Park & Song, "A genetic algorithm
for clustering problems", *Proc. 3rd Annual Conf. on Genetic Programming*,
pp. 568–575, 1998.

The engine is Kalyanmoy Deb & Himanshu Jain, "An Evolutionary Many-Objective
Optimization Algorithm Using Reference-Point-Based Nondominated Sorting
Approach, Part I", *IEEE TEC* 18(4):577–601, 2014. DOI 10.1109/TEVC.2013.2281535,
with the non-dominated sort of Deb, Pratap, Agarwal & Meyarivan, "NSGA-II",
*IEEE TEC* 6(2):182–197, 2002. DOI 10.1109/4235.996017, and the structured
weights of Das & Dennis, *SIAM J. Optimization* 8(3):631–657, 1998.

Modularity is Newman & Girvan, "Finding and evaluating community structure in
networks", *Phys. Rev. E* 69:026113, 2004. DOI 10.1103/PhysRevE.69.026113.

**Original implementation**
None published. The authors' sites and GitHub were searched; nothing is
available, so only the paper's text and tables constrain this port.

**Objectives**
Three. `KKM` and `RC` are **minimized**, `Q` is **maximized**. For a partition
`C = {V_1 … V_k}` of `n` nodes, with `L(V_a, V_b) = Σ_{i∈V_a, j∈V_b} A_ij`,
`m` the edge count, `l_c` the edges internal to community `c` (counted once) and
`d_c` the sum of the degrees of its nodes:

    L(V_i, V_i) = Σ_{v∈V_i} #neighbours of v inside V_i   = 2 · (internal edges)
    L(V_i, V̄_i) = Σ_{v∈V_i} deg(v) − L(V_i, V_i)          = the cut

    KKM = 2(n − k) − Σ_i L(V_i, V_i)/|V_i|      denser communities ⇒ lower KKM
    RC  =            Σ_i L(V_i, V̄_i)/|V_i|      fewer inter-links  ⇒ lower RC
    Q   = Σ_c l_c/m − (d_c / 2m)²

`L(V_i, V_i)` counts each internal edge from **both** endpoints, so it is twice
the internal-edge count; the formulas above assume that convention and
`objectives::community_stats` produces it.

The two search objectives are degenerate in opposite directions: `KKM` is
minimized by fragmentation (all singletons give exactly `0`), `RC` by
coarsening (one community has no cut, so `RC = 0`). Neither alone rules out its
own degenerate extreme, which is why they are optimized against each other;
`objectives.rs::split_sits_between_the_degenerate_extremes` pins both extremes
and the good split between them on a two-triangles graph.

The engine is a pure minimizer, so the vector an `Individual` actually stores is
`[KKM, RC, −Q]` (`nsga3/engine.rs::make_individual`). Every rank, dominance test
and reference-point distance in `nsga3/` lives in that flipped space, and the
sign is undone only by `api.rs`, which reads `−objectives[NEGATED_MODULARITY]`.

**Representation**
Locus-based (Pizzuti's GA-Net encoding). An individual holds a `Genome =
Vec<usize>` over node *positions* in the graph's stable node order: cell `p`
holds `p` itself or the position of one of `p`'s neighbours, so every genome an
operator can produce is valid and **no repair step exists anywhere in the
module**. Decoding is a union-find over the cells — union `p` with `genome[p]` —
and each connected component is one community, labelled by its union-find root
position. Labels are therefore arbitrary ids in `0..n`, not compacted, which is
exactly what lets `community_stats` and `canonical_labels` index flat `n`-sized
arrays by a raw label. Many distinct genomes encode one partition, so the
duplicate filter compares `Locus::canonical_labels` (communities relabelled by
first-seen position order) rather than the label arrays themselves.

Initialisation draws each cell uniformly from `{p} ∪ neighbours(p)`. Isolated
nodes have no neighbour to point at, decode to singletons, and are reported as
community `-1` by `normalize_community_ids` on the way out.

`Locus::nodes` is the module's only boundary with the shared graph: everything
between `Locus::build` and `api.rs::to_partition` speaks positions, never
`NodeId`s.

**Algorithm**
1. `Locus::build` freezes the node order and the per-position candidate lists.
2. `pop_size` random genomes; decode, evaluate `[KKM, RC, −Q]`, non-dominated sort.
3. Build the Das–Dennis reference points for `M = 3` objectives and `divisions`
   subdivisions (`H = C(M + p − 1, p) = 91` at the default `p = 12`).
4. Per generation:
   a. `pop_size` offspring. Two parents drawn uniformly at random (no mating
      selection). With probability `cross_rate` the child is their uniform,
      per-gene crossover, otherwise a clone of one of the two, chosen 50/50.
      Then every gene is independently resampled from `{p} ∪ neighbours(p)` with
      probability `mut_rate`. Decode and evaluate.
   b. Environmental selection over the `2·pop_size` pool (Deb & Jain, Alg. 1):
      keep whole fronts while they fit; fill the rest from the splitting front
      by translating `St` by the ideal point and scaling by the hyperplane
      intercepts (Alg. 2), associating each member with its nearest reference
      line by perpendicular distance (Alg. 3), and niching (Alg. 4) — the least
      crowded reference point wins, ties among equally crowded points are broken
      uniformly at random, an empty niche takes its closest member and a
      non-empty one takes a random member.
   c. The paper's two customizations, in this order: any individual whose
      canonical labelling repeats one already seen is replaced by a fresh random
      genome, then so is any individual that puts every node in one community.
      Replacements are decoded and evaluated immediately.
   d. Re-rank. This is load-bearing: (c) overwrites individuals *after*
      selection ranked them, so the replacements would otherwise inherit the
      rank of whoever they displaced and carry it into the next generation and
      into the returned population.
5. After the last generation the caller ranks the population once more and takes
   the rank-1 members. `krm` returns the one of highest modularity — the paper's
   decision rule when no ground truth is available — and `krm_fronts` returns
   all of them, which is what the paper's Table 1/2 protocol (best-NMI *and*
   best-Q over the front) needs.

**Parameters**
| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 100 | Population size. The paper's best combination on D1 / Zachary karate, and it has to stay at or above the reference-point count `H = 91`. |
| `num_gens` | 100 | Generations. Same combination. |
| `cross_rate` | 0.8 | Probability the child is a uniform crossover rather than a clone. Same combination. |
| `mut_rate` | 1/34 | Per-gene resampling probability — `1/n` on the 34-node karate graph. Same combination. |
| `divisions` | 12 | Das–Dennis subdivisions `p`. The paper does not state one; 12 is pymoo's `M = 3` convention and gives 91 reference points. |

The paper's grid is per-dataset, and only D1's row is shipped as the default.
The other three rows, as `pop_size / cross_rate / mut_rate`, are D2 200 / 0.85 /
1-124, D3 500 / 0.9 / 1-230, D4 400 / 0.9 / 2-105; pass them explicitly to
reproduce those datasets.

Internal constants:

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `NEGATED_MODULARITY` | 2 | `api.rs` | Index of `−Q` in the objective vector. |
| `NEAR_ZERO` | 1e-10 | `nsga3/normalize.rs` | "Treat as zero" for a Gauss–Jordan pivot, an intercept, and a fallback scale. |
| `MIN_INTERCEPT` | 1e-6 | `nsga3/normalize.rs` | Below this an intercept is rejected and the fallback scaling is used. |
| `ASF_OFF_AXIS_WEIGHT` | 1e-6 | `nsga3/normalize.rs` | Off-axis weight of the achievement scalarizing function that picks the extreme points. |
| `DEGENERATE_REF_NORM2` | 1e-30 | `nsga3/niching.rs` | A reference direction with a smaller squared norm is treated as the origin, and the distance becomes the point's own norm. |

**Determinism**
**Nondeterministic: two consecutive calls differ.** `nsga3::engine::run` and
`nsga3::niching::niche_select` both take `rand::rng()`, the thread-local
generator seeded from the OS, and no seed is exposed anywhere on the public API.
Every source of variation is a draw: the initial population, both parent picks,
the crossover coin flips, the mutation resamples, the replacement genomes of the
two customizations, and the two random tie-breaks inside niching (which
reference point among the equally crowded, and which member of a non-empty
niche).

The search is sequential — no Rayon anywhere in the module — so the thread count
is *not* a further source of variation, and float accumulation order is fixed:
`community_stats` compacts community ids in ascending position order, so a given
labelling always sums its objectives in the same order.

Measured over 30 runs on Zachary karate: AMI 0.2435 ± 0.0120, k = 4.20; over
300 runs, AMI 0.2455 ± 0.0123 (SEM 0.0007), k = 4.17. Because the module is
nondeterministic, a refactor of it cannot be checked by comparing outputs: it
has to be pure code motion, and the check is a distributional A/B — several
hundred runs before and after, comparing the mean and spread of AMI, `k`, the
isolated-node count and the rank-1 front length, on karate plus graphs that
exercise the edge cases (an isolated node, the empty graph, a single node).

**Divergences**
- `nsga3/` is a private, sequential NSGA-III instead of the shared engine, and
  uses no Rayon. This is a baseline: its cost and behaviour have to track the
  paper's rather than this project's infrastructure. The `//!` header of
  `mod.rs` says so, because it looks exactly like something to "optimize".
- Mating is a uniform random draw of two parents, with no fitness-based
  selection. That is pymoo's NSGA-III for unconstrained problems, whose mating
  comparison ranks only constraint violation and so ties every time and falls
  through to a coin flip. A reader expecting the usual binary tournament will
  not find one.
- The ideal point and the extreme points are recomputed on every call to
  `normalize_objectives`; pymoo carries them across generations. This is a known
  minor difference from the reference the paper's tooling used.
- `divisions = 12` resolves an omission in the paper — see the parameter table.
- Modularity is recomputed locally (`objectives::modularity`) rather than
  through `core::metrics::modularity`, so the hot loop never materializes a
  shared `Partition`. The formula is the same one.
- The duplicate filter compares *canonical* labellings, so it removes
  permutation duplicates and not just literally equal label arrays. Without
  that, the locus encoding's many-genomes-per-partition redundancy would defeat
  the filter almost entirely.
- The public entry points return normalized partitions, with isolated nodes
  reported as community `-1`; the paper leaves them as singletons.
- `Individual::rank` is 1-based (`fast_non_dominated_sort` writes `1` for the
  first front, and `usize::MAX` means "not yet ranked"). `survival.rs` skips
  index `0` of its `fronts` bucket vector for that reason.

**Files**
| Path | Holds |
|---|---|
| `mod.rs` | Module root; declares the submodules and re-exports `krm`, `krm_fronts` and the defaults. |
| `api.rs` | The two entry points, the rank-1 max-modularity decision rule, and the labels → `Partition` conversion at the module boundary. |
| `defaults.rs` | The five shipped defaults. |
| `locus.rs` | The locus genome: node order and neighbour-position table, random genome, union-find decode, canonical relabelling, single-community test. |
| `objectives.rs` | Per-community statistics, `(KKM, RC)`, and the in-module Newman modularity. |
| `operators.rs` | Uniform-random mating, uniform crossover, adjacency-constrained mutation. |
| `fixtures.rs` | The two-triangles graph shared by more than one test module (`#[cfg(test)]` only). |
| `nsga3/mod.rs` | The engine's module root. |
| `nsga3/individual.rs` | The population member, Pareto dominance, and the fast non-dominated sort. |
| `nsga3/engine.rs` | The generational loop, the breeding step, the two customizations, and the re-rank they force. |
| `nsga3/survival.rs` | Environmental selection: whole fronts first, then the niched splitting front. |
| `nsga3/normalize.rs` | Hyperplane normalization: ideal point, extreme points, intercepts, and the Gauss–Jordan solve they need. |
| `nsga3/niching.rs` | Reference-line association and the niche-preserving choice of survivors. |
| `nsga3/reference_points.rs` | The Das–Dennis simplex lattice. |
