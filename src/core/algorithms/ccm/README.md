# CCM — NSGA-III with Community Score, Community Fitness and Modularity

**Paper**
Tanveerulhuq Shaik, Vadlamani Ravi & Kalyanmoy Deb, "Evolutionary Multi-Objective
Optimization Algorithm for Community Detection in Complex Social Networks",
*SN Computer Science* 2:13, 2021. DOI [10.1007/s42979-020-00382-x](https://doi.org/10.1007/s42979-020-00382-x).
(The same paper defines this project's `krm` module; CCM is its Community
Score / Community Fitness / Modularity variant.)

The objectives and the genome come from Clara Pizzuti, "A Multi-Objective
Genetic Algorithm to Find Communities in Complex Networks", *IEEE TEC*
16(3):418–430, 2012 (and *IEEE ICTAI* 2009). DOI 10.1109/TEVC.2011.2161090.

The engine is Kalyanmoy Deb & Himanshu Jain, "An Evolutionary Many-Objective
Optimization Algorithm Using Reference-Point-Based Nondominated Sorting
Approach, Part I", *IEEE TEC* 18(4):577–601, 2014. DOI 10.1109/TEVC.2013.2281535,
with the non-dominated sort of Deb, Pratap, Agarwal & Meyarivan, "NSGA-II",
*IEEE TEC* 6(2):182–197, 2002. DOI 10.1109/4235.996017, and the structured
weights of Das & Dennis, *SIAM J. Optimization* 8(3):631–657, 1998.

**Original implementation**
None published.

**Objectives**
Three, **all maximized by the paper**. `S` is a community of the partition,
`|S|` its node count, `k_in(i)` the number of `i`'s neighbours that are also in
`i`'s own community, `deg(i)` its degree, `m` the edge count, `l_c` the edges
internal to community `c` (counted once) and `d_c` the sum of the degrees of its
nodes.

    mu_i     = k_in(i) / |S|
    M(S)     = ( Σ_{i∈S} mu_i^r ) / |S|            power mean, NO outer 1/r root
    v_S      = Σ_{i∈S} k_in(i) = 2·(internal edges of S)
    CS       = Σ_S M(S) · v_S                      community score
    CF       = Σ_S Σ_{i∈S} k_in(i) / deg(i)^α      community fitness, deg(i)=0 → term 0
    Q        = Σ_c l_c/m − (d_c / 2m)²             Newman modularity

`CF` peaks when no edge leaves a community, so maximizing it minimizes the
inter-community links. The search engine is a pure minimizer, so the vector it
actually receives is `(−CS, −CF, −Q)` (`objectives::evaluate`); every rank,
dominance test and reference-point distance in `nsga3/` is in that flipped
space, and the sign is undone only by the modularity recomputed in `api.rs`.

**Representation**
Locus-based (Pizzuti's GA-Net encoding). An individual holds a `Genome =
Vec<usize>` over node *positions* in the graph's stable node order: cell `p`
holds `p` itself or the position of one of `p`'s neighbours, so every genome an
operator can produce is valid and no repair step exists. Decoding is a union-find
over the cells: each connected component is one community, labelled by its
union-find root position. Every label is therefore an index in `0..n`, which is
what lets the objectives accumulate into flat `n`-sized arrays. Initialisation
draws each cell uniformly from `{p} ∪ neighbours(p)`. Isolated nodes decode to
singletons and are turned into community `-1` by `normalize_community_ids` on
the way out.

**Algorithm**
1. `pop_size` random genomes; decode, evaluate `(−CS, −CF, −Q)`, non-dominated sort.
2. Build the Das–Dennis reference points for `M = 3` objectives and `divisions`
   subdivisions (`H = C(M + p − 1, p) = 91` at the default `p = 12`).
3. Per generation:
   a. `pop_size` offspring. Two parents drawn uniformly at random (no mating
      selection). With probability `cross_rate` the child is their uniform
      crossover, otherwise a clone of one of the two, chosen 50/50. Then every
      gene is independently resampled from `{p} ∪ neighbours(p)` with
      probability `mut_rate`. Evaluate.
   b. Environmental selection over the 2N pool (Deb & Jain, Alg. 1): keep whole
      fronts while they fit; fill the rest from the splitting front by
      normalizing it onto the hyperplane (Alg. 2), associating each member with
      its nearest reference line (Alg. 3) and niching (Alg. 4) — the least
      crowded reference point wins, ties broken uniformly at random, and an
      empty niche takes its closest member while a non-empty one takes a random
      one.
   c. The paper's two customizations (Sec. 4), in order: any individual whose
      canonical labelling repeats one already seen is replaced by a fresh random
      genome, then so is any individual that puts every node in one community.
      Replacements are re-evaluated immediately.
   d. Re-rank, because (c) replaced members after selection had ranked them.
4. After the last generation the population is sorted once more and the rank-1
   members are the result. `ccm` returns the one of highest modularity — the
   paper's decision rule when no ground truth is available — and `ccm_fronts`
   returns all of them, which is what the paper's Table 1/2 protocol
   (best-NMI *and* best-Q of the front) needs.

**Parameters**
| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 200 | Population size. Sec. 8.1 of the paper says 20 for karate, but Table 3's grid and Fig. S.22 use 200, and the value has to stay above the reference-point count `H = 91`. |
| `num_gens` | 100 | Generations. The paper's best-reported combination on Zachary karate. |
| `cross_rate` | 0.8 | Probability the child is a uniform crossover rather than a clone. Paper's best combination. |
| `mut_rate` | 1/68 | Per-gene resampling probability. Paper's best combination (`1/68` on the 34-node karate graph). |
| `r` | 1.0 | Community Score power-mean exponent. Fixed by the paper: `r = 1` weights all nodes equally. |
| `alpha` | 1.0 | Community Fitness degree exponent. Fixed by the paper: `α = 1`, following MOGA-Net. |
| `divisions` | 12 | Das–Dennis subdivisions `p`. The paper does not state one; 12 is pymoo's `M = 3` convention and gives 91 reference points. |

Internal constants, all in `nsga3/normalize.rs`: `NEAR_ZERO = 1e-10` (the
"treat as zero" threshold for a pivot, an intercept and a fallback scale),
`MIN_INTERCEPT = 1e-6` (below this an intercept is rejected and the fallback
scaling is used), `ASF_OFF_AXIS_WEIGHT = 1e-6` (the off-axis weight of the
achievement scalarizing function that picks the extreme points).

**Determinism**
Nondeterministic: two consecutive calls differ. `nsga3::engine::evolve` seeds
`rand::rng()` — the thread-local generator, seeded from the OS — and there is no
seed parameter on the public API, so the initial population, every crossover,
mutation and clone draw, and the random tie-breaks inside niching all change
from run to run. The search itself is sequential (no Rayon), so the thread count
is not a further source of variation. Measured over 30 runs on Zachary karate:
AMI 0.2398 ± 0.0000, 4 communities every time — that graph is degenerate, and
the run-to-run spread shows up on anything larger (on a 250-node LFR graph the
rank-1 front is roughly 110 ± 30 members).

**Divergences**
- `nsga3/` is a private, sequential NSGA-III instead of a shared engine, and
  uses no Rayon: this is a baseline, and its cost and behaviour have to track
  the paper's rather than this project's infrastructure.
- Mating is a uniform random draw of two parents, with no fitness-based
  selection. That is pymoo's NSGA-III for unconstrained problems, which the
  paper adapted; the reader expecting the binary tournament of most GA
  implementations will not find one.
- `M(S)` is the plain mean of `mu_i^r`, with no outer `1/r` root. This follows
  Pizzuti's definition and is not a dropped operation.
- `pop_size = 200` and `divisions = 12` resolve, respectively, a contradiction
  and an omission in the paper — see the parameter table.
- Modularity is recomputed locally (`objectives::modularity_labels`) instead of
  calling `core::metrics::modularity`, so the hot loop never materializes a
  `Partition`. A test asserts the two agree to 1e-12.
- The public entry points return normalized partitions, with isolated nodes
  reported as community `-1`; the paper leaves them as singletons.
- The duplicate filter compares *canonical* labellings, so it removes
  permutation duplicates and not just literally equal label arrays.

**Files**
| Path | Holds |
|---|---|
| `mod.rs` | Module root; declares the submodules and re-exports `ccm`, `ccm_fronts` and the defaults. |
| `api.rs` | The two entry points, the shared rank-1 front extraction and the partition normalization. |
| `defaults.rs` | The seven shipped defaults. |
| `locus.rs` | The locus genome: neighbour-position table, random genome, union-find decode, canonical relabelling, uniform crossover, adjacency-constrained mutation. |
| `objectives.rs` | Community Score, Community Fitness, Newman modularity, and the sign flip that hands them to a minimizing engine. |
| `fixtures.rs` | The test graph shared by more than one test module. |
| `nsga3/mod.rs` | The engine's module root. |
| `nsga3/individual.rs` | The population member, Pareto dominance, and the fast non-dominated sort. |
| `nsga3/engine.rs` | The generational loop, the breeding operator, and the paper's two customizations. |
| `nsga3/survival.rs` | Environmental selection: whole fronts first, then the niched splitting front. |
| `nsga3/normalize.rs` | Hyperplane normalization: ideal point, extreme points, intercepts, and the Gauss–Jordan solve they need. |
| `nsga3/niching.rs` | Reference-line association and the niche-preserving choice of survivors. |
| `nsga3/reference_points.rs` | The Das–Dennis simplex lattice. |
