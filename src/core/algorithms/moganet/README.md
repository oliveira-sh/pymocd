# MOGA-Net — Multi-Objective Genetic Algorithm for community detection in networks

**Paper**
Pizzuti, C., "A Multi-Objective Genetic Algorithm for Community Detection in
Networks", *Proc. 21st IEEE Int. Conf. on Tools with Artificial Intelligence
(ICTAI)*, pp. 379–386, 2009. DOI 10.1109/ICTAI.2009.58.
Pizzuti, C., "A Multiobjective Genetic Algorithm to Find Communities in Complex
Networks", *IEEE Transactions on Evolutionary Computation* 16(3):418–430, 2012.
DOI 10.1109/TEVC.2011.2161090. (The TEVC paper is the parameter authority; the
ICTAI paper is the one that defines the operators.)

Representation: Park, Y. & Song, M., "A genetic algorithm for clustering
problems", *Proc. 3rd Annual Conf. on Genetic Programming*, pp. 568–575, 1998.
Engine conventions (non-dominated sort, crowding distance): Deb, Pratap,
Agarwal & Meyarivan, "A fast and elitist multiobjective genetic algorithm:
NSGA-II", *IEEE TEC* 6(2):182–197, 2002. DOI 10.1109/4235.996017.
Community fitness objective: Lancichinetti, Fortunato & Kertész, "Detecting the
overlapping and hierarchical community structure in complex networks",
*New Journal of Physics* 11:033015, 2009. DOI 10.1088/1367-2630/11/3/033015.

**Original implementation**
https://staff.icar.cnr.it/pizzuti/codice/Moganet2016.zip — MATLAB, shipped as
obfuscated P-code (`MOGANet.p`, `create_population.p`, `crossover_cluster.p`,
`mutate_cluster.p`), so the search loop cannot be read. What *is* readable are
its result files, and they were used to settle the objectives (see
**Divergences**).

**Objectives**
Both are **maximized**. `labels` is position-indexed with compact community
ids; for node `i` in community `S`, `k_in(i)` is its number of neighbours
inside `S`, `deg(i)` its degree, and `|S|` the node count of `S`.

    mu_i     = k_in(i) / |S|                        in [0,1)
    M(S)     = ( Σ_{i∈S} mu_i^r ) / |S|             mean of mu_i^r, NO outer root
    v_S      = Σ_{i∈S} k_in(i) = 2 · |E(S)|         internal edges, double-counted
    score(S) = M(S) · v_S
    CS       = Σ_S score(S)                         community score,   maximize
    CF       = Σ_S Σ_{i∈S} k_in(i) / deg(i)^α       community fitness, maximize
                                                    (deg(i) = 0 → term is 0)

`r` is the power that biases `M(S)` towards dense communities; `α` is the
community-fitness exponent. `CS` rises with cohesive but not-too-large
communities, `CF` rises monotonically with coarseness (a single blob maximizes
it, singletons give 0), so the two genuinely trade off — see the test
`cs_maximal_for_two_community_split` in `api.rs`.

The dominance rule inside the search minimizes, so an individual stores
`objectives = [-CS, -CF]`. Feeding `(CS, -CF)` — i.e. reading CF as a quantity
to minimize — is a sign error that over-fragments; CF is maximized.

Both sums are accumulated per community into `Vec` accumulators and summed in
ascending position / community-id order, so the float value is reproducible for
a given labelling.

**Representation**
Locus-based adjacency (Park & Song 1998). An individual is a `Genome =
Vec<usize>` of length `n`: gene `p` holds a *position* `q`, read as "link
position `p` to position `q`". Positions are indices into `Locus::nodes`, the
graph's sorted node vector; `NodeId`s appear only at the module boundary.

Decoding is a union-find over the `n` links, then a relabelling pass that
assigns community ids in ascending-position first-visit order — so two genomes
encoding the same partition decode to the identical label array.

Initialization is the paper's *safe* (biased) variant: every gene is drawn
uniformly from the neighbours of its own position, and both operators preserve
that (crossover copies genes verbatim from safe parents, mutation redraws from
neighbours). Consequently **no repair pass exists anywhere in the pipeline**.
The only self-referential allele, `genome[p] == p`, occurs for isolated nodes,
which have no neighbour to point at; they end up as singleton communities and
`normalize_community_ids` reports them as `-1`.

**Algorithm**
1. `Locus::build` — freeze the node order and the neighbour-position lists.
2. `pop_size` random safe genomes; decode and evaluate each into `[-CS, -CF]`.
3. Per generation `t = 0..num_gens`:
   a. `fast_non_dominated_sort` then `calculate_crowding_distance` over the
      whole population (size stays exactly `pop_size`; there is never a 2N
      pool).
   b. Sort indices by (rank ↑, crowding distance ↓) into `order`.
   c. Copy the first `elite_count = max(1, round(0.10 · pop_size))` members of
      `order` into the next generation unchanged.
   d. Fill the rest with children: two parents by roulette over `order` with
      weights `1/sqrt(position)`; with probability `cross_rate` the child is
      their uniform crossover, otherwise a clone of one parent chosen by a fair
      coin. Every child is then mutated (per gene, with probability `mut_rate`,
      redrawn from its neighbours) and evaluated.
   e. The children replace the population.
4. Rank the final population once more.
5. Decision rule (`moga_net`): among rank-1 members, return the one of highest
   Newman modularity `Q`, normalized. `moga_net_fronts` returns the whole
   rank-1 front instead, which is what Pizzuti's Table 1 protocol (best NMI
   over the front) needs.

**Parameters**
| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 300 | Population size; constant across generations. |
| `num_gens` | 30 | Generations. |
| `cross_rate` | 0.8 | Probability a child is a crossover rather than a parent clone. |
| `mut_rate` | 0.2 | Per-gene probability of redrawing the allele from the node's neighbours. |
| `r` | 2.0 | Exponent inside `M(S)`; higher `r` penalizes nodes weakly attached to their community. |
| `alpha` | 1.0 | Community-fitness exponent on `deg(i)`. |

`pop_size`, `num_gens`, `cross_rate`, `mut_rate` and the 10% elite fraction are
Pizzuti's own single-run settings: *"The population size was 300, the number of
generations 30, the crossover rate 0.8, the mutation rate 0.2, elite
reproduction 10% of the population size, roulette selection function."*
The elite fraction is `ELITE_FRACTION` in `search/engine.rs`, not a public
parameter.

`r = 2.0` is TEVC 2012 Sec. VI-C: *"the parameter r ... has been set to 2"*.
It is also confirmed numerically against the Moganet2016 package: fitting `r`
to the numeric columns of her `karate.pairs.ris` pins a unique minimum at
`r = 2` (`r = 1.5` is off by 7.16). Do **not** re-derive `r` from the paper's
reported karate NMI of 0.602 — her own binary does not produce that number (its
max-Q karate solution is `Q = 0.4151`, `NMI = 0.7071`). Moving the default from
the earlier 1.5 to the paper's 2 costs roughly **−0.006 mean AMI**; it is kept
because the point of this module is to reproduce the published method, not to
win.

**Determinism**
**Nondeterministic: two consecutive calls differ.** `search/engine.rs::run`
draws from `rand::rng()`, the thread-local generator, which is seeded from the
OS and is neither exposed nor re-seedable through the public API; there is no
seed parameter. Everything downstream of the draws is order-fixed (`Vec`
accumulators, a stable node order, first-visit relabelling, sequential sort and
selection), so the *only* source of run-to-run variation is the RNG stream.

Measured reference on karate over 30 runs: AMI 0.2554 ± 0.0000, k = 4.00. The
zero spread is the objective landscape of a 34-node graph, not determinism —
the returned partition itself is not guaranteed identical between calls.

**Divergences**
The two search-loop divergences are **unverifiable** against the reference,
because the parts of Moganet2016 that would settle them are P-code. They are
recorded in the `//!` header of `search/engine.rs` as well:

- **Replacement scheme.** The paper describes NSGA-II, whose replacement
  combines parents and offspring into a 2N pool and truncates by (rank,
  crowding). This loop instead copies the top `elite_count` and fills the rest
  with fresh children, so a child never competes against the parent generation.
  This is the MATLAB `gamultiobj`-style elitist + roulette generational
  replacement the paper's own tooling used.
- **Mutation applied to every child.** The paper applies mutation to crossover
  offspring; here every child is mutated, including one that was cloned rather
  than crossed. `crossoverFraction` in `gamultiobj` can be read either way, and
  the composed reading is the one kept.
- **Selection fitness.** The paper only says "roulette selection function". The
  weights used are `gamultiobj`'s default rank scaling, `1/sqrt(position)` over
  the (rank ↑, crowding ↓) order (`rank_fitness_weights`).
- **Objectives are *not* a divergence — they are proven identical.** The
  compiled `community_objectives` reproduces all three numeric columns of
  Pizzuti's shipped `karate.pairs.ris`, on all 7 of her solutions, to 7
  significant digits. Any future edit to `objectives.rs` can be re-checked
  against that file.
- **No shared engine, no Rayon.** The non-dominated sort and crowding distance
  are re-implemented here, single-threaded, rather than reusing the project's
  shared NSGA-II/NSGA-III machinery, so this baseline's runtime tracks the
  published single-threaded method rather than the project's optimizations.
- **No repair operator.** The paper describes repairing invalid genes to "one
  of the neighbors of `i`". Safe initialization plus locus-respecting operators
  make every gene safe by construction, so the repair pass is absent rather
  than implemented and never triggered.
- **Decision rule.** Reducing the rank-1 front to one partition by maximum
  Newman `Q` follows TEVC 2012 Sec. V-E. Pizzuti's Table 1 protocol instead
  reports the best-NMI member of the front, which is what `moga_net_fronts`
  exposes.

**Files**
| Path | Holds |
|---|---|
| `mod.rs` | Module root; re-exports `moga_net`, `moga_net_fronts` and the defaults. |
| `api.rs` | The two entry points, the rank-1 → partition plumbing, the max-modularity decision rule, and the module's tests. |
| `defaults.rs` | The six shipped defaults. |
| `locus.rs` | `Genome`, `Locus` (node order + neighbour positions), safe allele/genome sampling, union-find decode. |
| `objectives.rs` | `community_objectives` → `(CS, CF)`, and `label_modularity` for the decision rule. |
| `operators.rs` | Uniform crossover and neighbour-restricted mutation. |
| `search/mod.rs` | Search sub-module root. |
| `search/individual.rs` | `Individual`, Pareto dominance, fast non-dominated sort, crowding distance. |
| `search/engine.rs` | The generational loop, elitism, roulette selection — and the two unverifiable divergences. |
