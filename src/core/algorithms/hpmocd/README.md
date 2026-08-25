# HP-MOCD — High-Performance Multi-Objective Community Detection

**Paper**

Santos, G. O., Vieira, L. S., Rossetti, G., Ferreira, C. H. G. & Moreira, G. J. P.,
"A high-performance evolutionary multiobjective community detection algorithm",
*Social Network Analysis and Mining* 15(1):110, 2025.
DOI [10.1007/s13278-025-01519-7](https://doi.org/10.1007/s13278-025-01519-7).

Objective pair: Shi, C., Yan, Z., Cai, Y. & Wu, B., "Multi-objective community
detection in complex networks", *Applied Soft Computing* 12(2):850–859, 2012.
DOI [10.1016/j.asoc.2011.10.005](https://doi.org/10.1016/j.asoc.2011.10.005) —
Eqs. 3.5 and 3.6.

Engine: Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T., "A fast and elitist
multiobjective genetic algorithm: NSGA-II", *IEEE Transactions on Evolutionary
Computation* 6(2):182–197, 2002.
DOI [10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

**Original implementation**

This project is the reference implementation; there is no external one.

**Objectives**

Both **minimised**, over the communities `c` of the partition:

    intra = 1 − (Σ_c l_c) / m          Shi Eq. 3.5
    inter = Σ_c (d_c / 2m)^2           Shi Eq. 3.6

- `m` = `graph.edges.len()`, the number of undirected edges.
- `l_c` = edges with both endpoints in `c`, counted **once** (`objectives.rs`
  scans each node's adjacency and takes only `node < neighbor`).
- `d_c` = `Σ_{v ∈ c} deg(v)`, so every internal edge contributes twice.

The pair is an exact decomposition of Newman modularity:

    Q = 1 − intra − inter

so minimising the two jointly is maximising `Q` along the ray `intra + inter`,
and the Pareto front is the trade-off between few cut edges (low `intra`) and
many small communities (low `inter`).

A graph with `m = 0` short-circuits to `Metrics::default()`, i.e. `(0, 0)`.

**Representation**

Label-based, no decode step: an individual is a `Partition =
FxHashMap<NodeId, CommunityId>` holding one community id per node, plus its
objective vector, its NSGA-II rank and its crowding distance.

`generate_population` draws each node's label uniformly from `0..n` where `n` is
the node count, so the initial population sits at the maximally fine end of the
front. Isolated nodes carry a label through the search — they cannot move, since
mutation needs a neighbour — and `normalize_community_ids` rewrites them to `-1`
on the way out, renumbering the surviving communities from `0`.

**Algorithm**

1. `generate_population` — `pop_size` uniformly random labellings; evaluate.
2. Per generation `t = 0..num_gens`:
   a. `select_survivors` — fast non-dominated sort, then per-front crowding
      distance, then sort by (rank ↑, crowding ↓) and truncate the pool to
      `pop_size`.
   b. `create_offspring` — `pop_size` children. Each child draws
      `ENSEMBLE_SIZE = 4` **distinct** parent indices by binary tournament
      (`TOURNAMENT_SIZE = 2`, better rank wins, ties on higher crowding). With
      probability `cross_rate` the child is `ensemble_crossover` of those four
      (per node, the community most parents give it; a tie is broken by a uniform
      draw over the tied communities); otherwise it is a clone of one of the four,
      drawn uniformly. Then `mutation`: every node is drawn with probability
      `mut_rate` and moved to the community held by most of its neighbours.
   c. Evaluate the children, append them to the population, fire
      `on_generation`.
3. `evolve` returns the merged `2 × pop_size` pool **unranked on its offspring
   half**; `HpMocd::search` filters `rank == 1` (see *Divergences*).
4. Selection rule: `run` returns the rank-1 member of highest modularity,
   read as `n − Σ objectives` (`select.rs`). `generate_pareto_front` returns the
   whole rank-1 set as `[(partition, objectives), ...]`.

**Parameters**

| Parameter | Default | Effect |
|---|---|---|
| `graph` | — | `networkx.Graph` or `DiGraph` with integer node ids. |
| `debug_level` | `0` | `0` silent; `≥1` prints the graph once and logs every 10th generation and the last. |
| `pop_size` | `100` | NSGA-II population size. Must be `≥ 4` (see *Divergences*). |
| `num_gens` | `100` | Generations. |
| `cross_rate` | `0.7` | Probability a child is an ensemble crossover rather than a parent clone. Must be in `[0, 1]`. |
| `mut_rate` | `0.5` | Per-node probability of a neighbour-majority move. Must be in `[0, 1]`. |
| `objectives` | `None` | List of Python callables `(graph, partition_dict) -> float`, all minimised. Non-empty replaces the built-in `intra`/`inter` pair. Also settable after construction with `set_objectives`; an empty list reverts. |

`set_on_generation(cb)` registers `cb(generation, num_gens, front_size)`, called
once per generation after the offspring are merged; `None` clears it.

Internal constants: `TOURNAMENT_SIZE = 2`, `ENSEMBLE_SIZE = 4`,
`LOG_EVERY_GENERATIONS = 10`, and four input-size cut-overs —
`PARALLEL_COMMUNITY_THRESHOLD = 8` (objective folding goes to rayon above it),
`PARALLEL_NODE_THRESHOLD = 128` and `PARALLEL_CHUNK = 64` (mutation), plus the
allocation-only `PRESIZED_DRAW_MAX_RATE = 0.5`, `DRAW_CAPACITY_SLACK = 1.2` and
`FREQ_CAPACITY = 16`.

**Determinism**

**Nondeterministic: two consecutive calls on the same graph differ.** Measured on
karate, lesmis and florentine; only the trivial graphs (empty, single node) and
graphs whose front collapses to one partition come back stable. Over 1000 runs on
karate the detector spans `k ∈ {3, 4, 5}` with AMI ≈ 0.255 ± 0.016.

Five independent sources, none of them switchable off — there is no seed
parameter anywhere in the module:

1. **Unseeded RNGs.** `rand::rng()` (the thread-local generator) is taken fresh in
   `operators/init.rs`, once per child in `nsga2/offspring.rs`, and once per
   `mutation` call. Nothing salts them by slot, so the draws differ run to run.
2. **Rayon work distribution.** `create_offspring` calls `rng()` inside a
   `par_iter`, so which thread's stream serves which child depends on the
   scheduler.
3. **Front peeling order.** `fast_non_dominated_sort` peels each front with
   `par_iter().fold(..).reduce(..)`; membership of `next_front` is deterministic
   but its *order* is not, and nothing sorts it. That order survives into
   `sort_unstable_by` in `select_survivors`, which cannot separate individuals
   equal on (rank, crowding). (MR-MOCD sorts each peeled front for exactly this
   reason; HP-MOCD does not.)
4. **Hash iteration order.** `partition.keys()` fixes the order of the mutation
   draws and of `ensemble_crossover`'s node scan, and `community_counts.iter()`
   fixes the candidate order handed to `choose` in the crossover tie-break.
5. **Float summation order.** `calculate_objectives` folds over communities with
   rayon above `PARALLEL_COMMUNITY_THRESHOLD`, so the low bits of `intra` and
   `inter` — and therefore some dominance comparisons — vary between runs.

**Divergences**

Taken from the code and from the comments this cleanup replaced; nothing here is
inferred from the paper text.

- **The last generation's offspring are discarded.** `Individual::new` sets
  `rank = usize::MAX`, `evaluate` only writes `objectives`, and `evolve` returns
  the merged pool without a final `fast_non_dominated_sort`. The `rank == 1`
  filter in `api.rs` therefore only ever sees the survivors ranked at the *top*
  of the last generation; the `pop_size` children bred and evaluated in that
  generation cannot reach the front. Left as-is: it is the shipped behaviour the
  published results were produced with.
- **An early majority short-circuits the crossover tie-break.** In
  `ensemble_crossover`, `if *count >= majority_threshold { break }` stops the vote
  as soon as a community reaches `⌈|parents|/2⌉`. With the shipped four parents
  the threshold is 2, so an even 2–2 split is settled by whichever community
  reached 2 first — in parent order — and never reaches the uniform tie draw
  below it.
- **Mutation has two non-equivalent sweeps.** Below `PARALLEL_NODE_THRESHOLD`
  drawn nodes, `sequential_mutate` is Gauss–Seidel: a node sees moves made
  earlier in the same sweep. Above it, `parallel_mutate` is Jacobi: every node is
  scored against the pre-sweep labels and all moves are applied afterwards. The
  drawn-node count alone picks the path.
- **`pop_size` must be at least `ENSEMBLE_SIZE`.** `create_offspring` loops
  `while unique_parents.len() < 4`, so a population of fewer than four
  individuals never terminates. Nothing validates it — unlike `MoPots`, the
  constructor performs no parameter checks at all, and an out-of-range
  `cross_rate`/`mut_rate` panics inside `Bernoulli::new(..).unwrap()`.
- **Max-Q selection assumes the built-in encoding.** `select.rs` ranks the front
  by `n − Σ objectives`, an order-preserving shift of `Q = 1 − intra − inter`.
  With custom Python objectives the same expression still decides `run`'s answer,
  and it is then not modularity — use `generate_pareto_front` (or
  `pymocd.hpmocd_fronts`) and select outside.
- **Python objectives run sequentially and share one dict.** A single `PyDict` is
  allocated and cleared per individual, so a Python objective must not keep a
  reference to the partition dict between calls. The Python graph object is
  stored unconditionally, whether or not objectives were passed, so
  `set_objectives` works after construction.
- **The module-level functions are the supported route.** `pymocd.hpmocd` and
  `pymocd.hpmocd_fronts` in `src/api/detectors.rs` construct `HpMocd` with the
  defaults above; whether the class itself is exported to Python depends on
  `src/lib.rs`, and the front is reachable through `hpmocd_fronts` either way.
- Historical note, for anyone diffing against an older checkout: the individual
  and NSGA-II files used to describe themselves as a core shared with an `nsga3`
  engine, and to point at MR-MOCD's separate swarm engine. No engine outside
  this directory reads these files today; the sharing never existed in this
  module's tree.

**Files**

| Path | Holds |
|---|---|
| `mod.rs` | Module wiring; re-exports `HpMocd` and the defaults. |
| `api.rs` | The `HpMocd` pyclass: Python surface, objective dispatch (Rust or Python), per-generation reporting, and `search`, the rank-1 filter behind `run`/`generate_pareto_front`. |
| `defaults.rs` | The five shipped defaults. |
| `objectives.rs` | Shi's `intra`/`inter` and the `Metrics` value type. |
| `select.rs` | `max_q_selection`: the highest-`Q` member of the front. |
| `nsga2/mod.rs` | NSGA-II wiring; exports `evolve`, `Individual`, `TOURNAMENT_SIZE`. |
| `nsga2/individual.rs` | The population member and Pareto dominance. |
| `nsga2/sorting.rs` | Fast non-dominated sort, domination matrix built in parallel. |
| `nsga2/survival.rs` | Crowding distance and truncation to `pop_size`. |
| `nsga2/offspring.rs` | Binary tournament, the four-parent ensemble draw, child production. |
| `nsga2/engine.rs` | `evolve`, the generational loop. |
| `operators/mod.rs` | Operator wiring. |
| `operators/init.rs` | Uniformly random initial population. |
| `operators/mutation.rs` | The node draw and the two neighbour-majority sweeps. |
| `operators/crossover.rs` | Per-node majority-vote crossover over the parent ensemble. |
