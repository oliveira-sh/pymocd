# MO-POTS — Multi-Objective Potts community detection

**Paper**

Santos, G. — MO-POTS, in preparation (2026). This project's own method; there is
no published paper to check the implementation against yet.

Objective pair: Traag, Van Dooren & Nesterov, "Narrow scope for
resolution-limit-free community detection", *Phys. Rev. E* 84:016114, 2011.
DOI 10.1103/PhysRevE.84.016114 — the Constant Potts Model, which this module
splits rather than scalarises.

Engine: Deb, Pratap, Agarwal & Meyarivan, "A fast and elitist multiobjective
genetic algorithm: NSGA-II", *IEEE TEC* 6(2):182–197, 2002.
DOI 10.1109/4235.996017.

Selector: Newman & Girvan, "Finding and evaluating community structure in
networks", *Phys. Rev. E* 69:026113, 2004. DOI 10.1103/PhysRevE.69.026113.

**Original implementation**

This project's own; there is no external reference implementation.

**Objectives**

Notation, over `G = (V, E)` with `m = |E|` (`graph.edges` holds every undirected
edge exactly once) and a partition `C = {V_1 … V_k}`:

    |E(c)|  edges with both endpoints in community c, counted once
    n_a     number of nodes of non-zero degree in G
    n_c     number of nodes of non-zero degree in community c
    C(k,2)  k(k−1)/2

Both objectives are **minimised**, and both are read over the `n_a` non-isolated
nodes only:

    cut  = 1 − Σ_c |E(c)| / m           fraction of edges leaving their community
    pair = Σ_c C(n_c,2) / C(n_a,2)      fraction of non-isolated node pairs co-clustered

`cut = 0.0` when `m = 0`; `pair = 0.0` when `C(n_a,2) = 0`, i.e. fewer than two
nodes carry an edge.

The pair is an exact affine decomposition of the Constant Potts Model
`H_γ(C) = Σ_c (|E(c)| − γ·C(n_c,2))`:

    H_γ(C)/m = 1 − cut − (γ/γ_d)·pair,      γ_d = 2m / (n_a(n_a−1)) = m / C(n_a,2)

so `γ` never enters the search: it is the exchange rate between the two axes, and
one Pareto front is the whole resolution sweep. `objectives::cpm` is that
identity, and `cpm_equals_the_potts_hamiltonian_over_m` pins it against a direct
evaluation of `H_γ(C)/m` at six `γ` values on six (graph, partition) cases,
including one with isolated nodes.

Both objectives are accumulated as integer counts and divided exactly once, so
the value does not depend on which of the sequential or rayon edge scan produced
it. `Σ_c C(n_c,2)` is the one float sum in the module (see **Determinism**).

**Representation**

Label-based, with no decode step: an individual is a
`Partition = FxHashMap<NodeId, CommunityId>` mapping every node of the graph to
a community id. `Individual` carries that partition, its objective vector
`[cut, pair]`, its NSGA-II rank and its crowding distance.

Initialisation draws each node's label uniformly from `0..n` where `n` is the
node count, so the population starts near the pair-minimal (maximally fine) end
of the front. Isolated nodes are labelled like any other node, are ignored by
both objectives, and are overwritten with `-1` by `normalize_community_ids` on
the way out.

**Algorithm**

1. `generate_population` — `pop_size` uniformly random labellings, one RNG stream
   per slot; evaluate `[cut, pair]`.
2. For each generation `t = 0 .. num_gens−1`:
   a. `select_survivors` — fast non-dominated sort, then per-front crowding
      distance, then sort by (rank ascending, crowding descending) and truncate
      the pool back to `pop_size`. On the first generation the pool is already
      `pop_size` long, so this is the *ranking* pass as much as a truncation.
   b. `create_offspring` — `pop_size` children, one RNG stream per `(t, slot)`.
      Each child draws `ENSEMBLE_SIZE = 4` *distinct* parents by binary
      tournament (rank ascending, then crowding descending, first drawn wins a
      tie); with probability `cross_rate` the child is `ensemble_crossover` of
      those parents (per node, the label most of them give it; a tie is drawn
      uniformly over the sorted tied labels), otherwise it is a clone of one
      uniformly drawn parent. `mutation` then draws each node with probability
      `mut_rate` and moves it to the community held by most of its neighbours.
   c. Evaluate the children and append them, giving a `2·pop_size` pool.
3. Rank the final pool once more; the caller keeps rank 1.
4. Selection: `mopots` returns the rank-1 member of highest Newman `Q`.
   `mopots_fronts` returns the whole rank-1 front. `mopots_ladder` returns that
   front's lower convex hull in (co-clustered pairs, intra edges) as one
   `(partition, cut, pair, γ)` row per resolution, `γ` increasing, the first row
   always labelled `γ = 0.0`.

**Parameters**

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 100 | NSGA-II population size. Must be non-zero (`ValueError` otherwise). |
| `num_gens` | 100 | Generations. `0` returns the evaluated initial population. |
| `cross_rate` | 0.7 | Probability a child is an ensemble consensus rather than a parent clone. Must be a finite probability in `[0,1]`. |
| `mut_rate` | 0.5 | Per-node probability of a neighbour-majority move. Must be a finite probability in `[0,1]`. `0.0` skips mutation entirely. |
| `debug_level` | 0 | `0` silent; `≥1` prints the graph once and the rank-1 front size every 10 generations. |

Internal constants a reader will hit:

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `TOURNAMENT_SIZE` | 2 | `nsga2/individual.rs` | Binary tournament. |
| `ENSEMBLE_SIZE` | 4 | `nsga2/offspring.rs` | Parents voting in one consensus crossover, clamped to `pop_size`. |
| `PARALLEL_EDGE_THRESHOLD` | 1024 | `objectives.rs` | Above this many edges the objective edge scan goes through rayon. |
| `PARALLEL_NODE_THRESHOLD` | 128 | `operators/mutation.rs` | Above this many *drawn* nodes mutation takes the parallel path. |
| `PARALLEL_CHUNK` | 64 | `operators/mutation.rs` | Nodes per rayon chunk on that path. |
| `DRAW_CAPACITY_SLACK` | 1.2 | `operators/mutation.rs` | Slack on the expected draw count, capacity only. |
| `INIT_SALT` | `0x5EED_0001` | `operators/init.rs` | RNG salt of the initial population. |
| `RNG_BASE` | `0x5CA1_E5EED` | `sampling.rs` | Base seed shared with the other detectors that use `slot_rng`. |

**Determinism**

**Bit-deterministic** (measured: two consecutive calls produce identical output,
and the module's digests are unchanged across refactors). What enforces it:

- Every draw comes from `slot_rng(salt, slot)`, one independent `StdRng` stream
  per `(salt, slot)` pair, so no draw depends on the thread count or on rayon's
  scheduling. The initial population uses `INIT_SALT`, generation `t`'s children
  use `t`; those two families must not collide, which is why `INIT_SALT` is a
  large constant and not, say, `0`.
- Node order is fixed by `Graph::finalize`, which sorts `node_vec` and every
  adjacency list. Initialisation, crossover and mutation all walk `nodes_vec()`,
  so the *order in which a stream is consumed* is fixed too.
- Rayon appears as `into_par_iter().map(…).collect()` over slot indices
  (`create_offspring`, `evolve`), as `par_iter_mut()` elementwise (evaluation),
  as `par_chunks(…).flat_map(…).collect()` (parallel mutation) — all
  order-preserving — and as `par_iter().filter().count()` (the edge scan), which
  is an integer count and so order-free.
- The one exception is `fast_non_dominated_sort`, which peels fronts with a
  rayon `fold`/`reduce` over an `AtomicUsize` per individual. Membership of the
  next front is deterministic but the *order* of the pushes is a race, so the
  front is `sort_unstable()`-ed before ranks are written. Deleting that sort
  makes the whole detector non-reproducible.
- Float accumulation order is pinned by data structure. `cut` counts edges as
  integers and divides once. `Σ_c C(n_c,2)` sums over an `FxHashMap`'s
  `values()`, which is reproducible because `FxHash` is unseeded, but is
  layout-dependent: it is the one place where a change of hasher would change
  the last bits of `pair`.
- Both rayon cut-overs (`PARALLEL_EDGE_THRESHOLD`, `PARALLEL_NODE_THRESHOLD`)
  are decided by input size alone, never by the available thread count — which
  matters because the two mutation paths are not the same operator (see
  **Divergences**).
- Ties are resolved by fixed rules, never by a fresh draw or by hash order:
  the binary tournament keeps the first-drawn index (`<` and `>`, never `≤`);
  the ensemble crossover sorts the tied labels ascending before drawing among
  them; neighbour-majority mutation keeps the first community to reach the peak
  count over the sorted adjacency list; the resolution ladder's convex-hull
  predicate is exact `i128` arithmetic with no tolerance at all.
- `select_survivors` and `calculate_crowding_distance` use `sort_unstable_by`.
  Unstable is not random: the same input gives the same permutation every run.

`tests/test_determinism.py` additionally runs the whole detector in
subprocesses pinned to 1 and 4 threads and compares the partitions;
`api.rs` does the same in-process over rayon pools of 1 and 4.

**Divergences**

No reference implementation exists, so these are the deliberate departures from
the conventions the two source papers set:

- **`γ` is not a search parameter.** The CPM is split into `(cut, pair)` rather
  than scalarised, so one run sweeps every resolution instead of committing to
  one. `gamma_d` therefore exists only to state the identity: the ladder cancels
  it out, and outside `#[cfg(test)]` the function is dead code.
- **Both objectives are scoped to the `n_a` non-isolated nodes**, not to all of
  `V`. An isolated node contributes no edge to `cut` and is excluded from `n_c`
  and `n_a` in `pair`; mutation cannot move it (it has no neighbour), and it is
  reported as community `-1`. This is the only objective scope in the repository
  that is not all of `V` — `smocc`'s ObjSet 20 borrows it, the others do not.
- **Crossover is a 4-parent consensus** of *distinct* tournament winners, not
  NSGA-II's canonical two-parent recombination. A node that no parent labels
  keeps its own id (the singleton label). The vote stops early once a community
  holds an unbeatable majority, which is a shortcut, not a change of result:
  genuine ties still reach the tie-break.
- **Survivor truncation happens at the top of each generation**, not at the
  bottom, and `evolve` returns the unfiltered `2·pop_size` pool with fresh ranks
  for the caller to filter at rank 1.
- **Mutation has two non-equivalent paths.** At or below
  `PARALLEL_NODE_THRESHOLD` drawn nodes it is Gauss–Seidel (a node sees the moves
  made earlier in the same sweep); above it, it is Jacobi (every move is computed
  against the pre-sweep labels and applied afterwards). The input alone picks the
  path, so a run stays reproducible, but the operator is not the same operator on
  both sides of the threshold.
- **Mutation is not a hill-climb.** A drawn node moves to its neighbour-majority
  community whether or not that improves either objective.
- **`max_q_selection` recomputes Newman `Q`** from the partition instead of
  reusing HP-MOCD's `Q = n − Σ objectives` shortcut, which needs the intra/inter
  encoding that `(cut, pair)` does not carry.
- **The ladder omits front members that sit in a concave dent of the hull**, and
  members exactly collinear with a hull segment: neither is the strict CPM
  optimum at any `γ`, so `mopots_ladder` is a subset of `mopots_fronts`. The
  first rung is always labelled `γ = 0.0`, whether or not the front happens to
  contain a single-community member.
- **The ladder's hull predicate is exact integer arithmetic.** Objective values
  are turned back into their integer numerators (`round()` of `ratio ×
  denominator`, exact below `2^53`) and the turn is an `i128` cross product. An
  earlier relative-tolerance version dropped genuine vertices on large graphs,
  which `a_vertex_clearing_the_chord_by_one_pair_survives_at_scale` now guards.

**Files**

| Path | Holds |
|---|---|
| `mod.rs` | Module root: submodule wiring, the `MoPots`/defaults re-export, and the `#[cfg(test)]` `calculate_objectives` re-export that `smocc`'s CSR `cpm()` is cross-checked against. |
| `api.rs` | The `MoPots` pyclass: parameter validation, `search`, `run` / `generate_pareto_front` / `ladder`, and the two determinism fingerprint tests. |
| `defaults.rs` | The five shipped defaults. |
| `objectives.rs` | `(cut, pair)`, their denominators, `gamma_d`, `cpm`, and the CPM identity test. |
| `sampling.rs` | `slot_rng` and `bernoulli`: the per-slot RNG contract. |
| `fixtures.rs` | The two-triangles test graph shared by the module's unit tests (`#[cfg(test)]` only). |
| `operators/init.rs` | Uniformly random initial population, one stream per slot. |
| `operators/crossover.rs` | Consensus (ensemble) crossover with sorted uniform tie-breaks. |
| `operators/mutation.rs` | Neighbour-majority mutation: the draw, the Gauss–Seidel path, the Jacobi path and the move they share. |
| `nsga2/individual.rs` | The population member and Pareto dominance. |
| `nsga2/sorting.rs` | Fast non-dominated sort (Deb et al. 2002), rayon front peeling. |
| `nsga2/survival.rs` | Crowding distance and truncation to `pop_size`. |
| `nsga2/offspring.rs` | Binary tournament, distinct parent ensembles, child production. |
| `nsga2/engine.rs` | The generational loop `evolve`. |
| `front/select.rs` | `max_q_selection`, the single-partition decision rule. |
| `front/ladder.rs` | The resolution ladder: exact `i128` lower convex hull, `γ` per rung. |
