# GDPSO — Greedy Discrete Particle Swarm Optimization

**Paper**
Cai, Q., Gong, M., Ma, L., Ruan, S., Yuan, F. & Jiao, L., "Greedy discrete
particle swarm optimization for large-scale social network clustering",
*Information Sciences* 316:503–516, 2015. DOI 10.1016/j.ins.2014.10.041.

Objective: Newman, M. E. J. & Girvan, M., "Finding and evaluating community
structure in networks", *Physical Review E* 69:026113, 2004.
DOI 10.1103/PhysRevE.69.026113.
Velocity rule: Kennedy, J. & Eberhart, R., "A discrete binary version of the
particle swarm algorithm", *Proc. IEEE SMC*, pp. 4104–4108, 1997.
DOI 10.1109/ICSMC.1997.637339.
Constriction constants: Clerc, M. & Kennedy, J., "The particle swarm — explosion,
stability, and convergence in a multidimensional complex space", *IEEE TEC*
6(1):58–73, 2002. DOI 10.1109/4235.985692.

**Original implementation**
https://github.com/doctor-cai/GDPSO — C++, no licence. This module was written
clean-room from a description of that implementation; no reference source was
read while writing it, so every claim below about "the reference" comes from
the paper or from that description, not from copied code.

**Objectives**
One objective, **maximised**: Newman–Girvan modularity.

    Q = Σ_c [ e_c / m − (K_c / 2m)² ]

`m` is the edge count, `e_c` the number of edges internal to community `c`
counted once each, and `K_c = Σ_{i∈c} deg(i)` that community's degree sum
(`sigma[c]` in the code). An edgeless graph is defined to have `Q = 0` instead
of dividing by zero. There is no second objective and no Pareto front: the
answer is the single best position the swarm ever held, so there is no
`gdpso_fronts` entry point.

The greedy step ranks candidate moves by the comparable half of the exact
single-node gain. For node `i` of degree `k_i`, with `κ_{i,l}` its edge count
into community `l`:

    score(l) = κ_{i,l} − k_i · Σ_l / 2m
    ΔQ(i → l) = ( score(l) − score(own) ) / m

where `Σ_own` **excludes** `k_i` itself. `score` is monotone in `ΔQ` for a fixed
`i`, so ranking by `score` and accepting only a strict increase is exactly
best-improvement with a strictly positive gain. `objective::delta_q` computes
the full identity and exists only to pin `move_score` against it in tests.

**Representation**
Label-based, no decode. A particle's *position* is a `Vec<i32>` of length `n`,
one community label per dense CSR node index; the partition is read off it
directly. After every write the position is canonicalised by first appearance,
so the label set is always `0..k` — that is not cosmetic, it is what keeps every
label a valid index into the `n`-sized `kappa`/`remap`/`sigma` scratch arrays.

A particle's *velocity* is a `Vec<bool>`, not a displacement: it is a per-node
mask saying which nodes are offered a greedy move this generation. `pbest` is a
second label vector; `gbest` is one shared label vector.

Initialisation is a short asynchronous label-propagation run from all-singletons
(`lpa_seed`, `lpa_sweeps` sweeps, fixed node order, uniform tie-breaking among
the modal neighbour labels). Sweep 1 has every label distinct, so every node
adopts a uniformly random neighbour: that first sweep is the entire source of
inter-particle diversity, which makes `lpa_sweeps`, not `num_gens`, the lever on
how many distinct seeds the swarm actually explores.

Isolated nodes are carried through untouched by every operator and are reported
as community `-1` at the boundary.

**Algorithm**
1. Build the CSR graph. With `n == 0` or `m == 0` return all-singletons, which
   the boundary turns into all `-1`.
2. Init, one RNG stream per slot (`slot_rng(INIT_SALT, slot)`), in parallel:
   `lpa_seed` → `canonicalise` → `degree_sums` → `Q`. The seed is the particle's
   first `pbest`.
3. `elect`: scan the swarm in slot order and take the strictly best `Q` as
   `gbest`.
4. For each generation `t = 0..num_gens`, each particle in parallel with one
   stream `slot_rng(t, slot)`:
   a. `update_velocity` — per node: `s = w·[mask] + c1·r1·[pos ≠ pbest] +
      c2·r2·[pos ≠ gbest]`, then `mask = U(0,1) < σ(s)`. Three draws per node,
      always. (With `const_move_prob` set, `randomise_velocity` replaces this
      whole step.)
   b. `greedy_sweep` — the masked Gauss-Seidel sweep: each masked node, in
      increasing index order and in place, takes the neighbouring community with
      the largest `move_score`, ties to the smallest label, and only on a
      strictly positive gain. `sigma` is maintained incrementally.
   c. `canonicalise`.
   d. If the slot is selected for mutation, `mutate` (each node with probability
      `mut_rate` broadcasts its own label over all its neighbours), then
      `canonicalise` again.
   e. Recompute `Q`; if it beats `pbest_fitness`, `pbest` takes the position.
      There is no rejection step — the particle keeps whatever it landed on.
   f. After every particle has moved, `elect` again. `gbest` is frozen for the
      whole generation.
5. Return `gbest`. The boundary renumbers communities densely in ascending node
   order and gives isolated nodes `-1`.

Mechanically this is a population of LPA-seeded, randomly masked Louvain
local-moving sweeps with a coarsening mutation on a tenth of the swarm: the
cognitive and social terms carry no label information (see **Divergences**), so
every label decision is made by the greedy gain over a node's own neighbourhood.

**Parameters**
| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 100 | Number of particles. Sanitised to at least 1. |
| `num_gens` | 250 | Generations. 0 returns the best seed. |
| `w` | 0.7298 | Inertia weight on the previous move mask. |
| `c1` | 1.4961 | Cognitive weight, applied to the `pbest` disagreement indicator. |
| `c2` | 1.4961 | Social weight, applied to the `gbest` disagreement indicator. |
| `mut_rate` | 0.1 | Per-node label-broadcast probability inside a mutated particle. |
| `mut_frac` | 0.1 | Fraction of the swarm mutated each generation. |
| `lpa_sweeps` | 5 | Asynchronous label-propagation sweeps seeding each particle. |

`Config`-only, not reachable from the Python entry point and not part of the
method:

| Field | Default | Effect |
|---|---|---|
| `mut_by_index` | `true` | `true` reproduces the reference: the first `floor(pop_size · mut_frac)` slots are mutated every generation. `false` mutates each particle with probability `mut_frac`, the reading that rule most likely intended. |
| `const_move_prob` | `None` | Ablation: replace the whole velocity update with one fixed move probability, which tests whether the PSO layer carries any information at all. |

`w`, `c1` and `c2` are the Clerc constriction constants, inherited from
real-valued PSO. Because the velocity is re-binarised every generation they
carry no convergence theory here; they are reproduced as the method, not "fixed"
into canonical binary PSO. They do fix the sigmoid band: the argument lies in
`[0, w + c1 + c2] = [0, 3.722]`, so a node's move probability is always inside
`[0.5, 0.976]`. `Config::sanitized` is the one choke point every rate and
coefficient passes through — non-finite or negative coefficients map to `0.0`,
rates clamp into `[0, 1]`.

The reference ships 100 particles and 250 generations; its own in-source
comments recommend 20–60 particles and 50–300 generations, and the published
tables used 100 × 100. On karate the reference reports a best `Q = 0.41979` over
30 runs — the four-group modularity optimum. With the shipped `pop_size = 100`
this module lands strictly short of it (five propagation sweeps leave only ~19
distinct seeds), and reaches it at `pop_size = 300`; both facts are pinned by
`api.rs::karate_reaches_the_reference_modularity`.

GDPSO inherits modularity's resolution limit whole, with no mitigation, so
expect over-coarsening on large sparse graphs.

Internal constants: `INIT_SALT = 0x5EED_0002` (the seeding RNG salt, held clear
of the generation counters that salt the per-generation streams) and
`RNG_BASE = 0x5CA1_E5EED` in `sampling.rs`.

**Determinism**
Bit-deterministic (measured: `api.rs::two_runs_are_byte_identical`,
`swarm/engine.rs::thread_count_does_not_change_the_result`). What enforces it:

- Every draw comes from `slot_rng(salt, slot)`, one independent stream per
  `(salt, slot)` pair, so no draw depends on the thread count or on rayon's
  scheduling. Seeding uses `INIT_SALT`; generation `t` uses `t`.
- `update_velocity` draws `r1` and `r2` per node whatever the two indicators
  are, so stream consumption is a fixed three draws per node and cannot depend
  on the position.
- `gbest` is frozen for the whole generation, so particles inside one generation
  are genuinely independent and the rayon pass is byte-exact.
- `elect` runs sequentially over the swarm in slot order with a strict `>`, so a
  tie never displaces the incumbent and the lowest slot wins among equal
  challengers.
- `modularity` sums the null-model term in label-index order, a fixed summation
  order, so `Q` is bit-reproducible.
- `greedy_sweep` breaks score ties to the smallest label; `lpa_seed` breaks
  modal ties by a draw over `dirty`, which is built in neighbour order from the
  CSR adjacency, not from a hash container.

**Divergences**
Written clean-room, so these are departures from the paper and from the
described reference behaviour, not from source that was read.

- The pre-shuffle before each label-propagation sweep is omitted. On sweep 1 it
  permutes pairwise distinct labels and is therefore semantically a no-op.
- The seed counts as the particle's first personal best; the reference discards
  it and starts `pbest` from the first post-velocity position.
- `mut_rate` and `mut_frac` are two knobs. The reference overloads a single
  `0.1` for both the per-node broadcast probability and the fraction of the
  swarm that is mutated.
- Which particles mutate is a documented ambiguity, resolved by `mut_by_index`.
  The default (`true`) reproduces the reference's "first `floor(P · pm)` slots,
  every generation"; the probabilistic reading is available as an ablation.
- `const_move_prob` is not part of the method at all: it exists so the PSO layer
  can be ablated without a code change.
- Modularity is computed in the `O(n + m)` sparse form instead of the
  reference's dense double loop, and `sigma` is recomputed per particle per
  generation rather than carried across operators, which keeps it exact and free
  of drift (asserted in `local.rs::sweep`).
- In `greedy_sweep` a degree-1 node takes its neighbour's community
  unconditionally, with no gain test. This is the one gain-free move in the
  method and the only step that can lower `Q`; it reproduces the reference and
  must not be "fixed".
- In `mutate` a degree-1 node is pulled inward — it adopts its neighbour's label
  instead of broadcasting its own — which is the opposite direction to the
  operator's normal action.
- The cognitive and social terms transfer no structure. `pbest` and `gbest`
  enter only as two indicator bits (`position[i] != best[i]`), read off
  canonical labels, that shift a node's move probability inside `[0.5, 0.976]`.
  That comparison is order-dependent and is *not* a partition distance; it only
  gates move attempts. The code looks as though it should copy labels and
  deliberately does not.
- The stored velocity is re-binarised every generation, so the inertia term is a
  one-bit memory rather than a real velocity.
- `CsrGraph::from_edges` drops self-loops and keeps a repeated edge, counting it
  twice — the multigraph reading. `Q` and its gain stay consistent either way.
- Isolated nodes are outside every operator and are reported as community `-1`
  rather than as singleton communities.

**Files**
| Path | Holds |
|---|---|
| `mod.rs` | Module root; re-exports the three entry points, `Config` and the defaults. |
| `api.rs` | `gdpso`, `gdpso_with`, `gdpso_on_graph`, the `CsrGraph` → `Partition` boundary, and the end-to-end tests. |
| `config/mod.rs` | Config submodule root. |
| `config/defaults.rs` | The eight shipped defaults. |
| `config/params.rs` | The `Config` struct, its `Default`, and `sanitized`. |
| `sampling.rs` | `slot_rng` (the per-slot RNG contract) and the `probability` / `coefficient` clamps. |
| `objective.rs` | `degree_sums`, `modularity`, `move_score`, and the test-only exact `delta_q`. |
| `local.rs` | `greedy_sweep`: the masked Gauss-Seidel sweep of exact single-node modularity moves. |
| `swarm/mod.rs` | Swarm submodule root. |
| `swarm/particle.rs` | `Particle`, the shared `Workspace` scratch, and both velocity updates. |
| `swarm/operators.rs` | `canonicalise`, `lpa_seed`, `mutate`. |
| `swarm/engine.rs` | `run`: the generational loop, and the `elect` scan that refreshes `gbest`. |
