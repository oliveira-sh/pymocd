# MMCoMO — Macro-Micro population-based Co-evolutionary Multi-Objective community detection

**Paper**

Zhang, L., Yang, H., Yang, S. & Zhang, X., "A Macro-Micro Population-Based
Co-Evolutionary Multi-Objective Algorithm for Community Detection in Complex
Networks" [Research Frontier], *IEEE Computational Intelligence Magazine*
18(3):69–86, 2023. DOI 10.1109/MCI.2023.3277773.
<https://ieeexplore.ieee.org/document/10188453>

Bracketed reference numbers below (`[38]`, `[46]`, `[68]`) are that paper's own
reference indices, kept because the paper defers three design decisions to them.

Objectives: the KKM / RC pair, attributed in the source to Shi, Yan, Cai & Wu,
"A multi-objective approach for community detection in complex network",
*Applied Soft Computing* 12(2):850–859, 2012, and to Gong et al.; the exact
formulation used here is the one of Gong, Cai, Chen & Ma, "Complex Network
Clustering by Multiobjective Discrete Particle Swarm Optimization Based on
Decomposition", *IEEE TEC* 18(1):82–97, 2014.

Engine: Deb, Pratap, Agarwal & Meyarivan, "A fast and elitist multiobjective
genetic algorithm: NSGA-II", *IEEE TEC* 6(2):182–197, 2002.
DOI 10.1109/4235.996017.

Similarity: Kondor & Lafferty, "Diffusion Kernels on Graphs and Other Discrete
Input Spaces", *ICML* 2002, pp. 315–322 — the paper's ref [68].

Local search: the paper pins only the target (rank-1 micro members) and the
objective (Newman Q) and defers the move set to its ref [38]; this
implementation uses the first phase of Blondel, Guillaume, Lambiotte &
Lefebvre, "Fast unfolding of communities in large networks", *J. Stat. Mech.*
P10008, 2008.

**Original implementation**

None published. Neither the authors' site nor GitHub carries code for it, so
this module is a self-contained reimplementation from the paper alone; no
reference binary or trace exists to diff against.

[`smocc/`](../smocc/README.md) is **this project's own optimised variant of
MMCoMO**. It keeps the two-population macro-micro structure and replaces
essentially everything under it: sparse per-edge similarity instead of the dense
`O(n²)` kernel (so no `beta`), the local search deleted outright, HP-MOCD micro
operators, optionally different objective pairs per swarm, union refinement of
the final front, and a label-free scalarisation selector instead of max-`Q`.
It is a separate detector with its own paper, not a tuning of this one; this
module exists to be the faithful baseline that `smocc/` is measured against.
Its own divergence list is in [`smocc/README.md`](../smocc/README.md).

**Objectives**

Both **minimised**, over `G = (V, E)` with `n = |V|`, a partition
`C = {V_1 … V_k}` where `k` is the number of *distinct labels present*, and

    deg(v)                                       degree of v
    L(V_i, V_i)  = Σ_{u,v ∈ V_i} A_uv            twice the internal-edge count of V_i
    L(V_i, V̄_i)  = Σ_{v ∈ V_i} deg(v) − L(V_i,V_i)   cut mass of V_i

the paper's Eq. 1 is

    KKM = 2(n − k) − Σ_i L(V_i, V_i) / |V_i|     kernel k-means
    RC  =            Σ_i L(V_i, V̄_i) / |V_i|     ratio cut

`KKM` falls as communities become internally dense per member and as `k` rises;
`RC` rises as `k` rises. That opposition is the front. Both are scoped to **all
`n` nodes**, isolated ones included, and an isolated node contributes `0` to
both numerators — but *not* always `1` to `k`. In a micro individual it starts
as its own community, so it does add one to `k`; in a macro individual `decode`
attaches it to an existing centre (see below), where it adds `1` to that
community's `|V_i|` and so lowers both `L(V_i,V_i)/|V_i|` and
`L(V_i,V̄_i)/|V_i|`. Isolated nodes therefore perturb the objective values, even
though the reported partition always maps them to `-1`.

Newman modularity

    Q = Σ_i [ e_i/m − (d_i/2m)² ]

with `e_i` the internal edges of `V_i`, `d_i` its degree mass and `m = |E|`, is
**maximised**. It is not a search objective: it is the local-search ascent
criterion (Alg. 1 line 11) and the rule that picks one partition out of the
final front.

**Representation**

Two co-evolving populations of size `pop_size` each, over one index-space graph,
exchanging through a dense similarity matrix.

- **Micro — label-based, no decode.** `Labels = Vec<i32>`, one community label
  per node index. **Every label is itself a node index in `[0, n)`** — the
  objectives, the local search and the vote matrix all index accumulators of
  length `n` by label directly, so a label outside that range would panic. The
  invariant survives because every operator either copies an existing label or
  writes a node index. Initialisation gives each node the index of a uniformly
  drawn neighbour (an isolated node takes its own index).
- **Macro — centre-indicator genome with a decode.** `Genome = Vec<u8>`, one bit
  per node marking it a community centre. `decode` (Eqs. 3–5) labels each centre
  with its own index and gives every other node the centre of greatest
  similarity, `argmax_{c ∈ CN} SM[i][c]`. `encode` (Eq. 8) inverts it: per
  community, the member of greatest summed similarity to the rest becomes the
  centre; a singleton community is its own centre. An isolated node has a
  similarity row that is zero everywhere off its own diagonal, so unless it is
  itself a centre the argmax scan — which keeps the *first* maximum — always
  hands it to the lowest-indexed centre in `CN`.
- **`Sm = Vec<Vec<f64>>`** — the dense `n × n` diffusion kernel
  `SM = exp(β(A − D))`, computed once and thereafter *relaxed* toward the micro elites'
  consensus at every exchange. Both it and the per-exchange vote matrix are
  `O(n²)` in memory and the eigendecomposition is `O(n³)`, which is what caps
  this detector's usable graph size — `benchmarks/_exp_synt_net/algorithms.py`
  registers MMCoMO with `max_nodes=2000` (the hardened harness in
  `hardened.py` sets no cap of its own).

**Algorithm**

1. `SM ← exp(β(A − D))` by symmetric Jacobi eigendecomposition.
2. Init `pop` micro individuals (random-neighbour labels) and `pop` macro
   individuals (`c ∈ [1, ⌈√n⌉]` centres; the first half shuffles the `min(3c, n)`
   highest-degree nodes and takes `c`, the second half draws `c` uniformly).
   Macro individuals are decoded; both swarms are then evaluated under
   `(KKM, RC)`.
3. For `t = 1 … num_gens`:
   a. Rank + crowd the micro population; produce `pop` children — clone a
      binary-tournament winner, then with probability `cross_rate` **graft**
      (copy every node a second parent puts in a random node's community), then
      mutate each node with probability `1/n` to a random neighbour's current
      label (Gauss–Seidel: a node sees moves made earlier in the sweep).
   b. Rank + crowd the macro population; produce `pop` children by uniform
      crossover of two tournament winners plus per-bit flips at probability
      `mut_rate`; an all-zero child is repaired with one random centre. Decode
      and evaluate.
   c. If `t mod gap ≠ 0`, each swarm truncates parents+children back to `pop` by
      NSGA-II environment selection, independently.
      If `t mod gap = 0`, the swarms **exchange** first:
      - *guidance* (Alg. 2) — every rank-1 macro member is decoded **afresh with
        the current SM**, re-evaluated, and thrown into the micro pool before
        truncation;
      - *local search* (Alg. 1 line 11) — Louvain first-phase Q-ascent, in
        place, on the surviving rank-1 micro members, which are then re-scored;
      - *influence* (Alg. 3) — the rank-1 micro members are the elites. The vote
        matrix `SM^v` gives every pair of nodes an elite places together
        `1/|elites|`; then Eq. 7 relaxes `SM ← (1−ρ)SM + ρSM^v` with
        `ρ = 0.5·t/num_gens`; then each elite is `encode`d, re-`decode`d and
        thrown into the macro pool before truncation.
4. Mergence (Phase 3): rank the union of both final populations and keep rank 1.
   An empty front falls back to the all-singletons partition.
5. Selection rule: `mmcomo` returns the front member of highest Newman `Q`
   (Table III). `mmcomo_fronts` returns the whole merged front instead, which is
   what the paper's Table IV best-NMI rule needs.

**Parameters**

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 100 | Size of **each** swarm, so `2·pop_size` individuals are carried. |
| `num_gens` | 50 | Generations. The search always runs all of them; there is no convergence stop. |
| `cross_rate` | 0.1 | **Micro** graft probability only. Macro crossover is unconditional and uniform. |
| `mut_rate` | 0.1 | **Macro** per-bit flip probability only. The micro mutation rate is fixed at `1/n` and is not exposed. |
| `gap` | 10 | Co-evolution interval: the swarms exchange, and the local search runs, every `gap` generations. Clamped up to `1`, so `0` means every generation. |
| `beta` | 0.05 | Diffusion-kernel exponent. Never specified in the paper; a reimplementation choice, exposed for tuning. `beta = 0` makes `SM` the identity. |

Internal constants: `LOCAL_SEARCH_SWEEP_CAP = 64` and `MIN_GAIN = 1e-12`
(`operators/local_search.rs`); `MAX_JACOBI_SWEEPS = 100`, `OFF_DIAG_TOL = 1e-30`
and `MIN_PIVOT = 1e-300` (`similarity/kernel.rs`); the macro centre ceiling
`⌈√n⌉` and its `3c` sampling pool (`macro_micro/init.rs`).

**Determinism**

**Nondeterministic**: two consecutive calls on the same graph can return
different partitions, and there is no seed parameter. Every draw comes from
`rand::rng()`, the thread-local generator seeded from the OS, taken fresh in
`init_micro` (once per individual), `init_macro`, `micro_offspring` and
`macro_offspring`. That is the only source: everything downstream of the draws
is order-stable —

- both objectives and the modularity sum over a `Vec` in first-seen-label order
  (`objectives::community_stats`), never over a hash map;
- adjacency lists are sorted at construction and `init_macro`'s degree sort is
  `sort_unstable_by`, unstable but not random;
- the local search sorts its candidate communities ascending, so equal-gain ties
  resolve to the lowest community id;
- none of the module's hash containers can leak iteration order into a result.
  The three `HashMap`s — `graph::build`'s id→index table, `graph::to_output`'s
  label remap and `codec::encode`'s label lookup — are only ever read by key,
  walked in node order. Of the two `HashSet`s, `graph::from_indexed`'s
  neighbour sets are collected and `sort_unstable`ed before use, and
  `init_macro`'s `chosen` set is iterated only to set genome bits, which is
  order-independent (its rejection-sampling loop does consume a variable number
  of draws, but that is a property of the draws, not of the set);
- there is no parallelism at all in this module, so no float sum is reassociated
  by a scheduler.

Consequently the *spread* of a run is small and structural. Measured on karate,
30 runs: AMI `0.239848 ± 0.000000`, `k = 4.00 ± 0.00`, and all 30 label vectors
identical — the search converges to the same partition every time on that
graph, which is a property of the graph, not a guarantee.

**Degenerate inputs**

Unlike `mocd_q`, this detector raises nothing on the trivial cases; each has a
defined answer.

| Input | Result |
|---|---|
| Empty graph | `{}` — `build` yields `n = 0` and `api` returns before the search. |
| Single isolated node | `{0: -1}`. |
| `pop_size = 0` | Both swarms init empty, so the merged front is empty and the all-singletons fallback in `mergence` fires: every node its own community. This is the only way to reach that fallback — on a non-empty population `fast_nondominated_sort` always yields at least one rank-1 member. |
| `num_gens = 0` | The generational loop never runs; the initial populations are merged as they stand. |
| `gap = 0` | Clamped to `1`: the swarms exchange every generation. |

`run_fronts` carries its own `g.n == 0` guard as well; it is unreachable
through `mmcomo`/`mmcomo_fronts`, which both return first.

**Divergences**

Every item below is traceable to the code or to a comment in it; the paper is
silent or deferring on each.

- **`beta` is invented.** The kernel exponent is never specified in the paper,
  so it is a public parameter with a `0.05` default rather than a constant.
- **Macro initialisation is unpinned.** The paper defers it to its ref [46]. The
  centre count is drawn from `[1, ⌈√n⌉]`, the high-degree half of the population
  samples `c` centres out of the `min(3c, n)` highest-degree nodes, and the rest
  draw uniformly. Both the ceiling and the `3c` pool are this implementation's.
- **The local-search move set is unpinned.** The paper fixes only the target and
  the objective and defers to its ref [38]; this is the Louvain first phase,
  capped at 64 sweeps, taking a move only when it beats staying by more than
  `1e-12`. Note that removing node `i` from its own community before scoring
  (`tot[ci] -= ki`) is part of the `ΔQ` formula, not a bug.
- **Eq. 4's denominator is skipped.** `decode` takes `argmax_{c} SM[i][c]`
  directly, because Eq. 4 normalises row `i` by a quantity constant in `c`, so
  the argmax is identical and one `O(n)` pass per node is saved.
- **Two degenerate genomes are repaired.** An all-zero genome decodes with the
  single max-degree node as its centre (Eq. 3 with `s = 0` has no partition);
  an all-zero *offspring* genome instead gets one uniformly drawn centre, so
  that repair does not collapse the whole macro swarm onto one node.
- **The eigendecomposition is self-contained.** `SM` is computed by cyclic
  Jacobi rotations written out in `similarity/kernel.rs` rather than by a LAPACK
  binding, to keep the crate dependency-free; `H = A − D` is rebuilt from `adj`
  and explicitly re-symmetrised so the solver's input is exactly symmetric.
- **Isolated nodes are a repo convention.** They take part in the search — they
  are never excluded from Eq. 1's sums, unlike `mopots`, which scopes its
  objectives to non-isolated nodes — but are reported as community `-1`, as
  every detector in this project does. Their in-search treatment is *not*
  uniform across the two swarms (see **Objectives**), which is a quirk of the
  representations rather than a decision.
- **`cross_rate` and `mut_rate` are not symmetric knobs.** `cross_rate` gates
  only the micro graft and `mut_rate` only the macro bit flips; micro mutation
  is fixed at `1/n` and macro crossover has no probability at all.
- **Guidance re-decodes.** Alg. 2 line 5 decodes each rank-1 macro elite again
  with the *current* SM instead of reusing the labels stored on the individual.
  The two differ, because SM has moved since the individual was built; keeping
  the stored labels is the faithfulness trap this module deliberately avoids.

**Files**

| Path | Holds |
|---|---|
| `mod.rs` | Module root: the `Sm`/`Labels`/`Genome` type aliases and their invariants, the submodule tree, and the public re-exports. |
| `api.rs` | `mmcomo` (max-Q member) and `mmcomo_fronts` (whole front), plus the end-to-end tests. |
| `defaults.rs` | The six shipped default constants. |
| `fixtures.rs` | Test-only graph builders shared by more than one test module. |
| `graph.rs` | The index-space `Graph`, the id→index build and the `-1`-for-isolated output mapping. |
| `objectives.rs` | `community_stats` (the shared first-seen compaction), Eq. 1's `(KKM, RC)`, and Newman `Q`. |
| `macro_micro/mod.rs` | Co-evolution submodule wiring. |
| `macro_micro/engine.rs` | Algorithm 1: breeding both swarms, the exchange schedule, the local-search step and the final mergence. |
| `macro_micro/init.rs` | Initial micro (random-neighbour) and macro (degree-biased centre) populations. |
| `macro_micro/exchange.rs` | `guidance` (Alg. 2), the elite vote matrix, the Eq. 7 relaxation and `influence` (Alg. 3). |
| `macro_micro/swarms.rs` | The `Mic`/`Mac` population members and NSGA-II survivor selection over pools of them. |
| `nsga2/mod.rs` | NSGA-II submodule wiring. |
| `nsga2/sorting.rs` | Pareto dominance and the fast non-dominated sort; ranks are 1-based. |
| `nsga2/crowding.rs` | Crowding distance per front, boundaries at infinity. |
| `nsga2/survival.rs` | Environment selection: whole fronts by rank, the overflowing one broken by crowding. |
| `operators/mod.rs` | Operator re-exports. |
| `operators/mating.rs` | Binary tournament: lower rank wins, larger crowding breaks the tie. |
| `operators/micro_labels.rs` | Micro variation: the one-way graft and the neighbour mutation. |
| `operators/macro_genome.rs` | Macro variation: uniform crossover, per-bit mutation and the all-zero repair. |
| `operators/local_search.rs` | Louvain first-phase modularity ascent, in place. |
| `similarity/mod.rs` | Similarity submodule wiring. |
| `similarity/kernel.rs` | The diffusion kernel `exp(β(A − D))` and the cyclic Jacobi eigensolver it needs. |
| `similarity/codec.rs` | `decode` (Eqs. 3–5) and `encode` (Eq. 8). |
