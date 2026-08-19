# SMOCC — Sparse Multi-Objective Co-evolutionary Community detection

**Paper**

Santos, G. — SMOCC, in preparation (2026). This project's own method.

Derived from the macro–micro co-evolutionary detector: Zhang, Yang, Yang &
Zhang, *IEEE Computational Intelligence Magazine*, 2023.
<https://ieeexplore.ieee.org/document/10188453> (implemented separately in this
repository as [`mmcomo/`](../mmcomo/README.md)). SMOCC keeps that paper's
two-population structure and replaces its dense diffusion kernel; see
**Divergences**.

Engine: Deb, Pratap, Agarwal & Meyarivan, "A fast and elitist multiobjective
genetic algorithm: NSGA-II", *IEEE TEC* 6(2):182–197, 2002.
DOI 10.1109/4235.996017. The front assignment uses the two-objective sweep of
Jensen, "Reducing the run-time complexity of multiobjective EAs: the NSGA-II and
other algorithms", *IEEE TEC* 7(5):503–515, 2003. DOI 10.1109/TEVC.2003.817234.

Objective set 6 is HP-MOCD's decomposition (Santos et al., *Social Network
Analysis and Mining* 15, 2025), whose lineage is Shi, Yan, Cai & Wu, "A
multi-objective approach for community detection in complex network", *Applied
Soft Computing* 12(2):850–859, 2012. Objective set 20 is the Constant Potts
Model of Traag, Van Dooren & Nesterov, "Narrow scope for resolution-limit-free
community detection", *Phys. Rev. E* 84:016114, 2011.
DOI 10.1103/PhysRevE.84.016114, split exactly as in [`mopots/`](../mopots/README.md).

**Original implementation**

This project's own; there is no external reference implementation. The
macro–micro paper it descends from has none published either.

**Objectives**

Notation, over `G = (V, E)` with `n = |V|`, `m = |E|` (`g.edges` holds every
undirected edge exactly once) and a partition `C = {V_1 … V_k}` where `k` is the
number of *distinct labels present*:

    deg(v)            degree of v
    d_c   = Σ_{v∈V_c} deg(v)                    degree mass of community c
    e_c   = |{(u,v) ∈ E : u,v ∈ V_c}|           internal edges of c, counted once
    L(V_c,V_c) = 2·e_c                          internal degree mass
    L(V_c,V̄_c) = d_c − L(V_c,V_c)               cut mass of c

`obj_mode` picks one pair per swarm. All six values are **minimised**.

*ObjSet 0 — `(KKM, RC)`, kernel k-means and ratio cut.* Scope: all `n` nodes,
`|V_c|` counting isolated members.

    KKM = 2(n − k) − Σ_c L(V_c,V_c) / |V_c|
    RC  =            Σ_c L(V_c,V̄_c) / |V_c|

`KKM` falls as communities become internally dense per member; `RC` falls as
they leak less per member. Splitting raises `k`, which lowers `KKM`'s leading
term — that is the tension the front trades against `RC`.

*ObjSet 6 — `(intra, inter)`, the HP-MOCD modularity decomposition.* Scope: all
`n` nodes. Returns `(0, 0)` when `m = 0`.

    intra = 1 − (Σ_c e_c) / m
    inter = Σ_c (d_c / 2m)²

Newman modularity is recovered exactly as `Q = 1 − intra − inter`, so minimising
both jointly is maximising `Q` along a front rather than at one resolution.

*ObjSet 20 — `(cut, pair)`, the Constant Potts pair.* Scope: **the `n_a` nodes
of non-zero degree only**; `n_c` likewise counts only non-isolated members of
`c`. This is the one set whose scope is not all of `V`.

    cut  = 1 − (Σ_c e_c) / m                 fraction of edges leaving their community
    pair = Σ_c C(n_c,2) / C(n_a,2)           fraction of active node pairs co-clustered

with `C(k,2) = k(k−1)/2`, and `cut = 0` when `m = 0`, `pair = 0` when `n_a < 2`.
The pair is an exact affine decomposition of the CPM Hamiltonian:

    H_γ(C)/m = 1 − cut − (γ/γ_d)·pair,       γ_d = 2m / (n_a(n_a−1))

so `γ` never enters the search: it is the exchange rate between the two axes and
one Pareto front is the whole resolution sweep.

`obj_mode` may give the two swarms *different* pairs (see **Parameters**). When
it does, the micro pair is authoritative: at the end of the run the macro
members are re-evaluated under the micro objectives before the two halves are
ranked together, and `refine_front` also sorts under the micro set.

**Representation**

Two co-evolving populations over one CSR graph, plus a sparse edge similarity
they exchange through.

- **Micro — label-based, no decode.** `Labels = Vec<i32>`, one community label
  per dense node index. Initialisation gives each node the *index of a uniformly
  drawn neighbour* (an isolated node takes its own index), so the population
  starts already respecting adjacency rather than at random labels.
- **Macro — centre-indicator genome with a decode.** `Genome = Vec<u8>`, one
  bit per node marking it a community centre. `decode` seeds each centre with
  its own slot label and runs asynchronous weighted label propagation over
  `wadj`: a node adopts the label with the greatest incident weight, centres are
  pinned and never move, and only nodes adjacent to a node that just changed are
  re-queued. Slots are then rewritten to the centre's node id. Nodes no
  component of which contains a centre are left unlabelled and get their
  component's minimum node id by a second min-propagation, so a centreless
  component becomes exactly one community rather than singletons.
  `encode` inverts it: per community, the member of greatest internal weighted
  degree becomes the centre (lowest node index wins a tie).
  Initialisation draws `c ∈ [1, ceil(macro_cap·√n)]` centres; the first half of
  the population shuffles the `min(3c, n)` highest-degree nodes and takes `c`,
  the second half draws `c` uniformly at random. An all-zero genome is repaired
  to the highest-degree node.
- **`wadj: Vec<f64>`** — one weight per *directed* CSR adjacency slot, so it is
  indexed in lockstep with `g.adj` and each undirected edge appears twice.
  Starts at all-`1.0`. This is the sparse stand-in for the paper's dense
  similarity matrix.

**Algorithm**

1. `wadj ← 1`; build `pop` micro individuals and `pop` macro individuals;
   evaluate each under its swarm's objective pair.
2. For `t = 1 … num_gens`:
   a. Rank + crowd the micro population; produce `pop` children with RNG salt
      `2t`. Each child clones a binary-tournament winner and then, with
      probability `cross_rate`, is recombined: either the **graft** (copy every
      node carrying a random donor label from a second parent) or, under
      `topo_mode` bit 128, the **HP-MOCD ensemble consensus** of 4 distinct
      tournament winners (per node the majority label; ties sorted ascending,
      then drawn uniformly). Mutation then visits each node with probability
      `micro_mut` and moves it either to a uniformly drawn neighbour's current
      label or, under bit 2, to its `wadj`-heaviest neighbouring label. Both
      mutations are Gauss–Seidel: a node sees moves made earlier in the sweep.
   b. Rank + crowd the macro population; produce `pop` children with RNG salt
      `2t + 1` by uniform crossover of two tournament winners followed by
      per-bit flips at probability `mut_rate`. An all-zero child is repaired to
      one random centre. Decode and evaluate each.
   c. If `t mod gap ≠ 0`, each swarm independently truncates parents+children
      back to `pop` by NSGA-II environment selection.
      If `t mod gap = 0`, the two swarms **exchange** first:
      - *guidance* — every rank-1 macro member is decoded, re-scored under the
        micro objectives and thrown into the micro pool before truncation.
      - *influence* — the rank-1 micro members are the elites. `wadj` is
        relaxed toward their consensus, `w ← (1−ρ)w + ρ·c` with
        `ρ = 0.5·t/num_gens` and `c` the fraction of elites placing the edge's
        two ends together; then each elite is `encode`d, re-`decode`d and thrown
        into the macro pool before truncation.
3. Merge both final populations, re-scoring macro members under the micro
   objectives if the two pairs differ, and keep rank 1. An empty front falls
   back to the all-singletons partition.
4. If `refine` (always on for `smocc`): **union refinement**. For each member,
   offer the connected-component split of every disconnected community and the
   tiny-community reabsorption of both the member and that split; deduplicate;
   if anything new was produced, re-rank the union and keep rank 1.
   Reabsorption merges any community of size ≤ 2 into its `wadj`-heaviest
   neighbouring community whenever that pull exceeds the community's own
   internal weight (or the community is a singleton, or has no internal weight),
   for up to 5 passes.
5. Selection rule: `smocc` returns the front member of least cost under a
   **label-free scalarisation** — all four of `(KKM, RC, intra, inter)` are
   min-max normalised across the front and summed, lowest total wins, lowest
   index breaks a tie. A column that is constant across the front contributes
   `0`, not `NaN`. Note this is fixed and does **not** follow `obj_mode`.
   `smocc_fronts` returns the whole refined front instead.

**Parameters**

| Parameter | Default | Effect |
|---|---|---|
| `pop_size` | 100 | Size of **each** swarm, so `2·pop_size` individuals are carried. |
| `num_gens` | 100 | Generations. The search always runs all of them; there is no convergence stop. |
| `cross_rate` | 0.7 | **Micro** crossover probability only. Macro crossover is unconditional and uniform. |
| `mut_rate` | 0.5 | **Macro** per-bit flip probability only. |
| `micro_mut` | 0.5 | Micro per-node mutation probability. Any value `≤ 0` means `1/n`; values `> 1` clamp to `1`. |
| `gap` | 10 | Co-evolution interval: the swarms exchange every `gap` generations. Clamped up to `1`, so `0` means every generation. |
| `macro_cap` | 1.0 | Multiplier on the macro centre ceiling `ceil(macro_cap·√n)`, still hard-capped at `n`. `1.0` is the historical `ceil(√n)`. Raise it when the true community count exceeds `√n`. `NaN` falls back to `1.0`; anything else is clamped to `[1e-6, 1e6]`. |
| `refine` | `true` | Union refinement of the final front. Exposed on `smocc_fronts` only. |
| `topo_mode` | 130 | Micro operator bitmask. Exposed on `smocc_fronts` only. |
| `obj_mode` | 1020 | Objective placement. Exposed on `smocc_fronts` only. |

`smocc()` hardcodes `refine = true`, `topo_mode = DEFAULT_TOPO_MODE` and
`obj_mode = DEFAULT_OBJ_MODE`; the three are reachable only through
`smocc_fronts()`.

*`topo_mode`* — two live bits, freely combined; `130 = 128 | 2` is shipped and
`0` is the historical operator set.

| Bit | Name | Effect |
|---|---|---|
| `2` | `TOPO_MAJORITY_MUT` | Micro mutation moves a node to its `wadj`-heaviest neighbouring label instead of a uniformly drawn neighbour's. |
| `128` | `TOPO_HPMOCD_CROSS` | Micro crossover is the faithful HP-MOCD 4-parent ensemble consensus instead of the single-donor graft. |

Bits `1, 4, 8, 16, 32, 64` are **deleted and silently inert**. They used to
select a 3-parent ensemble crossover, a k-aware macro mutation, a
community-split mutation, a multi-community graft, the `wadj`-weighted local
search and a 2-hop-exclusion macro centre init. None beat the shipped mask, so
the code is gone; the bits are deliberately not reused so that old benchmark
rows naming them cannot be mistaken for a new operator. Two tests pin this
(`config/topo.rs` at the decoder, `api.rs` end to end).

*`obj_mode`* — three sets at their original ids: `0 = (KKM, RC)`,
`6 = (intra, inter)`, `20 = (cut, pair)`. Any other id decodes to `0`. Three
encodings, chosen by magnitude:

| Range | Meaning |
|---|---|
| `v < 100` | Homogeneous: both swarms use set `v`. |
| `100 ≤ v < 1000` | Heterogeneous, one decimal digit per side: `micro = (v−100)/10`, `macro = (v−100)%10`. So `160` is micro `(intra, inter)` / macro `(KKM, RC)`. This branch cannot name a two-digit id such as `20`. |
| `v ≥ 1000` | Heterogeneous, two digits per side: `micro = (v−1000)/100`, `macro = (v−1000)%100`. |

Measured placements: `1020` (micro `(KKM, RC)` / macro CPM) is shipped — at
matched front size it beat every other placement on LFR at mixing `μ ≥ 0.5`
(+0.029 AMI over `160`) at a cost of −0.026 on the annotated real networks, and
it is the only placement that survived the matched-front-size control. `3000`,
its mirror, was the worst tried. `3020` is homogeneous CPM, i.e. the same arm as
`20`.

Internal constants: `MAX_TINY = 2` and 5 reabsorption passes (`front/tiny.rs`),
`ENSEMBLE = 4` distinct parents with at most 64 tournament tries
(`operators/micro_labels.rs`), `MACRO_CAP_MIN/MAX = 1e-6 / 1e6`
(`macro_micro/init.rs`), `RNG_BASE` (`utils/sampling.rs`), and the `n`-sweep
ceiling on label propagation (`similarity/codec.rs`).

**Determinism**

**Bit-deterministic** (measured: two consecutive calls produce identical output,
and the module's digests are unchanged across refactors). What enforces it:

- Every draw comes from `slot_rng(salt, slot)`, one independent `StdRng` stream
  per `(salt, slot)` pair, so no draw depends on the thread count or on rayon's
  scheduling. The salts are disjoint by construction: `u64::MAX` for the initial
  micro population, `u64::MAX − 1` for the initial macro population, `2t` for
  generation `t`'s micro children and `2t + 1` for its macro children.
- Rayon appears only as `into_par_iter().map(…).collect()` over slot indices,
  which preserves order, and as an elementwise `zip` in `update_weights`. No
  reduction crosses threads, so no float sum is reassociated by the scheduler.
- Float accumulation order is pinned by data structure. `intra_inter` and `cpm`
  accumulate over a `Vec` in first-seen-label order; `cpm` in particular counts
  in integers and divides once at the end. `kkm_rc` is the exception: it sums
  over an `FxHashMap`'s iteration order, which is deterministic because `FxHash`
  is unseeded — the test `order_map_iterates_like_the_f64_map_it_replaced`
  exists to hold that iteration order fixed across a change of value type, and
  is load-bearing rather than a curiosity.
- Two more places read a hash map. `refine_tiny`'s merge target is chosen by
  `max_by` over an `FxHashMap`, so a tie on both weight *and* member count falls
  through to hash-layout order — reproducible, but layout-dependent. `encode`
  iterates an `FxHashMap` at the end, but only to set bits, so the order cannot
  matter.
- `init_macro` sorts nodes by descending degree with `sort_unstable_by`. That is
  unstable but not random, so equal-degree ties resolve identically on every
  run.
- Ties in the search are resolved by fixed rules, never by a fresh draw:
  binary tournament keeps the first-drawn index, front selection keeps the
  lowest index, and the ensemble crossover sorts tied labels ascending before
  drawing among them.

**Divergences**

Against the macro–micro paper this method descends from, and against the
`mmcomo/` reimplementation of it in this repository:

- **The paper's local-search step is deleted, not feature-flagged.** The
  Louvain-first-phase modularity ascent over the rank-1 micro members was
  removed outright; there is no parameter to enable it, and re-adding it is a
  change of algorithm that must be re-measured.
- **The dense `n × n` diffusion-kernel similarity is replaced by a sparse
  per-edge weight vector** over the CSR adjacency, reinforced from the elite
  consensus. Memory and time are `O(n + m)` instead of `O(n²)`, and there is
  consequently no `beta` (kernel exponent) parameter.
- **The two swarms may optimise different objective pairs** (`obj_mode`
  heterogeneous encodings). The paper runs one pair on both. The shipped default
  is heterogeneous.
- **A third objective set, the CPM `(cut, pair)` pair, is available and shipped
  on the macro side.** It is borrowed from `mopots/`, not from the paper, and it
  is the only set scoped to non-isolated nodes rather than to all of `V`, so
  `(KKM, RC)` and `(cut, pair)` do not measure the same node set.
- **Selection is a label-free normalised scalarisation over all four of
  `(KKM, RC, intra, inter)`**, not max-`Q`. It ignores `obj_mode` entirely, so
  changing the objective placement changes what the search *finds* but not how
  the single answer is *picked* out of the front.
- **Union refinement of the final front is an addition**: connected-component
  splitting of disconnected communities and reabsorption of communities of size
  ≤ 2. It can only add members before re-ranking, so it never loses a member
  that stays non-dominated.
- **The macro centre ceiling is parametrised** as `ceil(macro_cap·√n)` rather
  than fixed at `ceil(√n)`; `macro_cap = 1.0` reproduces the original exactly.
- **Six of the eight `topo_mode` operator bits were removed** rather than kept
  behind a flag, and are deliberately left inert instead of being reused.
- **The default micro operators are HP-MOCD's, not the paper's**: 4-parent
  ensemble consensus crossover and `wadj`-weighted neighbour-majority mutation.
  The paper's single-donor graft and uniform neighbour copy remain reachable as
  `topo_mode = 0`.
- **`cross_rate` and `mut_rate` are not symmetric knobs.** `cross_rate` gates
  only micro crossover and `mut_rate` only macro bit-flips; the micro mutation
  rate is the separate `micro_mut`, and macro crossover has no probability at
  all.

**Files**

| Path | Holds |
|---|---|
| `mod.rs` | Module root: the `Labels`/`Genome` type aliases, the submodule tree, and the `smocc`/`smocc_fronts`/defaults re-exports. |
| `api.rs` | The two entry points and the end-to-end test roster (topo bits, objective modes, `macro_cap`, determinism). |
| `config/mod.rs` | Config submodule wiring. |
| `config/defaults.rs` | The eight shipped default constants. |
| `config/modes.rs` | `Cfg`: resolves `obj_mode` into the micro and macro `ObjSet`s and evaluates against them. |
| `config/topo.rs` | `MicroOps`: the two live `topo_mode` bits, their mask, and the tests pinning the six dead bits. |
| `objectives/mod.rs` | Objective dispatch and the shared `UNSEEN` slot sentinel. |
| `objectives/sets.rs` | `ObjSet`, the `split_mode` three-branch `obj_mode` decoder, and `evaluate`. |
| `objectives/kkm_rc.rs` | `(KKM, RC)`, plus the test holding the `FxHashMap` summation order fixed. |
| `objectives/intra_inter.rs` | `(intra, inter)`, the HP-MOCD modularity decomposition. |
| `objectives/cpm.rs` | `(cut, pair)`, the CPM identity, and the cross-check that SMOCC and MO-POTS agree bit for bit. |
| `nsga2/mod.rs` | The `Obj` alias and the NSGA-II re-exports. |
| `nsga2/sorting.rs` | Non-dominated ranking by the two-objective Jensen sweep, checked against a generic `O(n²)` sort on random data. |
| `nsga2/crowding.rs` | Crowding distance, boundaries at infinity. |
| `nsga2/survival.rs` | Environment selection: fill front by front, break the last front by crowding. |
| `macro_micro/mod.rs` | Co-evolution submodule wiring. |
| `macro_micro/engine.rs` | `run_fronts`: the generational loop, the exchange schedule, the final merge and rank-1 extraction. |
| `macro_micro/init.rs` | Initial micro (neighbour-label) and macro (degree-biased centre) populations, and `macro_cmax`. |
| `macro_micro/exchange.rs` | `guidance` (macro elites → micro pool) and `influence` (micro elites → `wadj` and the macro pool). |
| `macro_micro/swarms.rs` | The `Mic`/`Mac` population members and survivor selection over pools of them. |
| `operators/mod.rs` | Operator re-exports. |
| `operators/macro_genome.rs` | Macro variation: uniform crossover plus bit-flip mutation over the centre genome. |
| `operators/micro_labels.rs` | Micro variation: graft or ensemble-consensus crossover, uniform or `wadj`-weighted neighbour mutation, and the mutation-rate rule. |
| `similarity/mod.rs` | Similarity submodule wiring. |
| `similarity/weights.rs` | `wadj` initialisation and the elite-consensus reinforcement that closes the co-evolutionary loop. |
| `similarity/codec.rs` | `decode` (seeded weighted label propagation with an active set) and `encode` (one centre per community). |
| `front/mod.rs` | Post-search submodule wiring. |
| `front/refine.rs` | `refine_front`: the union of split and reabsorbed variants, re-ranked. |
| `front/components.rs` | Connected-component splitting of disconnected communities. |
| `front/tiny.rs` | Tiny-community reabsorption into the heaviest neighbour. |
| `front/select.rs` | `select_best`: the label-free four-objective normalised scalarisation. |
| `utils/mod.rs` | Helper submodule wiring. |
| `utils/sampling.rs` | `slot_rng`, `bernoulli` and `tournament` — the determinism contract and the shared draws. |
| `utils/output.rs` | Remapping to output labels, with isolated nodes reported as `-1`. |
| `utils/fixtures.rs` | Graph builders shared by more than one test module. |
