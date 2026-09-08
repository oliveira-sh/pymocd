# RIMPSO

**RIMPSO-CD — Resolution-Indexed Memetic Particle Swarm Optimisation for Community Detection** — a memetic particle swarm
over the Constant Potts Model, with decomposition-based archive truncation and
parameter-free solution selection.

The source carries no comments. Everything that was written in the code — what each
piece does, why each constant is what it is, and what was measured to decide it — lives
here.

---

## 1. What the algorithm returns

A swarm of particles, each pinned to its own resolution on a geometric ladder, flies over
partitions of the graph. What it produces is not one partition but an **archive**: a
Pareto front over two objectives that together encode CPM at *every* resolution. That
front is the graph's resolution profile. A separate, label-free **selector** then picks
one member out of it.

The split matters: the archive is the deliverable, and choosing one member of it is a
different problem solved by different code. `rimpso_select` exposes the selector alone so
it can be run over a candidate set this crate did not produce — for example a front
pooled from an external CPM solver over the same ladder. That is the control that
separates the search's contribution from the selector's.

## 2. The objective

CPM at resolution `gamma` is `sum_c (e_c - gamma * C(n_c, 2))`. The module never scores
that directly. It decomposes it into two terms that do not mention `gamma`:

| Term | Definition | Meaning |
|---|---|---|
| `cut` | `1 - internal / m` | fraction of edges leaving their community — the partition's own realised mixing parameter |
| `pair` | `2 * pair_sum / (n(n-1))` | fraction of node pairs sharing a community |

where `internal` counts intra-community edges once and `pair_sum` is `sum_c C(n_c, 2)`.

Every `gamma` of CPM is a weighted sum of these two, so **the Pareto front over
`(cut, pair)` is the whole resolution profile of the graph** — one front, computed once,
answers every resolution at once. `objectives/cpm.rs::obj_of` is the conversion.

Both raw counts are integers (`Counts = (i64, i64)`), maintained incrementally through
every node move rather than recomputed. Being integers they cannot drift from a full
rescan, which is what makes a particle's objective free to read at any point.

## 3. Pipeline

```
ladder  ->  seed  ->  [ advance -> archive.offer -> archive.prune ] x gens  ->  select
```

| Stage | File | What happens |
|---|---|---|
| ladder | `swarm/ladder.rs` | one resolution per particle, geometric over `[1/n^2, 1]` |
| seed | `swarm/init.rs` | one particle per rung, each at a raw scatter |
| flight | `swarm/motion.rs` | velocity update, perturbation, repair, merge |
| archive | `pareto/archive.rs` | bounded non-dominated set, pruned by rung |
| selection | `front/assortative.rs` | the best-fitting assortative block model |

### The ladder

`[1/n^2, 1]` is the whole range where `gamma` can still change the answer: a CPM
community must be internally denser than `gamma` to survive, so above 1 everything
shatters and below `1/n^2` nothing splits. Rungs are spaced geometrically because
resolution is a multiplicative quantity.

On a graph with `n < 2` or no edges the ladder degenerates to all-ones — there is no
density for `gamma` to be measured against.

### Seeding

Every node takes a random neighbour's label. The start is *locally coherent* — so the
local move has structure to sharpen rather than to invent — but carries no optimisation.

**There is deliberately no seeding local search, and no `seed_rounds`.** Driving every
particle to a CPM local optimum before the first iteration was measured to be unnecessary
once the flight repairs: from a bare scatter the swarm recovers the same archive within
about ten iterations, and *overtakes* the pre-optimised start beyond that, because a
particle placed at a local optimum has to be dragged out of it before it can move. The
warm start is worth a handful of iterations and costs the swarm its claim to doing the
optimisation, so it is not shipped.

### The flight

Per node, velocity is the **probability that the node is unstable**:

```
v_j = clamp(inertia * v_j + r1 * [best_j != here] + r2 * [leader_j != here], 0, 1)
```

with `r1 = U(0,1) * cognitive` and `r2 = U(0,1) * social`, drawn once per particle per
iteration. Then one uniform `u` decides both branches, so they are mutually exclusive:

- `u < v` — the node is unstable. It adopts the personal best's label or the leader's, in
  proportion to the two pulls. When neither attractor pulls (both already agree with where
  the node is), it drifts to a random neighbour's label instead.
- `u >= v` and `u > 1 - local_rate` — the node is stable and gets one
  resolution-directed local move. This branch runs only on iterations where the full local
  search does not, so both dials are live in the shipped configuration.

Splitting the interval `[0, v)` by `t = u / v` gives a uniform on `[0,1)` conditioned on
instability, so choosing between the two attractors costs no further draw.

Isolated vertices are skipped entirely: an isolated vertex belongs to no community and no
local move could undo it joining one.

The flight always ends with `canonicalize`, because both attractors are read as raw labels
next iteration and they have to mean the same thing.

### Repair, and why it is the difference between searching and drifting

On local-search iterations the flight runs `repair` then `merge_sweep`.

`repair` drives the particle back to a local optimum of CPM at its own resolution,
starting from a queue of exactly the vertices the perturbation disturbed. A vertex that
moves can only have changed the best move of itself and its neighbours, so the work is
proportional to what the perturbation actually touched rather than to `n` — which is what
makes a full repair affordable at all. The queue is compacted once it has consumed 4096
entries and half its length, so a long repair cannot grow it without bound.

**Measured:** without the repair the perturbation is never undone. Seeding leaves each
particle at a local optimum, the flight scatters roughly a third of its vertices toward
the attractors, and the single probabilistic local move per vertex is far too weak to
climb back. Over 100 generations of 100 particles the personal best then improves
**between 0 and 9 times in total**, and the net effect on the LFR grid is negative. With
the repair, **43 to 88 particles out of 100** end above their own seed.

### Merge, and why refinement alone is not enough

`repair` sharpens; `merge_sweep` coarsens. Neither reaches the other's granularity. No run
of single-node moves can merge two communities that are each individually stable, so
without the merge the flight can only ever refine.

**Measured on dblp, amazon and youtube:** removing the merge inflated the community count
**2.6 to 4.5 times** and dropped AMI by **0.04 to 0.05**.

One sweep accepts a *matching* — each community is spoken for at most once — so the
accepted gains cannot interact and every accepted merge is valid at the moment it is
applied. A run of sweeps still reaches full agglomeration.

## 4. The archive

A bounded set of mutually non-dominated partitions, one per point of the resolution
profile. It is also the swarm's shared memory: leaders are drawn from it, and it is
read-only for the whole of each flight so every particle sees the same front.

**`offer`** rejects a candidate that an incumbent dominates or whose point is already
occupied; otherwise it evicts everything the candidate dominates and joins. The position
is cloned on acceptance only.

**`prune` keeps the argmin of `cut + w * pair` at every rung**, where `weights[k] =
gamma_k / gamma_d` and `gamma_d = m / C(n,2)` is the resolution at which CPM sees the
graph as its own density. Dividing the ladder through by `gamma_d` turns each rung into
exactly the weight that makes `cut + w * pair` the CPM objective at that rung — so keeping
the argmin at every rung keeps the best partition the archive holds at every resolution
the swarm actually searches.

This replaced pruning by crowding distance. **Crowding is a diversity criterion with no
notion of quality**, and it was measured evicting the archive's best member while that
member was not dominated by anything.

The archive is **deliberately not filled back up to the cap** with the least crowded of
the remainder. Doing so enriches the front past what the selector can exploit, which cost
**0.44 AMI on LFR n=1000 mu=0.5 at an unchanged oracle**.

Crowding distance is still computed after each prune, but only to feed leader selection:
`leader` is a binary tournament on crowding, so leaders come from the sparse end of the
front.

## 5. Selection

Every member of the archive is CPM-optimal at its own `gamma`, so the profile alone cannot
say which `gamma` is the graph's. **One criterion answers it: the member that a
degree-corrected assortative block model fits best, once its own free densities are paid
for.**

The model gives every community its own internal edge density and lets everything between
communities share one:

```
A_ij ~ Poisson(k_i k_j omega_c)     both endpoints in community c
A_ij ~ Poisson(k_i k_j omega_out)   otherwise
```

Profiling out the densities leaves a log-likelihood ratio against the configuration model
that is two bincounts and nothing else:

```
L(C)  = sum_c e_c ln(e_c / E_c)  +  e_out ln(e_out / E_out)
E_c   = d_c^2 / 4m      E_out = m - sum_c E_c      e_out = m - sum_c e_c
```

`E_c` is how many intra-community edges the degree sequence alone would produce, so `L` is
how many nats of edge placement the *partition* explains. Each community brings a free
density, so the densities are charged the Schwarz penalty — half a nat per parameter per
log observation, over the `B + 1` densities and the graph's `2m` edge endpoints, where `B`
counts the communities holding at least one non-isolated vertex:

```
score(C) = L(C) - (B + 1)/2 * ln(2m)          maximised
```

A partition whose communities are internally *sparser* than the space between them fits the
same model with the two densities swapped and is not a community structure; it is rejected
outright. That guard changes no choice on any of the 362 archives measured — no such
partition can sit on a CPM front — but `rimpso_select` takes candidates from anywhere.

`L` does **not** rise monotonically with `B`. That holds for the unrestricted block model,
where a refinement's model nests the coarser one; it fails here, because splitting a
community moves the pairs it sheds off its own free density and onto the single shared
`omega_out`. On `ring_of_cliques(12, 5)` the planted partition scores `L = 259.02` and a
strict refinement of it `L = 53.52`, a gap of 205 nats before any penalty is charged.

Both degenerate partitions lose, neither by a special case: one community has `e_1 = m` and
`E_1 = m`, so `L = 0` and the score is `-ln(2m)`; all singletons have `e_c = 0` everywhere,
so `edges_in = 0` while `expect_in > 0` and the assortativity guard rejects the partition
outright. **There is no degeneracy filter, no fallback stage and nothing to abstain with**
— one criterion, evaluated once per member, always returns an answer.

### What it is worth

Measured over **46 benchmark cells** — LFR at `n` in {1000, 10000, 50000, 100000} crossed
with `mu` from 0.1 to 0.8 (20 seeds at n <= 10000, 3 at 50000, 1 at 100000), five annotated
social graphs of 34 to 1005 vertices, and the five SNAP `com-` graphs up to 4 million
vertices — as the mean gap to the **front oracle**, the archive member with the highest AMI
against ground truth:

| | this criterion | the chain it replaced |
|---|---|---|
| mean gap to the oracle, over the 46 cells | **0.021** | 0.044 |
| worst single cell | **0.368** (karate) | 0.368 (LFR n=1000 mu=0.5) |
| LFR, mu <= 0.4 | **0.000** | 0.000 |
| LFR, mu >= 0.5 | **0.008** | 0.065 |
| the five small annotated graphs | **0.143** | 0.154 |
| the five SNAP graphs | **0.006** | 0.011 |

It is never worse than the chain on any real network, and the whole gain is at `mu >= 0.5`,
where the chain's first stage abstains: on LFR `n = 1000, mu = 0.5` it scores **0.666**
against the chain's 0.330, and on `n = 10000, mu = 0.7`, **0.278** against 0.152.

### What it replaced

A three-stage chain: shortest two-level map-equation code length, then the widest
resolution plateau on the front's lower convex hull, then maximum modularity. Stage 1
abstained on 12 of 24 LFR cells at `n = 1000`, all at `mu >= 0.5`, and the plateau that
caught those is the weakest of the three (0.119 from the oracle on its own).

### What was measured and rejected

Forty-five criteria were scored over the same 362 archives. Mean gap to the oracle over the
46 cells, against **0.021** for the shipped rule:

| Criterion | Gap | Why it loses |
|---|---|---|
| Surprise per bit of partition, `S / L_partition` | 0.036 | scale-free and the best of all on graphs under 1000 vertices, but the denominator collapses for a near-degenerate partition: on orkut it takes one that is 99.5% one community and scores 0.14 against an oracle of 0.69 |
| Bernoulli planted-partition description length, `S - L_partition` | 0.040 | textbook MDL, and it abstains where MDL should: at LFR `n = 10000, mu = 0.7` *every* member has `S < L` — the AMI-0.294 member costs 66542 nats to state and buys 53163 — so it returns one community |
| Synwalk | 0.032 | the best random-walk criterion, but it over-splits every graph under a few hundred vertices |
| degree-corrected Surprise, no penalty | 0.032 | same |
| Reichardt-Bornholdt self-consistency fixed point | 0.032 | same |
| two-level map equation | 0.090 | collapses to one module at `mu >= 0.6` |
| flat degree-corrected SBM description length | 0.162 | the `B(B+1)/2` prior on the block edge counts caps `B` at about `sqrt(n)`; it picks 8 of 30 cliques on `ring_of_cliques(30,5)` |
| maximum modularity | 0.118 | the resolution limit, whole |
| the widest resolution plateau | 0.119 | a dense front subdivides one plateau; pooling by octave only half-fixes it |
| Erdos-Renyi modularity, `(1 - cut) - pair` | 0.165 | **every parameter-free knee rule on this front reduces to exactly this** — Das's maximum bulge, Kneedle, KnEA, the max-min distance to a control front, HP-MOCD's own selector — and it is a linear scalarisation, so it can only ever return a hull vertex |
| leave-one-pair-out predictive log-loss | 0.045 | the cross-validation family; over-splits on small graphs by a bounded, computable amount |
| Bethe-Hessian block count, then the member with the nearest `B` | 0.143 | recovers the exact community count at `mu <= 0.3` and collapses at `mu >= 0.5` |
| max-min distance to a G(n,m) control front | 0.228 | the control front is nearly a point; the rule degenerates to "furthest from the origin" |
| Bayesian (Dirichlet-regularised) map equation | 0.204 | its regularisation pushes toward *fewer* modules, which is the wrong direction here |
| Significance, exact Surprise, Z-modularity, Li-Pan structural entropy, the CPM self-consistency fixed point, hierarchical clustering entropy, the Rosvall-Bergstrom 1997 two-part code, the dense SBM, both degree-corrected planted-partition description lengths | 0.054 - 0.632 | measured, all worse |

### The one thing no criterion here can do

At `mu >= 0.7` the archive still holds members that score AMI 0.28-0.35, and **not one of
them compresses the graph**: at `n = 50000, mu = 0.7` the oracle member's Surprise is 381620
nats against a partition description length of 428732. karate is the same story at the
other end — its oracle member (four communities, AMI 0.850) has `S - L = -6`, so no
description-length criterion in this family can ever return it, and 0.670 is the best any
of the forty-five reached. Where the answer is not compressive, a criterion that is
correct by its own lights is wrong by AMI. That is the whole of the residual gap.

## 6. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `pop_size` | 100 | swarm size, and the number of ladder rungs |
| `num_gens` | 100 | iterations |
| `inertia` | 0.4 | fraction of a node's instability surviving to the next iteration |
| `cognitive` | 0.7 | pull toward the particle's own best partition |
| `social` | 0.7 | pull toward the archive leader |
| `local_rate` | 0.35 | per-node rate of the resolution-directed local move |
| `archive` | 100 | archive capacity — one slot per particle, so it holds the whole profile |
| `ls_period` | 10 | run the full local search once every this many iterations; 0 disables |

Out-of-range values are clamped rather than rejected, because they arrive from Python.
`pop` and `archive` are floored at 2; the four rates are clamped to `[0, 1]` with NaN
mapped to 0.

### Why `num_gens = 100`

Chosen under a curve that is **not monotone**. Mean AMI over twelve benchmarks rises to
`T = 50`, is flat to `T = 200`, and at `T = 400` falls *below* its value at `T = 25`, with
the collapse concentrated on the hardest cell. 100 wins eight of the twelve cells and
every large network, while 50 takes the mean on the strength of two graphs of under 110
vertices. **It is a compromise, not a maximum.**

### Why `ls_period = 10`

The local search is the swarm's servant, not its engine. Running it on every iteration
makes the flight a wrapper around a Louvain sweep; replicated over ten graphs of the cell
that decides it, that costs **0.135 AMI**. Running it every `ls_period` iterations leaves
the particles to move under the attractors in between.

What actually distinguishes 10 is **variance**: it has both the highest mean and the
lowest per-graph spread of the settings swept, and is the only one never catastrophic on
any graph. Both extremes are bad and the useful setting is interior — but it is
instance-dependent, and on other cells smaller periods win by up to 0.019.

### Removed parameters

- **`ls_start`** — iterations to fly before the local search was allowed to run. Shipped
  as 0, which made its guard `step > 0 && (step - 0) % ls_period == 0`, i.e. exactly the
  unconditional schedule. It never did anything and has been removed from the Rust API,
  the Python signature and the docs.
- **`MR-MOCD_RANDPERT`** (spelled with the method's former name) — an environment-gated research control that held the perturbation
  rate but sent every unstable vertex to a random neighbour's label instead of an
  attractor's, reducing the flight to a randomised iterated local search. It was the
  control for whether the swarm is load-bearing. Never set in any shipped path; removed.

## 7. Invariants

These are enforced by `debug_assert!` and hold by construction in release.

| Invariant | Where | Upheld by |
|---|---|---|
| labels are slots in `[0, n)` | everywhere | `canonicalize`, `load_sizes` |
| `pop >= 2` | `ladder` | `Cfg::new` floors it |
| `weights` is never empty | `Archive::with_rungs` | one weight per particle |
| `relocate` is never handed the vertex's own community | `Particle::relocate` | both callers compare first; without this the pair count would silently gain 1 |
| `best_move` never sees an isolated vertex | `swarm/local.rs` | `advance` and `repair` both skip empty adjacency |
| no two archive members share a coordinate | `crowding` | one would dominate the other, so `offer` would have rejected it |

## 8. Determinism

The result does not depend on how rayon schedules anything.

- **One independent RNG stream per (iteration, particle)**, seeded from
  `RNG_BASE ^ salt.rotate_left(32) ^ slot * PHI`. No stream is shared. Seeding uses
  `salt = u64::MAX`, reserved so no iteration can collide with it.
- **Every tie is broken by the lower index or the lower label** — in `best_move`, in
  `merge_sweep`'s ranking, in the rung prune, and in `select_index`'s parallel reduction.
  No outcome follows scan order, neighbour layout,
  or how a parallel reduction happened to split.
- The archive is offered candidates **in slot order** after each flight, never in
  completion order.
- Particles are canonicalised to the shared naming at the end of each flight, so a label
  copied from an attractor means the same thing to both particles.

## 9. Performance notes

- **Incremental counts.** `internal` and `pair_sum` are updated by `relocate` on every node
  move. Integer arithmetic, so no drift; a full rescan is needed only at seeding and after
  a merge sweep.
- **One `Scratch` per worker, not per particle.** Rayon hands each particle out as its own
  task, so a scratch built per task is a scratch built per particle per iteration.
  `ScratchPool` builds the `n`-sized buffers once for the whole run and hands each worker
  its own slot; the mutex carries the borrow rather than serialising anything. Each slot is
  `#[repr(align(128))]` — the buffer headers live inside the slot and a push in the
  innermost loop writes one, so packing two workers' slots together would have them
  fighting over a cache line.
- **Nothing leaks between particles.** `load` rebuilds the sizes, `link` is restored to
  zero by every path that touches it, and the rest is rebuilt or cleared before use.
- **`load_sizes` clears adaptively.** Clearing a scattered entry costs a whole cache line,
  so once the live set reaches a sixteenth of the slots it is cheaper to `fill(0)` the
  whole array than to visit only the live ones.
- **`best_move` takes and clears each count in one visit** — the difference between one
  scattered write per neighbouring community and two. Staying put is scored with the vertex
  already removed from its community, so every comparison is like for like.
- **`merge_sweep` counts one community at a time**, so edges into each neighbouring
  community land in the same flat array a node move uses and the pair never has to be
  hashed. Only the upper side of each pair is counted, which offers each pair exactly once
  and leaves half the edges costing nothing but the label they are read through. The
  candidate merges are packed into a `u128` as `(!gain_bits, a, b)` — `gain` is strictly
  positive there, so complementing its IEEE bits reverses the order and the whole
  comparison becomes a plain integer sort.
- **`bucket_by_community` is a counting sort** off the sizes that are already maintained:
  one pass, no comparison. Filling from the back turns each running end into the start it
  will be read as.
- **Selection costs one pass over the labels and one over the edges** per member, and the
  members are scored in parallel with one pair of `n`-sized buffers per worker, cleared
  through `live` rather than refilled. Nothing in it is larger than `m ln m`, so unlike a
  description length written with binomials it needs no special numerics at four million
  vertices.

## 10. File map

| File | Contents |
|---|---|
| `api.rs` | `rimpso`, `rimpso_fronts`, `rimpso_select` — the public entry points |
| `config/mod.rs` | `Cfg`, the parameter bundle, with Python-facing clamping |
| `config/defaults.rs` | the shipped constants |
| `objectives/cpm.rs` | the CPM split, size bookkeeping, community counting |
| `pareto/dominance.rs` | Pareto dominance over the two objectives |
| `pareto/crowding.rs` | Deb's crowding distance, the leader-selection density estimate |
| `pareto/archive.rs` | the bounded external archive and the rung prune |
| `swarm/ladder.rs` | the resolution ladder |
| `swarm/init.rs` | seeding — scatter, no local search |
| `swarm/particle.rs` | `Particle`, `Scratch`, `ScratchPool`, incremental counts |
| `swarm/local.rs` | the resolution-directed single-node move |
| `swarm/merge.rs` | the community-merge sweep |
| `swarm/motion.rs` | the velocity update, perturbation and repair |
| `swarm/engine.rs` | the swarm loop |
| `front/assortative.rs` | the degree-corrected assortative block model, its Schwarz penalty, and the two entry points |
| `utils/sampling.rs` | the per-slot RNG, i.e. the determinism contract |
| `utils/output.rs` | internal labels to dense output ids, isolated nodes as -1 |
| `utils/fixtures.rs` | graph builders shared across test modules |
