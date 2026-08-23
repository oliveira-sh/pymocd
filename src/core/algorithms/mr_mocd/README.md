# MR-MOCD

**Multi-Resolution Multi-Objective Community Detection** — a memetic particle swarm
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
different problem solved by different code. `mr_mocd_select` exposes the selector alone so
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
| selection | `front/select.rs` | codelength, then plateau, then modularity |

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
say which `gamma` is the graph's. The selector is a three-stage chain, each stage handing
over only when it has nothing to say.

Members with one community, or with at least half of `n` communities, are filtered out
first as degenerate. If that leaves nothing, the filter is dropped rather than the answer.

### 1. Shortest two-level code (`front/codelength.rs`)

The map equation: how many bits per step it costs to describe a random walk under a
partition. Implemented as `L = q H(Q) + sum_c p_c H(P^c)` in the algebraically equivalent
form needing only each community's exit rate and volume. Counts accumulate as integers and
are divided once, so the value depends on neither node order nor summation order.

This carries the resolution the front cannot: it answers from the *flow* rather than from
the hull's geometry, and it has no parameter to set.

**Measured over the five SNAP `com-` networks** — held out from everything that shaped the
plateau rule — it scores **0.690 mean AMI against a front ceiling of 0.705**, where the
plateau alone scores 0.636. That is **78% of the reachable gap**, and **+0.17 on Orkut**
alone.

It **abstains** when no member compresses the walk better than one module. That is
Infomap's significance test: past the detectability limit there is no partition worth the
index codebook. It abstains on **8 of 42 LFR archives**, which is where the plateau still
earns its place.

### 2. Widest resolution plateau (`front/plateau.rs`)

Take the lower-left convex hull of the candidates in `(pair, cut)`. Only hull vertices
optimise `cut + lambda * pair` for some `lambda > 0`. Each interior vertex is the CPM
optimum for `lambda` between the slopes of its two incident hull edges, and the width of
that interval in `log lambda` is how long that partition survives as the resolution
sweeps. The widest span wins.

- The two degenerate partitions join the hull as **anchors** so the coarsest and finest
  real members still get an interval instead of landing on an endpoint. Anchors bound the
  profile but are never chosen, and are exempt from the dominance staircase — otherwise a
  disconnected graph would delete the one-community anchor with its own `cut = 0`
  partition into components.
- Widths are **pooled by octave** of the community count, because a dense front subdivides
  one true plateau across several nearly collinear vertices and no single one keeps the
  full width. A partition with twice as many communities is a different answer; one with a
  few more is the same answer resampled.
- A vertex whose interval runs past either end of the ladder's range is **dropped rather
  than truncated**: a width the range decided is not evidence about the graph.
- Fewer than 5 hull vertices (two of which are always anchors) is too thin to be evidence,
  and the stage declines.

### 3. Maximum modularity (`front/modularity.rs`)

The fallback for the one case a plateau cannot speak to — a hull too thin to have an
interior, which happens only on graphs of a few dozen vertices. Scored as
`Q = (1 - cut) - sum_c (d_c / 2m)^2`; the first term is already in the archive, so scoring
a member costs one pass over the labels and none over the edges.

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
- **`MR-MOCD_RANDPERT`** — an environment-gated research control that held the perturbation
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
  `merge_sweep`'s ranking, in the rung prune, in `shortest_code`'s parallel reduction, in
  `max_modularity`, and in the hull sort. No outcome follows scan order, neighbour layout,
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
- **`modularity` reuses `cut` from the archive**, so scoring a member costs one pass over
  the labels and none over the edges. `deg` is marked with `UNSEEN = -1.0` rather than
  zero, because an isolated vertex legitimately has degree mass zero.

## 10. File map

| File | Contents |
|---|---|
| `api.rs` | `mr_mocd`, `mr_mocd_fronts`, `mr_mocd_select` — the public entry points |
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
| `front/codelength.rs` | the map equation and the abstention test |
| `front/plateau.rs` | the hull and the widest-plateau rule |
| `front/modularity.rs` | Newman modularity, the fallback |
| `front/select.rs` | the three-stage selection chain |
| `utils/sampling.rs` | the per-slot RNG, i.e. the determinism contract |
| `utils/output.rs` | internal labels to dense output ids, isolated nodes as -1 |
| `utils/fixtures.rs` | graph builders shared across test modules |
