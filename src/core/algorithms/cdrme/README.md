# CDRME — Community Detection based on Random walk and Multi-objective Evolutionary algorithm

**Paper**
Dabaghi-Zarandi, F., Afkhami, S. & Ashoori, M., "Community Detection method
based on Random walk and Multi objective Evolutionary algorithm in complex
networks", *Journal of Network and Computer Applications* 234:104070, 2025.
DOI 10.1016/j.jnca.2024.104070.

Softmax random-walk transition rule cited by the paper: Blanchard, P., Higham,
D. J. & Higham, N. J., "Accurately computing the log-sum-exp and softmax
functions", *IMA Journal of Numerical Analysis* 41(4):2311–2330, 2021.

**Original implementation**
No public repository exists. The reference is a **private Python notebook
supplied by the authors** (`src_community_detection_randomwalk.zip`); there is
no URL to cite and none should be invented, so it is vendored in this
repository at [`res/original_algs/cdrme`](../../../../res/original_algs/cdrme),
converted from the notebook to plain Python with its code cells verbatim and
its rendered outputs dropped. It was consulted only to disambiguate
what the paper leaves open, and is followed nowhere it contradicts the text.
Four of its choices were deliberately **not** reproduced:

- Its fitness `kkm_rc()` is not the paper's objective and is mathematically
  degenerate. It returns `kkm = 1 − Σ_c inner_c / (2·Σ_c inner_c) = 0.5` and
  `rc = Σ_c outer_c / (2·Σ_c outer_c) = 0.5` for **every** partition, so
  `kkm_rc_fitness` is the constant `1.0` and carries no search signal at all.
  Eqs. (9)–(12) are implemented instead.
- Its similarity is `alpha·|N(v) ∩ N(u)|/|N(v)| + (1−alpha)·connection(v,u)`,
  which is not Eq. (4). Eq. (4) is implemented.
- Its walker is uniform over neighbours; Algorithm 1 line 10 is softmax-weighted
  by the common-neighbour count. Algorithm 1 is implemented.
- Its centre count is 10% of the nodes; Eq. (6) is `|V| / AvgDegree(G)`. Eq. (6)
  is implemented.

Where the notebook and the paper agree, the notebook is followed: it selects its
final chromosome by **modularity**, which resolves the paper's unspecified
Sec. 4.4.4 rule (see **Objectives**).

**Objectives**
One objective, **maximised**. Eq. (12):

    ObjFunc(C) = innerLinkage(C) + outerLinkage(C)

    innerLinkage(C) = Σ_c innerLinks(c) / totalLinks(c)                    (9)
    outerLinkage(c) = max_{c' ≠ c}  outerLinks(c,c') / totalLinks(c)      (11)
    outerLinkage(C) = Σ_c [ max_b outerLinkage(b) − outerLinkage(c) ]     (10)

`innerLinks(c)` counts links with both ends in `c`; `totalLinks(c)` counts links
with **at least one** end in `c`; `outerLinks(c,c')` counts links between `c` and
`c'`. `objective.rs` evaluates Eq. (10) in the algebraically identical form
`k·max_b outerLinkage(b) − Σ_c outerLinkage(c)`, which is one pass instead of
two nested ones.

There is no second objective, no dominance test and no non-dominated sorting
anywhere in the paper: the "multi-objective" name refers to the two linkage
measures being summed. Hence **no Pareto front and no `cdrme_fronts` entry
point**.

*Both terms grow with `k`.* On the paper's own Fig. 6 example `ObjFunc = 3.97`
at `k = 5` and exactly `1.0` at `k = 1`, so merging is almost always a downhill
move. That is not a bug in the implementation — it is what makes Sec. 4.3.2's
acceptance rule (below) the load-bearing part of the search, and it is why
Eq. (12) is *not* used to rank chromosomes for mutation (see `elite_size`).

Guarded divisions, all reached by degenerate inputs:
- `totalLinks(c) = 0` (a community holding no link at all): the community
  contributes `0` to both Eq. (9) and Eq. (11) instead of dividing by zero.
- `k = 1`: the `max` of Eq. (11) is over an empty set. `outerLinkage(C)` is then
  defined to be `0` and `ObjFunc` is `innerLinkage` alone.
- `Σ freq(r) = 0` in Eq. (8) (an empty `primWalk`): the similarity column stays
  all-zero except the centre's own gene.
- `m = 0`: modularity is defined to be `0`.

**Selection rule (Sec. 4.4.4)**
The paper says only "the quality of evaluation measures described in
Section 2.3.3" and names three: modularity (Eq. 1), NMI (Eq. 2) and density
(Eq. 3). NMI needs ground truth and cannot ship inside a detector; density is
maximised by the single-community partition and cannot rank on its own.
**Modularity at `gamma = 1` is the shipped rule** — the one label-free measure
left, and the one the reference notebook itself sorts on. Eq. (12) is only the
tie-break under equal modularity, then fewer communities, then the label vector,
so the choice is reproducible.

**Representation**
Two chromosome forms, exactly as in the paper.

*Sec. 4.2.5, `chromosome/genes.rs`* — one gene per node holding
`(center, similarity)`, the `|V|`-gene form. Merging the per-walk chromosomes is
a running element-wise argmax fold rather than the paper's sequential pairwise
pass; Fig. 5 *is* the element-wise argmax of the five Fig. 4 chromosomes, so the
operator is associative and only one chromosome is ever held. Ties go to the
lower centre id.

*Sec. 4.3.1, `chromosome/relation.rs`* — the lighter form: one gene per
community. `inner[c]` is the Fig. 6 loop weight, `adj[c][d]` the link weight and
`total[c]` is `totalLinks(c)`. This is the community relation graph the whole
merge phase runs on, so a merge is `O(deg(c))` on the community graph rather
than `O(m)` on the network.

Isolated nodes keep a dense id so they can be reported, but are absent from
`active` and from every equation: Eq. (4) scores them `0` against everything,
Algorithm 1 cannot step out of them, and including them in Eq. (5) would deflate
`AvgDegree` with vertices the method never touches. They come back as community
`-1`.

**Algorithm**
1. **Pre-processing (4.1, `topology.rs`).** Build sorted, duplicate-free CSR
   rows; self-loops and repeated edges are dropped, so `|E|` and every degree
   are those of the simple graph. `AvgDegree(G) = 2m/|V_active|` (Eq. 5),
   `ENC = round(|V_active| / AvgDegree)` clamped to `[1, |V_active|]` (Eq. 6).
2. **Primary composition (4.2, `primary.rs`), `ENC` rounds.** Each round:
   a. `walk::centers` draws a centre with `P(v) ∝ degree(v)` from a Fenwick
      roulette (4.2.1).
   b. `walk_length(v) = round(Degree(v) + alpha_walk · AvgDegree(G))`, at least
      1 (Eq. 7 — see **Divergences** for why `ENC` cancels).
   c. `walk::procedure::Walker::run` runs Algorithm 1: `n_walk` walks of that
      length, each step choosing a neighbour `v` of `current` with probability
      `softmax(|N(current) ∩ N(v)|)`. Keeps the `walk_length` most frequent
      nodes as `primWalk = {(node, freq)}`.
   d. `similarity::avg_similarity` fills Eq. (8) for every node, then rescales
      the column onto `[0,1]` (see **Divergences**).
   e. `Chromosome::absorb` folds the column into the running chromosome; the
      centre's own gene is pinned to `1.0`.
   f. `Centers::decay` divides each `primWalk` node's draw weight by its
      frequency (4.2.3). It decays, it never excludes, so a node can be drawn
      twice — two walks from one centre simply feed the same gene through the
      fold.
3. **Population and merging (4.3, `evolve/merge.rs`).** Build the relation graph
   once. `pop_size` independent chains run in parallel, chain `slot` getting
   `2·k·slot / pop_size` merge attempts. One attempt: draw a live community `i`
   uniformly, draw a neighbour `j` of `i` uniformly, score the merged set with
   `objective_if_merged` (which never touches the graph, so a rejected merge
   costs nothing to undo), and apply the Sec. 4.3.2 acceptance rule.
4. **Mutation (4.4.1–4.4.3, `evolve/mutate.rs`).** The `elite_size` best chains
   are decoded back to the `|V|`-gene form. Each sweeps every active node in
   ascending order: a gene whose similarity is below `alpha_mut` sums its
   neighbours' similarities per candidate community, takes the arg-max community
   (ties to the lower id), and moves there with new similarity
   `max_sum / total_sum` **only if** that is strictly higher than its current
   similarity and is a different community. Sweeps stop on the first pass that
   moves nothing.
5. **Selection (4.4.4, `evolve/select.rs`).** Max modularity, as above.
6. The boundary hands out dense community ids in ascending node order and gives
   isolated nodes `-1`.

**The Sec. 4.3.2 acceptance rule — the one repaired formula**
The paper replaces `C` by `C'` "with a probability of
`1 − 1/(ObjFunc(C') − ObjFunc(C))`". That expression is a probability only when
the difference is at least 1: a worsening merge (`delta < 0`) puts it **above**
1, and a gain below 1 puts it **below** 0. `evolve::merge::accept` is the
literal formula clamped into `[0,1]`, which is the whole repair:

| `delta = ObjFunc(C') − ObjFunc(C)` | literal value | shipped |
|---|---|---|
| `delta ≤ 0` (or NaN) | `> 1` | accept |
| `0 < delta < 1` | `< 0` | reject |
| `delta ≥ 1` | in `[0,1)` | accept with probability `1 − 1/delta` |

Both branches the paper's own stop condition needs survive: a merge that does
**not** improve the set is taken, which is the only way a chromosome ever
reaches "only one community remains" given that Eq. (12) grows with `k`; and a
merge that gains at least 1 is taken with the stated probability. The perverse
middle branch — a *gain* under 1 is refused — is left exactly as written rather
than "fixed", because inverting it is the easy misreading and it changes the
method.

**Parameters**
| Parameter | Default | Effect |
|---|---|---|
| `alpha_walk` | 1.0 | Eq. (7) length coefficient. The paper gives only "typically 1 to 2"; its own worked example (Fig. 3-b: `deg(0)=5`, `AvgDegree=4`, a 9-node walk) is the `alpha = 1` end. Sanitised into `[0, 2]`; non-finite falls back to the default. |
| `n_walk` | 50 | Walks per centre (Algorithm 1). No value is given. The notebook runs 500 walks of a fixed length 4; Eq. (7) makes the length `Degree(v) + AvgDegree` instead, so 50 walks cover a comparable number of steps per centre. At least 1. |
| `pop_size` | 300 | `N_p`. Every chromosome starts identical and the merge budget is stratified over the slots, so this is how many points along the merge chain are sampled, not a breeding pool. Cost is linear in it (measured on a 50k-node, 241k-edge SBM, 48 cores: **6.0 s at 100, 14.3 s at 300**). At least 1. |
| `elite_size` | 300 (= `pop_size`) | `N_sp ≤ N_p` (Sec. 4.4.1), the chromosomes with "acceptable" `ObjFunc` that reach mutation. Neither the count nor "acceptable" is defined, and Eq. (12) grows with `k`, so ranking by it drops exactly the coarse chromosomes Sec. 4.4.4 has to choose between. The shipped value is the upper end of the paper's own `N_sp ≤ N_p`: keep them all. Clamped into `[1, pop_size]`. |
| `alpha_mut` | 0.5 | The Sec. 4.4.2 threshold. No value and no range is given anywhere in the paper; on the `[0,1]` similarity scale of Figs. 4 and 7 the midpoint marks a gene as more unlike than like its own centre. Non-finite falls back to the default. |
| `mut_sweeps` | 10 | Cap on the 4.4.2 ↔ 4.4.3 loop, which the paper leaves unbounded. The loop also stops on the first sweep that moves no gene, so this is a ceiling, not a count. |

Internal, not reachable from Python:

| Constant | Value | Effect |
|---|---|---|
| `MERGE_ATTEMPTS_PER_COMMUNITY` | 2 | Sets the total merge budget `2·k`, which slot `s` receives `s/pop_size` of. The paper's only stated stop is "the best objective function is reached or only one community remains"; since a merge that does not gain is always taken, two attempts per starting community lets the deepest slots run the ladder down to a single community. |
| `RNG_BASE` | `0x5CA1_E5EED` | Shared with `rimpso` and `gdpso` so every module documents one RNG contract. |
| `SALT_CENTER` / `SALT_WALK` / `SALT_MERGE` | `0x0CD_0001` … `0x0CD_0003` | The three independent stream families. |

**Determinism**
Bit-deterministic and independent of the thread count (measured:
`api.rs::two_runs_are_byte_identical`,
`api.rs::the_thread_count_does_not_change_the_result`). What enforces it:

- Every draw comes from `slot_rng(salt, slot)`, one independent stream per
  `(salt, slot)` pair, so no draw depends on rayon's scheduling. Centre
  selection uses `(SALT_CENTER, 0)`, walk round `r` uses `(SALT_WALK, r)`, merge
  chain `s` uses `(SALT_MERGE, s)`.
- The `ENC` composition rounds are sequential: each round's `Centers::decay`
  feeds the next round's draw.
- A merge chain never reads another chain, so the `pop_size` chains are
  genuinely independent and the rayon pass is byte-exact. Same for the
  `elite_size` mutation tasks.
- `Relation::neighbors` sorts the community ids before the uniform draw, so no
  draw ever depends on `FxHashMap` iteration order.
- Every ranking is a total order on values that are already computed: the chain
  sort is `objective` desc → `k` asc → labels asc; `best` is modularity →
  Eq. (12) → fewer communities → labels; `Chromosome::absorb` breaks similarity
  ties to the lower centre id; `mutate` breaks sum ties to the lower community
  id; `Walker::run` breaks frequency ties by visit count then node id.
- `softmax_cumulative` shifts the exponent by the row maximum before `exp` (the
  accurate-softmax form of the paper's own citation). Without the shift a dense
  row overflows on raw common-neighbour counts.

**Divergences**
Written from the paper, so these are departures from the text, or from the
private notebook where the text is silent.

- **`ENC` cancels out of Eq. (7).** Eq. (6) makes `ENC = |V| / AvgDegree(G)`, so
  Eq. (7)'s `|V|/ENC` term is identically `AvgDegree(G)` and the walk length
  collapses to `Degree(v) + alpha_walk · AvgDegree(G)`. `ENC` survives only as
  the *number* of walks, i.e. the number of primary communities. The code
  computes the collapsed form; the algebra is noted at the site.
- **Eq. (8) is rescaled onto `[0,1]`.** Eq. (8) is a frequency-weighted mean of
  Eq. (4), so it is an unbounded count whose scale is set by how dense the
  centre's neighbourhood is. Figs. 4 and 5 nonetheless show every gene in
  `[0,1]` with each centre at exactly `1.0`, and the Sec. 4.4.2 threshold
  `alpha` is only meaningful on a bounded scale. Each walk's column is divided
  by its own maximum. Without it, the Sec. 4.2.5 merge would compare raw
  neighbourhood densities and hand almost every gene to whichever centre sits in
  the densest region.
- **The acceptance probability is clamped into `[0,1]`** — see the table above.
  This is the single place where the literal formula had to be repaired.
- **The merge schedule is stratified, not drawn.** Sec. 4.3.2 hands each
  iteration to one uniformly random chromosome, so the deeper a chromosome is
  drawn the further down its own merge chain it sits. Drawing the schedule
  uniformly spreads those depths as a Poisson around the mean, which
  concentrates as `k` grows and stops covering the ladder; giving slot `s` a
  fixed `s/pop_size` share of the same total budget samples the whole ladder for
  every `k`, and makes the chains independent (hence parallel and
  thread-count-free).
- **Sec. 4.2.5's pairwise merge is a running argmax fold**, which is the same
  operator — see **Representation**.
- **Mutation order and iteration count are chosen.** The paper fixes neither.
  Genes are rewritten in place in ascending node order, so a move is visible to
  the rest of the sweep, and the loop stops at a fixed point or at
  `mut_sweeps`.
- **A gene whose neighbours are all already in its own community is left
  alone.** Fig. 7 is a *membership change*; rewriting the gene in place would
  pin its similarity at `1.0` and make it permanently immune to mutation.
- **`elite_size` defaults to `pop_size`**, i.e. the Sec. 4.4.1 filter is
  effectively off. Ranking by Eq. (12) as written would systematically drop the
  coarse chromosomes, because Eq. (12) grows with `k`.
- **Empty communities are counted in the `k` of Eq. (10)** when Eq. (12) is
  re-evaluated *after* mutation. Mutation can empty a community; the relation
  graph is rebuilt with the pre-mutation `k`, and a community with
  `totalLinks = 0` contributes `0` to Eqs. (9) and (11) but still contributes
  `max_b outerLinkage(b)` to Eq. (10)'s bracket. This affects only the tie-break
  in `best`, never the merge chain (which only ever holds non-empty
  communities), and `Candidate::k` is the true distinct-label count.
- **Isolated nodes are reported as `-1`**, not as singleton communities, and are
  excluded from `|V|` in Eqs. (5) and (6). The paper never mentions them.
- **Self-loops and repeated edges are dropped.** Eq. (4)'s `connection(v,u)` and
  `|N(v) ∩ N(u)|` are simple-graph quantities.
- The three evaluation measures of Sec. 2.3.3 are for *evaluation*; only
  modularity is used inside the detector — see **Selection rule**.

**Measured quality**
Shipped defaults, one run (the detector is deterministic, so one run is exact),
48 cores. `k` is the number of non-isolated communities found.

| Network | `n` | `m` | `k` | NMI | AMI | Q | time |
|---|---|---|---|---|---|---|---|
| karate | 34 | 78 | 2 | 0.837 | 0.833 | 0.371 | <0.01 s |
| dolphins | 62 | 159 | 6 | 0.395 | 0.369 | 0.438 | <0.01 s |
| polbooks | 105 | 441 | 3 | 0.406 | 0.394 | 0.404 | <0.01 s |
| football | 115 | 613 | 11 | 0.857 | 0.814 | 0.560 | <0.01 s |
| email-EU | 1 005 | 16 064 | 25 | 0.531 | 0.469 | 0.316 | 0.02 s |
| SBM 5k | 5 000 | 24 094 | 49 | 0.445 | 0.410 | 0.346 | 0.15 s |
| SBM 50k | 50 000 | 241 088 | 139 | 0.436 | 0.388 | 0.350 | 14.3 s |

karate's `k = 2` is the exact club split: `NMI = 0.837` is the fixed value
between the two ground-truth conventions for that network, not a partial
recovery.

**Files**
| Path | Holds |
|---|---|
| `mod.rs` | Module root; re-exports the three entry points, `Config` and the defaults. |
| `api.rs` | `cdrme`, `cdrme_with`, `cdrme_on_graph`, the `search` pipeline, the `Partition` boundary, and the end-to-end tests. |
| `topology.rs` | The sanitised CSR topology, `common_neighbors`, Eq. (4), Eq. (5) and Eq. (6). |
| `sampling.rs` | `slot_rng` (the per-slot RNG contract) and the Fenwick-indexed `Wheel`. |
| `similarity/mod.rs` | Similarity submodule root. |
| `similarity/pairwise.rs` | `accumulate_column`: a whole Eq. (4) column in `O(Σ_{w∈N(r)} deg(w))`, without materialising the matrix. |
| `similarity/walk_sim.rs` | `avg_similarity`: Eq. (8) plus the `[0,1]` rescale. |
| `walk/mod.rs` | Walk submodule root. |
| `walk/centers.rs` | `Centers`: Sec. 4.2.1 degree-weighted draws and the Sec. 4.2.3 frequency decay. |
| `walk/procedure.rs` | `walk_length` (Eq. 7) and `Walker`: Algorithm 1 with the softmax transition row cached per node. |
| `chromosome/mod.rs` | Chromosome submodule root. |
| `chromosome/genes.rs` | `Chromosome`: the `\|V\|`-gene form, the argmax fold, and the dense community decode. |
| `chromosome/relation.rs` | `Relation`: the Fig. 6 community relation graph and its merge. |
| `objective.rs` | Eqs. (9)–(12), and `objective_if_merged`, the trial merge that never touches the graph. |
| `primary.rs` | `compose`: the `ENC` rounds of 4.2.1 → 4.2.2 → 4.2.3 → 4.2.4, folded into one chromosome. |
| `config/mod.rs` | Config submodule root. |
| `config/defaults.rs` | The six shipped defaults plus `MERGE_ATTEMPTS_PER_COMMUNITY`. |
| `config/params.rs` | The `Config` struct, its `Default`, and `sanitised`. |
| `evolve/mod.rs` | Evolve submodule root. |
| `evolve/merge.rs` | `diversify`: one Sec. 4.3.2 merge chain per slot, and the clamped `accept`. |
| `evolve/mutate.rs` | `mutate`: the Sec. 4.4.2 ↔ 4.4.3 loop. |
| `evolve/select.rs` | `modularity`, `Candidate` and `best`: the Sec. 4.4.4 rule. |
