//! ariadne — Adaptive Resolution Inference via Agreement-guided, Density-aware
//! NSGA-II Evolution for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

pub mod gamma_pred;
mod stats;

use crate::core::graph::CsrGraph;
use crate::core::metaheuristics::helpers::objectives::cpm::{cpm_objectives, cpm_q};
use crate::core::metaheuristics::helpers::objectives::sbm_mdl::dl_sbm_score;
use rand::distr::Bernoulli;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use stats::welch_p;
use std::cmp::Ordering;
use std::sync::atomic::AtomicUsize;

mod defaults;
use defaults::*;

pub type DensePartition = Vec<i32>;

/// Deterministic per-work-item RNG, reproducible across thread counts.
#[inline]
fn item_rng(seed: u64, a: u64, b: u64) -> ChaCha8Rng {
    let mixed =
        seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xD1B5_4A32_D192_ED03);
    ChaCha8Rng::seed_from_u64(mixed)
}

#[derive(Clone)]
pub struct Individual {
    pub partition: DensePartition,
    pub objectives: [f64; N_OBJ],
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Individual {
    fn new(partition: DensePartition) -> Self {
        Individual {
            partition,
            objectives: [0.0; N_OBJ],
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    #[inline(always)]
    fn dominates(&self, other: &Individual) -> bool {
        let mut better = false;
        for i in 0..N_OBJ {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                better = true;
            }
        }
        better
    }
}

/// Merges tiny communities into their strongest neighboring community while
/// preserving internally cohesive small groups. Singletons always merge; pairs
/// merge only when they lack an internal edge or have more external than
/// internal support. Returns a refined partition without modifying the input.
fn refine_tiny(g: &CsrGraph, part: &[i32], max_size: usize) -> Vec<i32> {
    let mut p = part.to_vec();
    for _ in 0..5 {
        let mut members: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
        for (u, &c) in p.iter().enumerate() {
            members.entry(c).or_default().push(u);
        }
        let tiny: Vec<i32> = members
            .iter()
            .filter(|(_, v)| v.len() <= max_size)
            .map(|(&c, _)| c)
            .collect();
        if tiny.is_empty() {
            break;
        }
        let mut moved = false;
        for c in tiny {
            let nodes = &members[&c];
            let mut internal = 0i64;
            let mut ext: FxHashMap<i32, i64> = FxHashMap::default();
            for &u in nodes {
                for &v in g.neighbors(u) {
                    let cv = p[v as usize];
                    if cv == c {
                        internal += 1;
                    } else {
                        *ext.entry(cv).or_insert(0) += 1;
                    }
                }
            }
            internal /= 2; // each internal edge counted from both ends
            // target: most shared edges; tie → larger community.
            let target = ext.iter().max_by(|a, b| {
                a.1.cmp(b.1).then_with(|| {
                    let sa = members.get(a.0).map_or(0, |v| v.len());
                    let sb = members.get(b.0).map_or(0, |v| v.len());
                    sa.cmp(&sb)
                })
            });
            let Some((&tc, &te)) = target else { continue };
            if nodes.len() == 1 || internal == 0 || te > internal {
                for &u in nodes {
                    p[u] = tc;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    p
}

fn select_sbm(g: &CsrGraph, front: Vec<Vec<i32>>) -> Vec<i32> {
    if front.is_empty() {
        return vec![0; g.n];
    }
    let scores: Vec<f64> = front.par_iter().map(|p| dl_sbm_score(g, p)).collect();
    let best = (0..front.len())
        .min_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap_or(Ordering::Equal))
        .unwrap();
    front.into_iter().nth(best).unwrap()
}

fn random_population(g: &CsrGraph, pop_size: usize, seed: u64) -> Vec<Individual> {
    let n = g.n;
    (0..pop_size)
        .into_par_iter()
        .map(|i| {
            let mut rng = item_rng(seed, i as u64, 0);
            // Each node a random community in [0, n); ids stay < n forever
            // (operators only copy existing labels), so `vol[part[u]]` is safe.
            let part: Vec<i32> = (0..n).map(|_| rng.random_range(0..n) as i32).collect();
            Individual::new(part)
        })
        .collect()
}

/// Local-move mutation: each selected node adopts its neighbours' majority
/// community. Counts neighbour communities in a dense `vote` array (zero
/// invariant maintained via `touched`) — O(degree) per node, no hashing.
/// (A linear-scan stack table was tried and was ~6x slower: in the early
/// random-partition phase a node's neighbours span ~degree distinct
/// communities, making the scan O(degree^2).)
fn mutate(
    g: &CsrGraph,
    part: &mut [i32],
    rate: f64,
    rng: &mut impl Rng,
    vote: &mut [u32],
    touched: &mut Vec<u32>,
) {
    if rate <= 0.0 {
        return;
    }
    let dist = Bernoulli::new(rate).unwrap();
    for u in 0..g.n {
        if !dist.sample(rng) {
            continue;
        }
        touched.clear();
        let mut best = part[u];
        let mut max_count = 0u32;
        for &v in g.neighbors(u) {
            let c = part[v as usize] as usize;
            if vote[c] == 0 {
                touched.push(c as u32);
            }
            vote[c] += 1;
            if vote[c] > max_count {
                max_count = vote[c];
                best = c as i32;
            }
        }
        if max_count > 0 {
            part[u] = best;
        }
        for &c in touched.iter() {
            vote[c as usize] = 0;
        }
    }
}

/// Ensemble crossover: per node, majority community across `parents`, written
/// into the caller-owned `child` buffer (pooled — no allocation). With ≤
/// ENSEMBLE_SIZE parents there are ≤ ENSEMBLE_SIZE distinct labels per node, so
/// a tiny stack table beats a hashmap (no hashing either).
fn ensemble_crossover(parents: &[&DensePartition], child: &mut [i32]) {
    let n = child.len();
    let np = parents.len();
    let majority = (np / 2 + 1) as u32;
    for node in 0..n {
        let mut comms = [0i32; ENSEMBLE_SIZE];
        let mut counts = [0u32; ENSEMBLE_SIZE];
        let mut len = 0usize;
        let mut best = parents[0][node];
        let mut max_count = 0u32;
        for p in parents {
            let c = p[node];
            let mut k = 0;
            while k < len {
                if comms[k] == c {
                    break;
                }
                k += 1;
            }
            if k == len {
                comms[len] = c;
                counts[len] = 0;
                len += 1;
            }
            counts[k] += 1;
            if counts[k] > max_count {
                max_count = counts[k];
                best = c;
                if max_count >= majority {
                    break;
                }
            }
        }
        child[node] = best;
    }
}

#[inline]
fn tournament(pop: &[Individual], rng: &mut impl Rng) -> usize {
    let mut best = rng.random_range(0..pop.len());
    for _ in 1..TOURNAMENT_SIZE {
        let cand = rng.random_range(0..pop.len());
        if pop[cand].rank < pop[best].rank
            || (pop[cand].rank == pop[best].rank
                && pop[cand].crowding_distance > pop[best].crowding_distance)
        {
            best = cand;
        }
    }
    best
}

/// Builds one offspring per recycled buffer in `bufs` (pooled — no partition
/// allocation in steady state). Each buffer is overwritten with a crossover or
/// cloned parent, then mutated in place.
fn create_offspring(
    pop: &[Individual],
    g: &CsrGraph,
    cross_rate: f64,
    mut_rate: f64,
    bufs: Vec<DensePartition>,
    seed: u64,
    generation: u64,
) -> Vec<Individual> {
    let cross_dist = Bernoulli::new(cross_rate).unwrap();
    bufs.into_par_iter()
        .enumerate()
        .map_init(
            || (vec![0u32; g.n], Vec::<u32>::with_capacity(64)),
            |(vote, touched), (i, mut child)| {
                // Seed by (generation, item index) so the stream is independent
                // of rayon scheduling — reproducible across thread counts.
                let mut rng = item_rng(seed, generation.wrapping_shl(20) ^ i as u64, 1);
                let mut idxs = Vec::with_capacity(ENSEMBLE_SIZE);
                while idxs.len() < ENSEMBLE_SIZE {
                    let i = tournament(pop, &mut rng);
                    if !idxs.contains(&i) {
                        idxs.push(i);
                    }
                }
                let parents: Vec<&DensePartition> =
                    idxs.iter().map(|&i| &pop[i].partition).collect();
                if cross_dist.sample(&mut rng) {
                    ensemble_crossover(&parents, &mut child);
                } else {
                    child.copy_from_slice(parents[rng.random_range(0..parents.len())]);
                }
                mutate(g, &mut child, mut_rate, &mut rng, vote, touched);
                Individual::new(child)
            },
        )
        .collect()
}

fn fast_non_dominated_sort(pop: &mut [Individual]) {
    let n = pop.len();
    if n == 0 {
        return;
    }
    use std::sync::atomic::{AtomicUsize, Ordering as AOrd};

    let dom_count: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
    let relations: Vec<(Vec<usize>, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dominated = Vec::new();
            let mut count = 0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                if pop[i].dominates(&pop[j]) {
                    dominated.push(j);
                } else if pop[j].dominates(&pop[i]) {
                    count += 1;
                }
            }
            (dominated, count)
        })
        .collect();

    let mut dominated_data: Vec<usize> = Vec::new();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(n);
    let mut front: Vec<usize> = Vec::new();
    for (i, (dominated, count)) in relations.into_iter().enumerate() {
        let start = dominated_data.len();
        dominated_data.extend(dominated);
        ranges.push(start..dominated_data.len());
        dom_count[i].store(count, AOrd::Relaxed);
        if count == 0 {
            pop[i].rank = 1;
            front.push(i);
        }
    }

    let mut rank = 1;
    while !front.is_empty() {
        let mut next = Vec::new();
        for &i in &front {
            for &j in &dominated_data[ranges[i].clone()] {
                if dom_count[j].fetch_sub(1, AOrd::Relaxed) == 1 {
                    pop[j].rank = rank + 1;
                    next.push(j);
                }
            }
        }
        rank += 1;
        front = next;
    }
}

fn crowding_distance(pop: &mut [Individual]) {
    if pop.is_empty() {
        return;
    }
    for ind in pop.iter_mut() {
        ind.crowding_distance = 0.0;
    }
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (i, ind) in pop.iter().enumerate() {
        groups.entry(ind.rank).or_default().push(i);
    }
    for idxs in groups.values() {
        if idxs.len() <= 2 {
            for &i in idxs {
                pop[i].crowding_distance = f64::INFINITY;
            }
            continue;
        }
        for obj in 0..N_OBJ {
            let mut sorted = idxs.clone();
            sorted.sort_unstable_by(|&a, &b| {
                pop[a].objectives[obj]
                    .partial_cmp(&pop[b].objectives[obj])
                    .unwrap_or(Ordering::Equal)
            });
            let lo = pop[sorted[0]].objectives[obj];
            let hi = pop[sorted[sorted.len() - 1]].objectives[obj];
            pop[sorted[0]].crowding_distance = f64::INFINITY;
            pop[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;
            if (hi - lo).abs() > f64::EPSILON {
                let scale = 1.0 / (hi - lo);
                for k in 1..sorted.len() - 1 {
                    let prev = pop[sorted[k - 1]].objectives[obj];
                    let next = pop[sorted[k + 1]].objectives[obj];
                    pop[sorted[k]].crowding_distance += (next - prev) * scale;
                }
            }
        }
    }
}

/// Migration topology for the coupled-island ablation (`evolve_islands`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MigMode {
    /// Independent islands — no migration (the shipped consensus behaviour).
    None,
    /// Unidirectional ring: island i receives from (i-1) mod N, wrapping the
    /// finest gamma back to the coarsest.
    Ring,
    /// Coarse->fine chain: island i receives from i-1 for i>0 (gammas are in
    /// ascending order, so low gamma = coarse). Island 0 receives nothing; no
    /// wraparound. Cousin of the (reverted) coarse-seeding experiment.
    Directional,
}

pub struct Ariadne {
    pub graph: CsrGraph,
    pub pop_size: usize,
    pub cross_rate: f64,
    pub mut_rate: f64,
    /// CPM resolution. gamma=1 ⇒ effective resolution = graph edge density.
    /// Higher gamma ⇒ smaller communities.
    pub gamma: f64,
    /// RNG seed — same seed + same thread-independent item seeding ⇒ reproducible.
    pub seed: u64,
    /// Adaptive stopping: every `check_every` generations a Welch t-test compares
    /// the mean population quality of the last `check_every`-gen window to the
    /// previous one. While the gain is significant (p < `conv_pval`) the run
    /// continues; once it is not (plateau / local minimum) it stops. No fixed
    /// generation count. `max_gens` is only a hard safety ceiling.
    pub conv_pval: f64,
    pub check_every: usize,
    pub min_gens: usize,
    pub max_gens: usize,
    /// Generations actually run by the most recent `evolve_seeded` call.
    pub last_gens: AtomicUsize,
}

impl Ariadne {
    pub fn new(graph: CsrGraph) -> Self {
        Ariadne {
            graph,
            pop_size: DEFAULT_POP_SIZE,
            cross_rate: DEFAULT_CROSS_RATE,
            mut_rate: DEFAULT_MUT_RATE,
            gamma: DEFAULT_GAMMA,
            seed: DEFAULT_SEED,
            conv_pval: CONV_PVAL,
            check_every: CHECK_EVERY,
            min_gens: MIN_GENS,
            max_gens: usize::MAX,
            last_gens: AtomicUsize::new(0),
        }
    }

    fn evaluate_population(&self, pop: &mut [Individual], gamma: f64) {
        let g = &self.graph;
        // for_each_init reuses one count buffer per worker thread instead of
        // allocating per individual.
        pop.par_iter_mut().for_each_init(
            || vec![0.0f64; g.n],
            |cnt, ind| {
                ind.objectives = cpm_objectives(g, &ind.partition, cnt, gamma);
            },
        );
    }

    /// Coupled-island evolution with optional migration — the ablation harness
    /// for the "humans migrate between separated populations" idea. Runs one
    /// island per gamma in lockstep for `gens` **fixed** generations (no
    /// per-island adaptive stop, so the budget is identical across migration
    /// modes — a clean comparison). Every `every` generations each destination
    /// island's worst `k` survivors are replaced by clones of a source island's
    /// best `k`, then re-evaluated under the destination gamma (cross-scale
    /// transfer). Source per topology: see `MigMode`. Deterministic migrant
    /// pick (top/worst after sort) + deterministic seeding ⇒ still reproducible.
    /// `emigrate_worst`: when false the source sends its **best** k (classic
    /// elite migration — homogenises the ensemble); when true it sends its
    /// **worst** k (each island keeps its elite, only low-fitness material
    /// circulates — diversity injection that preserves the consensus spread).
    /// Returns the final population of every island, in gamma order (the callers
    /// `evolve_islands` and `run_islands_migration_fronts` reduce these to
    /// per-island bests or the pooled rank-1 front respectively).
    fn evolve_islands_pops(
        &self,
        gammas: &[f64],
        gens: usize,
        mig: MigMode,
        k: usize,
        every: usize,
        emigrate_worst: bool,
    ) -> Vec<Vec<Individual>> {
        let n_isl = gammas.len();
        // Per-island carrying capacity. Starts at pop_size; every immigrant an
        // island accepts raises it PERMANENTLY (the population grows over the run,
        // it is never trimmed back down to pop_size). No buffer pool here — caps
        // grow, so offspring buffers are freshly allocated each generation
        // (this is the ablation path; perf is not the concern).
        let mut cap: Vec<usize> = vec![self.pop_size; n_isl];
        let mut pops: Vec<Vec<Individual>> = Vec::with_capacity(n_isl);
        let mut seeds: Vec<u64> = Vec::with_capacity(n_isl);
        for (gi, &gamma) in gammas.iter().enumerate() {
            let seed = self
                .seed
                .wrapping_add((gi as u64).wrapping_mul(0x1000_0000));
            let mut pop = random_population(&self.graph, self.pop_size, seed);
            self.evaluate_population(&mut pop, gamma);
            pops.push(pop);
            seeds.push(seed);
        }

        // Ensemble adaptive stop: watch the pooled survivor mean-q plateau (a
        // Welch t-test on its last two windows), capped at `gens` generations.
        let mut history: Vec<f64> = Vec::new();
        let (g0, w, alpha) = (STOP_WARMUP, STOP_WINDOW, STOP_ALPHA);

        for g in 0..gens {
            // 1. selection: trim each island to its current capacity (best-first).
            for gi in 0..n_isl {
                let pop = &mut pops[gi];
                fast_non_dominated_sort(pop);
                crowding_distance(pop);
                pop.sort_unstable_by(|a, b| {
                    a.rank.cmp(&b.rank).then_with(|| {
                        b.crowding_distance
                            .partial_cmp(&a.crowding_distance)
                            .unwrap_or(Ordering::Equal)
                    })
                });
                pop.truncate(cap[gi]);
            }

            // Ensemble convergence: pooled survivor mean-q across all islands;
            // stop the whole (lockstep) loop once it plateaus.
            let (mut sumq, mut cntq) = (0.0f64, 0usize);
            for pop in &pops {
                for ind in pop.iter() {
                    sumq += cpm_q(&ind.objectives);
                    cntq += 1;
                }
            }
            history.push(sumq / cntq.max(1) as f64);
            if history.len() >= g0 && history.len() >= 2 * w && history.len().is_multiple_of(w) {
                let h = history.len();
                if welch_p(&history[h - 2 * w..h - w], &history[h - w..h]) >= alpha {
                    break;
                }
            }

            // 2. migration: each source's emigrants (best-k or worst-k per flag)
            //    JOIN the destination, raising its capacity PERMANENTLY — the
            //    population only grows, residents are never evicted. Snapshot
            //    every island's emigrants first so all exchanges see the
            //    pre-migration state. Each source feeds ≤1 destination (ring is a
            //    bijection; directional a chain), so the snapshot is consumed once.
            if mig != MigMode::None && g > 0 && g % every == 0 && k > 0 {
                let mut emigrants: Vec<Vec<DensePartition>> = (0..n_isl)
                    .map(|gi| {
                        let pop = &pops[gi];
                        let kk = k.min(pop.len());
                        let idx: Vec<usize> = if emigrate_worst {
                            (pop.len() - kk..pop.len()).collect()
                        } else {
                            (0..kk).collect()
                        };
                        idx.iter().map(|&i| pop[i].partition.clone()).collect()
                    })
                    .collect();
                for dst in 0..n_isl {
                    let src = match mig {
                        MigMode::Ring => Some((dst + n_isl - 1) % n_isl),
                        MigMode::Directional if dst > 0 => Some(dst - 1),
                        _ => None,
                    };
                    let Some(src) = src else { continue };
                    let gd = gammas[dst];
                    let gref = &self.graph;
                    let mut newcomers: Vec<Individual> = std::mem::take(&mut emigrants[src])
                        .into_iter()
                        .map(Individual::new)
                        .collect();
                    newcomers.par_iter_mut().for_each_init(
                        || vec![0.0f64; gref.n],
                        |cnt, ind| ind.objectives = cpm_objectives(gref, &ind.partition, cnt, gd),
                    );
                    cap[dst] += newcomers.len();
                    pops[dst].extend(newcomers);
                }
            }

            // 3. offspring per island — one child per slot at the (grown) capacity.
            for gi in 0..n_isl {
                let bufs: Vec<DensePartition> =
                    (0..cap[gi]).map(|_| vec![0i32; self.graph.n]).collect();
                let mut offspring = create_offspring(
                    &pops[gi],
                    &self.graph,
                    self.cross_rate,
                    self.mut_rate,
                    bufs,
                    seeds[gi],
                    g as u64,
                );
                self.evaluate_population(&mut offspring, gammas[gi]);
                pops[gi].extend(offspring);
            }
        }

        pops
    }

    /// Simpler "migration + pooled front" pipeline: run one island per gamma with
    /// elite-ring migration coupling them, then UNION the rank-1 Pareto fronts of
    /// every island (deduplicated by partition) and return the whole pooled front.
    /// No per-island selection and no gate: `∪_i {rank-1 members of island i}`.
    pub fn run_islands_migration_fronts(
        &self,
        gammas: &[f64],
        gens: usize,
        mig: MigMode,
        k: usize,
        every: usize,
        emigrate_worst: bool,
    ) -> Vec<Vec<i32>> {
        let pops = self.evolve_islands_pops(gammas, gens, mig, k, every, emigrate_worst);
        let mut seen: std::collections::HashSet<Vec<i32>> = std::collections::HashSet::new();
        let mut out: Vec<Vec<i32>> = Vec::new();
        for mut pop in pops {
            fast_non_dominated_sort(&mut pop);
            for ind in pop {
                if ind.rank == 1 && seen.insert(ind.partition.clone()) {
                    out.push(ind.partition);
                }
            }
        }
        out
    }

    /// Ariadne's output: the refined "final frontier". Predicts the five γ
    /// (`gamma_pred`), evolves one island per γ coupled by elite ring migration,
    /// unions their rank-1 Pareto members, then adds the cohesion-guarded
    /// tiny-merge refinement (`refine_tiny`) of every member alongside the raw
    /// ones, deduplicated. A label-free parameter selector over this frontier is
    /// left to future work; the caller receives the whole frontier.
    pub fn run_auto_fronts(&self) -> Vec<Vec<i32>> {
        let gammas = gamma_pred::predict_gammas(&self.graph, self.seed);
        let raw =
            self.run_islands_migration_fronts(&gammas, MIGRATION_GENS, MigMode::Ring, 5, 10, false);
        let mut seen: std::collections::HashSet<Vec<i32>> = raw.iter().cloned().collect();
        let mut out = raw.clone();
        for p in &raw {
            let refined = refine_tiny(&self.graph, p, 2);
            if seen.insert(refined.clone()) {
                out.push(refined);
            }
        }
        out
    }

    /// Ariadne's deployed output: a SINGLE crisp partition. Builds the frontier
    /// (`run_auto_fronts`) and selects the member of minimum SBM description length
    /// (`select_sbm`), a label-free criterion that recovers the frontier's near-
    /// oracle partition with no ground truth.
    pub fn run_auto(&self) -> Vec<i32> {
        select_sbm(&self.graph, self.run_auto_fronts())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn ring(n: usize) -> CsrGraph {
        // Two cliques joined by one edge → clear 2-community structure.
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i as u32, j as u32));
                edges.push(((i + n) as u32, (j + n) as u32));
            }
        }
        edges.push((0, n as u32));
        let total = 2 * n;
        let mut rows = vec![Vec::new(); total];
        for &(u, v) in &edges {
            rows[u as usize].push(v);
            rows[v as usize].push(u);
        }
        let mut xadj = vec![0u32; total + 1];
        for u in 0..total {
            xadj[u + 1] = xadj[u] + rows[u].len() as u32;
        }
        let mut adj = vec![0u32; xadj[total] as usize];
        let mut deg = vec![0u32; total];
        for u in 0..total {
            let s = xadj[u] as usize;
            adj[s..s + rows[u].len()].copy_from_slice(&rows[u]);
            deg[u] = rows[u].len() as u32;
        }
        let mut uniq = Vec::new();
        for u in 0..total as u32 {
            for &v in &rows[u as usize] {
                if u < v {
                    uniq.push((u, v));
                }
            }
        }
        let m = uniq.len();
        CsrGraph {
            n: total,
            m,
            xadj,
            adj,
            deg,
            edges: uniq,
            labels: (0..total as i32).collect(),
        }
    }

    #[test]
    fn finds_two_cliques() {
        let g = ring(6);
        let front = Ariadne::new(g).run_auto_fronts();
        let ok = front.iter().any(|part| {
            let (c_a, c_b) = (part[0], part[6]);
            c_a != c_b && part[..6].iter().all(|&c| c == c_a) && part[6..].iter().all(|&c| c == c_b)
        });
        assert!(ok, "frontier missing the two-clique partition");
    }
}
