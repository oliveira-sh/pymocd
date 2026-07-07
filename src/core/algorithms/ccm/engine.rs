//! CCM's own, single-threaded NSGA-III loop (Deb & Jain, IEEE TEC 18(4):577–601,
//! 2014) over locus genomes (see `super::locus`), plus the paper's two search
//! customizations — the duplicate-permutation filter and the single-community
//! exclusion (Shaik, Ravi & Deb 2021, Sec. 4). This module never calls the
//! shared NSGA-III engine (`crate::core::metaheuristics::nsga3`'s entry
//! point): representation, dominance, sorting, reference points,
//! niching and offspring generation are all reimplemented here from scratch,
//! and nothing in this file uses data-parallel iterators (Rayon) — every loop
//! over the population is a plain sequential loop.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rand::{Rng, RngExt, rng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;

use super::locus::{self, Genome};

/// One population member: its locus genome, the decoded partition (cached so
/// it's computed once per genome), its objective vector (caller's convention —
/// CCM stores `(-CS, -CF, -Q)` so this engine can dominance-check as a pure
/// minimization problem), and its current Pareto rank (1 = non-dominated).
#[derive(Clone)]
pub struct Individual {
    pub genome: Genome,
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub rank: usize,
}

impl Individual {
    #[inline]
    fn dominates(&self, other: &Individual) -> bool {
        let mut strictly_better = false;
        for i in 0..self.objectives.len() {
            if self.objectives[i] > other.objectives[i] {
                return false;
            }
            if self.objectives[i] < other.objectives[i] {
                strictly_better = true;
            }
        }
        strictly_better
    }
}

fn new_individual(nodes: &[NodeId], index_of: &FxHashMap<NodeId, usize>, genome: Genome) -> Individual {
    let partition = locus::decode(nodes, index_of, &genome);
    Individual {
        genome,
        partition,
        objectives: Vec::new(),
        rank: usize::MAX,
    }
}

/// Sequential (single-threaded, no data-parallel iterators) fast
/// non-dominated sort (Deb et al. 2002, Alg. 1); assigns each individual's
/// `rank` field (1-based).
pub fn fast_non_dominated_sort(pop: &mut [Individual]) {
    let n = pop.len();
    if n == 0 {
        return;
    }

    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_count: Vec<usize> = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if pop[i].dominates(&pop[j]) {
                dominated[i].push(j);
            } else if pop[j].dominates(&pop[i]) {
                dom_count[i] += 1;
            }
        }
    }

    let mut front: Vec<usize> = (0..n).filter(|&i| dom_count[i] == 0).collect();
    let mut rank = 1usize;
    while !front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &front {
            pop[i].rank = rank;
            for &j in &dominated[i] {
                dom_count[j] -= 1;
                if dom_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        rank += 1;
        front = next_front;
    }
}

/// Binary tournament mating selection: lower rank wins; ties broken randomly
/// (NSGA-III has no crowding distance to tie-break with).
fn binary_tournament(pop: &[Individual], rng: &mut impl Rng) -> usize {
    let i = rng.random_range(0..pop.len());
    let j = rng.random_range(0..pop.len());
    match pop[i].rank.cmp(&pop[j].rank) {
        Ordering::Less => i,
        Ordering::Greater => j,
        Ordering::Equal => {
            if rng.random_bool(0.5) {
                i
            } else {
                j
            }
        }
    }
}

/// Relabel a partition's communities by first-seen order over `nodes`, so that
/// permutation-equivalent partitions (same grouping, different community ids)
/// compare equal. Used by the duplicate-permutation filter.
fn canonical_labels(nodes: &[NodeId], partition: &Partition) -> Vec<i32> {
    let mut next_id = 0i32;
    let mut remap: FxHashMap<CommunityId, i32> = FxHashMap::default();
    nodes
        .iter()
        .map(|node| {
            let c = partition[node];
            *remap.entry(c).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

/// Paper customizations (Shaik, Ravi & Deb 2021, Sec. 4), applied to the
/// freshly environmentally-selected population:
///   (a) duplicate-permutation filter — every individual after the first
///       occurrence of a given canonical partition is replaced;
///   (b) single-community exclusion — any individual whose decoded partition
///       is the single all-spanning community is replaced.
/// Replacements get a fresh random genome and are re-evaluated immediately so
/// their objectives/rank are current going into the next generation.
fn apply_customizations<F>(
    pop: &mut [Individual],
    graph: &Graph,
    nodes: &[NodeId],
    index_of: &FxHashMap<NodeId, usize>,
    rng: &mut impl Rng,
    evaluate: &mut F,
) where
    F: FnMut(&Partition) -> Vec<f64>,
{
    let mut replace = |ind: &mut Individual| {
        *ind = new_individual(nodes, index_of, locus::random_genome(graph, nodes, rng));
        ind.objectives = evaluate(&ind.partition);
    };

    // (a) duplicate-permutation filter.
    let mut seen: FxHashSet<Vec<i32>> = FxHashSet::default();
    for ind in pop.iter_mut() {
        let canon = canonical_labels(nodes, &ind.partition);
        if !seen.insert(canon) {
            replace(ind);
        }
    }

    // (b) single-community exclusion.
    for ind in pop.iter_mut() {
        let distinct: FxHashSet<CommunityId> = ind.partition.values().copied().collect();
        if distinct.len() == 1 {
            replace(ind);
        }
    }
}

/// CCM's self-contained NSGA-III generational loop. `evaluate` maps a decoded
/// partition to its objective vector (caller's min/max sign convention).
/// Returns the final, rank-sorted population.
#[allow(clippy::too_many_arguments)]
pub fn evolve<F>(
    graph: &Graph,
    nodes: &[NodeId],
    index_of: &FxHashMap<NodeId, usize>,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
    mut evaluate: F,
) -> Vec<Individual>
where
    F: FnMut(&Partition) -> Vec<f64>,
{
    let mut r = rng();

    // Initial population (gen 0): pop_size random genomes, evaluated
    // single-threaded, then ranked for generation-0 mating selection.
    let mut pop: Vec<Individual> = (0..pop_size)
        .map(|_| new_individual(nodes, index_of, locus::random_genome(graph, nodes, &mut r)))
        .collect();
    for ind in pop.iter_mut() {
        ind.objectives = evaluate(&ind.partition);
    }
    fast_non_dominated_sort(&mut pop);

    let m = pop.first().map(|i| i.objectives.len()).unwrap_or(0);
    let ref_points = das_dennis(m, divisions);

    for _ in 0..num_gens {
        // a-c: binary-tournament mating + locus-respecting uniform crossover
        // (probability cross_rate; otherwise the child clones one randomly
        // chosen parent) + adjacency-constrained mutation.
        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let pa = binary_tournament(&pop, &mut r);
            let pb = binary_tournament(&pop, &mut r);
            let mut child_genome = if r.random_bool(cross_rate) {
                locus::uniform_crossover(&pop[pa].genome, &pop[pb].genome, &mut r)
            } else {
                let pick = if r.random_bool(0.5) { pa } else { pb };
                pop[pick].genome.clone()
            };
            locus::mutate(&mut child_genome, graph, nodes, mut_rate, &mut r);
            offspring.push(new_individual(nodes, index_of, child_genome));
        }
        // d: evaluate offspring single-threaded, combine R = P ∪ Q (size 2N).
        for ind in offspring.iter_mut() {
            ind.objectives = evaluate(&ind.partition);
        }
        let mut combined = pop;
        combined.extend(offspring);

        // e: NSGA-III environmental selection down to pop_size.
        pop = environmental_selection(combined, pop_size, &ref_points, &mut r);

        // f-h: paper customizations, then re-rank (mating selection reads
        // `rank`, which must reflect any individuals replaced just now).
        apply_customizations(&mut pop, graph, nodes, index_of, &mut r, &mut evaluate);
        fast_non_dominated_sort(&mut pop);
    }

    pop
}

/// NSGA-III survivor selection (Deb & Jain 2014, Algorithm 1): keep whole
/// fronts while they fit, then fill the remaining slots from the splitting
/// front by reference-point niching.
fn environmental_selection(
    mut combined: Vec<Individual>,
    n: usize,
    ref_points: &[Vec<f64>],
    rng: &mut impl Rng,
) -> Vec<Individual> {
    if combined.len() <= n {
        fast_non_dominated_sort(&mut combined);
        combined.sort_by_key(|i| i.rank);
        return combined;
    }

    fast_non_dominated_sort(&mut combined);

    let max_rank = combined.iter().map(|i| i.rank).max().unwrap_or(0);
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for (i, ind) in combined.iter().enumerate() {
        fronts[ind.rank].push(i);
    }

    let mut chosen: Vec<usize> = Vec::new();
    let mut last_front: Vec<usize> = Vec::new();
    for front in fronts.into_iter().skip(1) {
        if front.is_empty() {
            continue;
        }
        if chosen.len() + front.len() <= n {
            chosen.extend(front);
            if chosen.len() == n {
                break;
            }
        } else {
            last_front = front;
            break;
        }
    }

    if last_front.is_empty() {
        return gather(combined, &chosen);
    }

    let k_chosen = chosen.len();
    let mut st_indices = chosen;
    st_indices.extend(&last_front);

    let picks = niche_select(&combined, &st_indices, k_chosen, n - k_chosen, ref_points, rng);

    let mut keep = st_indices[..k_chosen].to_vec();
    keep.extend(picks.iter().map(|&pos| st_indices[pos]));
    gather(combined, &keep)
}

/// Move the individuals at `indices` out of `combined`, rank-sorted.
fn gather(combined: Vec<Individual>, indices: &[usize]) -> Vec<Individual> {
    let keep: FxHashSet<usize> = indices.iter().copied().collect();
    let mut out: Vec<Individual> = combined
        .into_iter()
        .enumerate()
        .filter_map(|(i, ind)| if keep.contains(&i) { Some(ind) } else { None })
        .collect();
    out.sort_by_key(|i| i.rank);
    out
}

/// Reference-point niching over `St` (Deb & Jain 2014, Algs. 2–4). Returns the
/// positions *within `st_indices`* (always `>= k_chosen`, i.e. members of the
/// splitting front) selected to fill the remaining `need` slots.
fn niche_select(
    combined: &[Individual],
    st_indices: &[usize],
    k_chosen: usize,
    need: usize,
    ref_points: &[Vec<f64>],
    rng: &mut impl Rng,
) -> Vec<usize> {
    let m = combined[st_indices[0]].objectives.len();

    // --- Normalize (Alg. 2): translate by the ideal point, scale by the
    // hyperplane intercepts (with the documented fallbacks). ---
    let mut ideal = vec![f64::INFINITY; m];
    for &idx in st_indices {
        for (j, id) in ideal.iter_mut().enumerate() {
            *id = id.min(combined[idx].objectives[j]);
        }
    }
    let translated: Vec<Vec<f64>> = st_indices
        .iter()
        .map(|&idx| (0..m).map(|j| combined[idx].objectives[j] - ideal[j]).collect())
        .collect();

    // Extreme point per axis: the St member minimizing the achievement
    // scalarizing function with weight 1 on axis j, 1e-6 elsewhere.
    let mut extreme = vec![0usize; m];
    for (j, ext) in extreme.iter_mut().enumerate() {
        let mut best = f64::INFINITY;
        for (i, t) in translated.iter().enumerate() {
            let asf = (0..m)
                .map(|k| t[k] / if k == j { 1.0 } else { 1e-6 })
                .fold(f64::NEG_INFINITY, f64::max);
            if asf < best {
                best = asf;
                *ext = i;
            }
        }
    }
    let intercepts = intercepts(&translated, &extreme, m);
    let normalized: Vec<Vec<f64>> = translated
        .iter()
        .map(|t| {
            (0..m)
                .map(|j| {
                    let a = if intercepts[j].abs() < 1e-10 { 1.0 } else { intercepts[j] };
                    t[j] / a
                })
                .collect()
        })
        .collect();

    // --- Associate each St member to its nearest reference line (Alg. 3). ---
    let ref_norm2: Vec<f64> = ref_points.iter().map(|r| r.iter().map(|v| v * v).sum::<f64>()).collect();
    let assoc: Vec<(usize, f64)> = normalized
        .iter()
        .map(|pt| {
            let mut best_r = 0;
            let mut best_d = f64::INFINITY;
            for (ri, rp) in ref_points.iter().enumerate() {
                let d = perp_distance(pt, rp, ref_norm2[ri]);
                if d < best_d {
                    best_d = d;
                    best_r = ri;
                }
            }
            (best_r, best_d)
        })
        .collect();

    // Niche counts from already-chosen members (F1..Fl-1).
    let mut rho = vec![0usize; ref_points.len()];
    for a in &assoc[..k_chosen] {
        rho[a.0] += 1;
    }
    // Fl candidates bucketed by associated reference point (positions into assoc).
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ref_points.len()];
    for (pos, a) in assoc.iter().enumerate().skip(k_chosen) {
        members[a.0].push(pos);
    }

    // --- Niche-preserving selection (Alg. 4). ---
    let mut picks = Vec::with_capacity(need);
    while picks.len() < need {
        let min_rho = match rho
            .iter()
            .enumerate()
            .filter(|(j, _)| !members[*j].is_empty())
            .map(|(_, &v)| v)
            .min()
        {
            Some(v) => v,
            None => break,
        };
        let candidates: Vec<usize> = (0..ref_points.len())
            .filter(|&j| !members[j].is_empty() && rho[j] == min_rho)
            .collect();
        let j = candidates[rng.random_range(0..candidates.len())];

        let pick = if rho[j] == 0 {
            // First member for an empty niche: the one closest to the ref line.
            members[j]
                .iter()
                .enumerate()
                .min_by(|a, b| assoc[*a.1].1.partial_cmp(&assoc[*b.1].1).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap()
        } else {
            rng.random_range(0..members[j].len())
        };
        let pos = members[j].swap_remove(pick);
        picks.push(pos);
        rho[j] += 1;
    }
    picks
}

/// Hyperplane intercepts `a_j` through the `M` extreme points: solve `Z·x = 1`
/// for `x = 1/a_j`. Falls back to `max_x f'_j` (then `1.0`) when the system is
/// singular or any intercept is non-positive (Deb & Jain 2014, §IV-C).
fn intercepts(translated: &[Vec<f64>], extreme: &[usize], m: usize) -> Vec<f64> {
    let fallback = || -> Vec<f64> {
        (0..m)
            .map(|j| {
                let mx = translated.iter().map(|t| t[j]).fold(0.0_f64, f64::max);
                if mx > 1e-10 { mx } else { 1.0 }
            })
            .collect()
    };
    let z: Vec<Vec<f64>> = extreme.iter().map(|&i| translated[i].clone()).collect();
    match gaussian_solve(z, vec![1.0; m]) {
        Some(x) if x.iter().all(|&v| v.abs() > 1e-10) => {
            let a: Vec<f64> = x.iter().map(|&v| 1.0 / v).collect();
            if a.iter().all(|&aj| aj > 1e-6) { a } else { fallback() }
        }
        _ => fallback(),
    }
}

/// Gauss–Jordan solve of a dense `n×n` system with partial pivoting; `None` if
/// (near-)singular.
fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        let pivot_row = a[col].clone();
        let pivot_b = b[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col] / d;
            for (c, val) in a[r].iter_mut().enumerate().skip(col) {
                *val -= factor * pivot_row[c];
            }
            b[r] -= factor * pivot_b;
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

/// Perpendicular distance from `point` to the line through the origin with
/// direction `ref_dir` (squared norm `rnorm2` precomputed).
fn perp_distance(point: &[f64], ref_dir: &[f64], rnorm2: f64) -> f64 {
    if rnorm2 < 1e-30 {
        return point.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
    let dot: f64 = point.iter().zip(ref_dir).map(|(p, r)| p * r).sum();
    let scale = dot / rnorm2;
    point
        .iter()
        .zip(ref_dir)
        .map(|(p, r)| {
            let d = p - scale * r;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Das–Dennis structured reference points: every `m`-tuple of non-negative
/// integers summing to `divisions`, each divided by `divisions`
/// (count `H = C(m + divisions − 1, divisions)`).
fn das_dennis(m: usize, divisions: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    if m == 0 {
        return out;
    }
    let div = divisions.max(1);
    let mut point = vec![0usize; m];
    fn rec(idx: usize, left: usize, m: usize, div: usize, point: &mut [usize], out: &mut Vec<Vec<f64>>) {
        if idx == m - 1 {
            point[idx] = left;
            out.push(point.iter().map(|&v| v as f64 / div as f64).collect());
            return;
        }
        for i in 0..=left {
            point[idx] = i;
            rec(idx + 1, left - i, m, div, point, out);
        }
    }
    rec(0, divisions, m, div, &mut point, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn das_dennis_count() {
        // M=3, p=12 → C(14,2) = 91 reference points, all summing to 1.
        let pts = das_dennis(3, 12);
        assert_eq!(pts.len(), 91);
        for p in &pts {
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert_eq!(das_dennis(2, 4).len(), 5);
    }

    #[test]
    fn dominates_minimization_convention() {
        let mut a = new_individual(&[0], &build_index_for_test(&[0]), vec![0]);
        let mut b = a.clone();
        a.objectives = vec![-1.0, -1.0];
        b.objectives = vec![-1.0, -0.5];
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    fn build_index_for_test(nodes: &[NodeId]) -> FxHashMap<NodeId, usize> {
        locus::build_index(nodes)
    }

    #[test]
    fn fast_non_dominated_sort_ranks() {
        let idx = build_index_for_test(&[0]);
        let mut pop: Vec<Individual> = vec![
            new_individual(&[0], &idx, vec![0]),
            new_individual(&[0], &idx, vec![0]),
            new_individual(&[0], &idx, vec![0]),
        ];
        pop[0].objectives = vec![0.0, 0.0]; // dominates the other two
        pop[1].objectives = vec![1.0, 1.0];
        pop[2].objectives = vec![2.0, 2.0];
        fast_non_dominated_sort(&mut pop);
        assert_eq!(pop[0].rank, 1);
        assert_eq!(pop[1].rank, 2);
        assert_eq!(pop[2].rank, 3);
    }
}
