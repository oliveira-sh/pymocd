//! Self-contained, single-threaded NSGA-III generational loop (Deb & Jain,
//! IEEE TEC 18(4):577-601, 2014) over the locus representation, plus the
//! paper's two customizations: (a) the duplicate-permutation filter and
//! (b) the single-community exclusion. The reference-point math (Das-Dennis
//! generation, normalization, association, niche-preserving fill) follows
//! the same published algorithm as this repo's shared `nsga3` engine -- that
//! is the paper's method, not this repo's optimization -- but every line here
//! is its own implementation: nothing in this file calls the shared
//! `core::metaheuristics::nsga3` module's entry point, and nothing in this
//! module's population/evaluation loop uses a data-parallel iterator crate
//! (Rayon) or any other parallelism.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::{Individual, fast_non_dominated_sort};
use super::locus::{Genome, Locus};
use super::operators::{binary_tournament, crossover, mutate};
use crate::core::graph::Graph;
use crate::core::metaheuristics::helpers::objectives::kernel_ratiocut::kkm_ratiocut;
use crate::core::metaheuristics::helpers::operators::get_modularity_from_partition;
use rand::rngs::ThreadRng;
use rand::{RngExt, rng}; // rand 0.10: random_range/random_bool live on RngExt
use rustc_hash::FxHashSet;
use std::cmp::Ordering;

/// Decode + evaluate a freshly-built genome (single function call -- no
/// batch/parallel evaluation anywhere in this module).
fn make_individual(graph: &Graph, locus: &Locus, genome: Genome) -> Individual {
    let partition = locus.decode(&genome);
    let (kkm, rc) = kkm_ratiocut(graph, &partition);
    let q = get_modularity_from_partition(&partition, graph);
    // KKM & RC are minimized (fed as-is); Q is maximized -> feed negated.
    Individual { genome, partition, objectives: vec![kkm, rc, -q], rank: usize::MAX }
}

/// Paper customization (a): after environmental selection produces the new
/// `pop_size` population, every individual after the first occurrence of a
/// given canonical (permutation-relabeled) partition is replaced by a fresh
/// random genome. Paper customization (b): any individual whose decoded
/// partition is the single all-nodes community is also replaced. Both
/// replacements are decoded + evaluated immediately (step (h): their
/// objectives must be current before the next generation's mating).
fn apply_paper_customizations(
    graph: &Graph,
    locus: &Locus,
    pop: &mut [Individual],
    rng: &mut ThreadRng,
) {
    let mut seen: FxHashSet<Vec<i32>> = FxHashSet::default();
    for ind in pop.iter_mut() {
        let canon = locus.canonical_labels(&ind.partition);
        if !seen.insert(canon) {
            *ind = make_individual(graph, locus, locus.random_genome(rng));
        }
    }
    for ind in pop.iter_mut() {
        if locus.is_single_community(&ind.partition) {
            *ind = make_individual(graph, locus, locus.random_genome(rng));
        }
    }
}

/// Generational loop: init -> evaluate -> rank, then for `num_gens`
/// generations: binary-tournament mating -> locus-respecting crossover ->
/// adjacency-constrained mutation -> evaluate offspring -> combine parents +
/// offspring (size 2N) -> NSGA-III environmental selection down to
/// `pop_size` -> the two paper customizations -> re-rank. Returns the final
/// (rank-assigned) population; the caller applies the rank-1 max-modularity
/// decision rule.
pub fn run(
    graph: &Graph,
    locus: &Locus,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Vec<Individual> {
    let mut rng = rng();

    let mut pop: Vec<Individual> = (0..pop_size)
        .map(|_| make_individual(graph, locus, locus.random_genome(&mut rng)))
        .collect();
    fast_non_dominated_sort(&mut pop);

    let m = pop.first().map(|i| i.objectives.len()).unwrap_or(0);
    let ref_points = das_dennis(m, divisions);

    for _generation in 0..num_gens {
        let mut offspring: Vec<Individual> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let pa = binary_tournament(&pop, &mut rng);
            let pb = binary_tournament(&pop, &mut rng);
            let mut child_genome = crossover(&pop[pa].genome, &pop[pb].genome, cross_rate, &mut rng);
            mutate(&mut child_genome, locus, mut_rate, &mut rng);
            offspring.push(make_individual(graph, locus, child_genome));
        }

        let mut combined = std::mem::take(&mut pop); // R = P ∪ Q (size 2N)
        combined.extend(offspring);

        pop = environmental_selection(combined, pop_size, &ref_points);

        apply_paper_customizations(graph, locus, &mut pop, &mut rng);
        // Replacements above change objectives, so ranks must be refreshed
        // before the next generation's tournament reads them.
        fast_non_dominated_sort(&mut pop);
    }

    pop
}

/// NSGA-III survivor selection (Deb & Jain 2014, Algorithm 1): keep whole
/// fronts while they fit, then fill the remaining slots from the splitting
/// front `Fl` by reference-point niching over `St = F1 ∪ … ∪ Fl`.
fn environmental_selection(
    mut combined: Vec<Individual>,
    n: usize,
    ref_points: &[Vec<f64>],
) -> Vec<Individual> {
    if combined.len() <= n {
        fast_non_dominated_sort(&mut combined);
        combined.sort_unstable_by_key(|i| i.rank);
        return combined;
    }

    fast_non_dominated_sort(&mut combined);

    // Bucket individual indices by (1-based) rank.
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
        // |St| == N exactly: F1..Fl fill the population.
        return gather(combined, &chosen);
    }

    let k_chosen = chosen.len();
    let mut st_indices = chosen;
    st_indices.extend(&last_front);

    let picks = niche_select(&combined, &st_indices, k_chosen, n - k_chosen, ref_points);

    let mut keep = st_indices[..k_chosen].to_vec();
    keep.extend(picks.iter().map(|&pos| st_indices[pos]));
    gather(combined, &keep)
}

/// Move the individuals at `indices` out of `combined`, rank-sorted.
fn gather(combined: Vec<Individual>, indices: &[usize]) -> Vec<Individual> {
    let mut keep = vec![false; combined.len()];
    for &i in indices {
        keep[i] = true;
    }
    let mut out: Vec<Individual> = combined
        .into_iter()
        .enumerate()
        .filter_map(|(i, ind)| if keep[i] { Some(ind) } else { None })
        .collect();
    out.sort_unstable_by_key(|i| i.rank);
    out
}

/// Reference-point niching over `St` (Deb & Jain 2014, Algs. 2-4). Returns the
/// positions *within `st_indices`* (always `>= k_chosen`, i.e. members of the
/// splitting front `Fl`) selected to fill the remaining `need` slots.
fn niche_select(
    combined: &[Individual],
    st_indices: &[usize],
    k_chosen: usize,
    need: usize,
    ref_points: &[Vec<f64>],
) -> Vec<usize> {
    let m = combined[st_indices[0]].objectives.len();

    // --- Normalize (Alg. 2): translate by the ideal point, then scale by the
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
    let mut rng = rng();
    let mut picks = Vec::with_capacity(need);
    while picks.len() < need {
        // j_min = argmin ρ over refs that still have an unselected Fl member
        // (a ref with no remaining member is implicitly excluded, ρ=∞). Ties → random.
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

/// Hyperplane intercepts `a_j` through the `M` extreme points: solve
/// `Z·x = 1` for `x = 1/a_j`. Falls back to `max_x f'_j` (then `1.0`) when the
/// system is singular or any intercept is non-positive (Deb & Jain 2014, §IV-C).
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

/// Gauss-Jordan solve of a dense `n×n` system with partial pivoting; `None` if
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

/// Das-Dennis structured reference points: every `m`-tuple of non-negative
/// integers summing to `divisions`, each divided by `divisions` (count
/// `H = C(m + divisions − 1, divisions)`).
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
        // M=2, p=4 → 5 points.
        assert_eq!(das_dennis(2, 4).len(), 5);
    }

    #[test]
    fn gaussian_solve_identity() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let x = gaussian_solve(a, vec![2.0, 4.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9);
        // Singular matrix → None.
        assert!(gaussian_solve(vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![1.0, 1.0]).is_none());
    }
}
