//! Shared NSGA-III core (Deb & Jain, IEEE TEC 18(4):577–601, 2014) over the same
//! `Partition` / `Vec<f64>`-objective `Individual` as `nsga2`. Offspring
//! generation (locus-based ensemble crossover + mutation) and the
//! `fast_non_dominated_sort` are reused verbatim from `nsga2`; only the
//! survivor-selection step differs — crowding distance is replaced by
//! reference-point niching (normalization → association → niche-preserving
//! selection). The number of objectives `M` is inferred from
//! `Individual::objectives.len()`, so the engine is objective-count generic.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::Graph;
use crate::core::metaheuristics::helpers::operators;
use crate::core::metaheuristics::helpers::individual::{
    Individual, create_offspring, fast_non_dominated_sort,
};
use rand::{prelude::*, rng};
use rustc_hash::FxHashSet as HashSet;
use std::cmp::Ordering;

/// Generic NSGA-III generational loop. Mirrors `nsga2::evolve`'s signature plus
/// `divisions` (the Das–Dennis reference-point granularity `p`). `evaluate` sets
/// each individual's `objectives`; `on_generation(gen, num_gens, &pop)` runs
/// after each survivor-selection step. Returns the final population
/// (rank-sorted); the caller applies its own rank-1 filter and final selection.
/// Generic over error type `E` so pyo3 callers can propagate `PyErr`.
#[allow(clippy::too_many_arguments)]
pub fn evolve<E>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    tournament_size: usize,
    divisions: usize,
    mut evaluate: impl FnMut(&mut [Individual]) -> Result<(), E>,
    mut on_generation: impl FnMut(usize, usize, &[Individual]) -> Result<(), E>,
) -> Result<Vec<Individual>, E> {
    use rayon::prelude::*;

    let mut pop: Vec<Individual> = operators::generate_population(graph, pop_size)
        .into_par_iter()
        .map(Individual::new)
        .collect();
    evaluate(&mut pop)?;
    // Ranks for generation-0 mating selection (NSGA-III mating may be random,
    // but the reused nsga2 tournament reads `rank`).
    fast_non_dominated_sort(&mut pop);

    // M is inferred from the evaluated objectives; reference points are built once.
    let m = pop.first().map(|i| i.objectives.len()).unwrap_or(0);
    let ref_points = das_dennis(m, divisions);

    for generation in 0..num_gens {
        let mut offspring = create_offspring(&pop, graph, cross_rate, mut_rate, tournament_size);
        evaluate(&mut offspring)?;

        let mut combined = std::mem::take(&mut pop); // R = P ∪ Q (size 2N)
        combined.extend(offspring);

        pop = environmental_selection(combined, pop_size, &ref_points);
        on_generation(generation, num_gens, &pop)?;
    }

    Ok(pop)
}

/// NSGA-III survivor selection (Deb & Jain 2014, Algorithm 1): keep whole fronts
/// while they fit, then fill the remaining slots from the splitting front `Fl`
/// by reference-point niching over `St = F1 ∪ … ∪ Fl`.
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

    // Accumulate fronts until the next would overflow N. `chosen` = F1..Fl-1
    // (all kept); `last_front` = Fl (the splitting front, niched). `st_indices`
    // = chosen ++ last_front, so st_indices[..chosen.len()] are the already-kept
    // members and st_indices[chosen.len()..] are the Fl candidates.
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
    let keep: HashSet<usize> = indices.iter().copied().collect();
    let mut out: Vec<Individual> = combined
        .into_iter()
        .enumerate()
        .filter_map(|(i, ind)| if keep.contains(&i) { Some(ind) } else { None })
        .collect();
    out.sort_unstable_by_key(|i| i.rank);
    out
}

/// Reference-point niching over `St` (Deb & Jain 2014, Algs. 2–4). Returns the
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
        .map(|&idx| {
            (0..m)
                .map(|j| combined[idx].objectives[j] - ideal[j])
                .collect()
        })
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
                    let a = if intercepts[j].abs() < 1e-10 {
                        1.0
                    } else {
                        intercepts[j]
                    };
                    t[j] / a
                })
                .collect()
        })
        .collect();

    // --- Associate each St member to its nearest reference line (Alg. 3). ---
    let ref_norm2: Vec<f64> = ref_points
        .iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f64>())
        .collect();
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
                .min_by(|a, b| {
                    assoc[*a.1]
                        .1
                        .partial_cmp(&assoc[*b.1].1)
                        .unwrap_or(Ordering::Equal)
                })
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
            if a.iter().all(|&aj| aj > 1e-6) {
                a
            } else {
                fallback()
            }
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
    rec(0, divisions, m, divisions.max(1), &mut point, &mut out);
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

    #[test]
    fn perp_distance_on_axis() {
        // Point on the reference line → zero perpendicular distance.
        let d = perp_distance(&[2.0, 2.0], &[1.0, 1.0], 2.0);
        assert!(d < 1e-9);
        // Point orthogonal to the line.
        let d = perp_distance(&[1.0, 0.0], &[0.0, 1.0], 1.0);
        assert!((d - 1.0).abs() < 1e-9);
    }
}
