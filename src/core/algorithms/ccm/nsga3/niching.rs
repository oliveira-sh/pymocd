//! Reference-point niching: which members of the splitting front survive.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};
use std::cmp::Ordering;

use super::individual::Individual;
use super::normalize::hyperplane_coords;

struct Association {
    ref_point: usize,
    distance: f64,
}

/// Deb & Jain 2014, Alg. 4. Returns positions *within `st_indices`* (always
/// `>= k_chosen`, i.e. members of the splitting front) filling the `need`
/// remaining slots.
pub fn niche_select(
    combined: &[Individual],
    st_indices: &[usize],
    k_chosen: usize,
    need: usize,
    ref_points: &[Vec<f64>],
    rng: &mut impl Rng,
) -> Vec<usize> {
    let normalized = hyperplane_coords(combined, st_indices);
    let assoc = associate(&normalized, ref_points);

    let mut niche_count = vec![0usize; ref_points.len()];
    for a in &assoc[..k_chosen] {
        niche_count[a.ref_point] += 1;
    }
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); ref_points.len()];
    for (pos, a) in assoc.iter().enumerate().skip(k_chosen) {
        candidates[a.ref_point].push(pos);
    }

    let mut picks = Vec::with_capacity(need);
    while picks.len() < need {
        let Some(min_count) = niche_count
            .iter()
            .enumerate()
            .filter(|(j, _)| !candidates[*j].is_empty())
            .map(|(_, &v)| v)
            .min()
        else {
            break;
        };
        let least_crowded: Vec<usize> = (0..ref_points.len())
            .filter(|&j| !candidates[j].is_empty() && niche_count[j] == min_count)
            .collect();
        let j = least_crowded[rng.random_range(0..least_crowded.len())];

        let pick = if niche_count[j] == 0 {
            candidates[j]
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    assoc[*a.1]
                        .distance
                        .partial_cmp(&assoc[*b.1].distance)
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap()
        } else {
            rng.random_range(0..candidates[j].len())
        };
        let pos = candidates[j].swap_remove(pick);
        picks.push(pos);
        niche_count[j] += 1;
    }
    picks
}

/// Deb & Jain 2014, Alg. 3: nearest reference line, by perpendicular distance.
fn associate(normalized: &[Vec<f64>], ref_points: &[Vec<f64>]) -> Vec<Association> {
    let ref_norm2: Vec<f64> = ref_points
        .iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f64>())
        .collect();
    normalized
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
            Association {
                ref_point: best_r,
                distance: best_d,
            }
        })
        .collect()
}

/// Distance from `point` to the line through the origin along `ref_dir`, whose
/// squared norm `rnorm2` the caller has precomputed.
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
