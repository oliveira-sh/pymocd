//! NSGA-III survivor selection: keep whole fronts, niche the splitting one.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::Rng;

use super::individual::{Individual, fast_non_dominated_sort};
use super::niching::niche_select;

/// Deb & Jain 2014, Alg. 1.
pub fn environmental_selection(
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

    let picks = niche_select(
        &combined,
        &st_indices,
        k_chosen,
        n - k_chosen,
        ref_points,
        rng,
    );

    let mut keep = st_indices[..k_chosen].to_vec();
    keep.extend(picks.iter().map(|&pos| st_indices[pos]));
    gather(combined, &keep)
}

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
    out.sort_by_key(|i| i.rank);
    out
}
