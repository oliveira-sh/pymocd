//! NSGA-III survivor selection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::individual::{Individual, fast_non_dominated_sort};
use super::niching::niche_select;

/// Deb & Jain 2014, Alg. 1: keep whole fronts while they fit, then fill the
/// remaining slots from the splitting front `Fl` by reference-point niching
/// over `St = F1 ∪ … ∪ Fl`.
pub fn environmental_selection(
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
        return retain_rank_sorted(combined, &chosen);
    }

    let k_chosen = chosen.len();
    let mut st_indices = chosen;
    st_indices.extend(&last_front);

    let picks = niche_select(&combined, &st_indices, k_chosen, n - k_chosen, ref_points);

    let mut keep = st_indices[..k_chosen].to_vec();
    keep.extend(picks.iter().map(|&pos| st_indices[pos]));
    retain_rank_sorted(combined, &keep)
}

fn retain_rank_sorted(combined: Vec<Individual>, indices: &[usize]) -> Vec<Individual> {
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
