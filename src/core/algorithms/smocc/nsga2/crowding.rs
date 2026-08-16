//! Crowding distance: the density estimate that breaks ties inside a front.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Obj;

pub fn crowding_distance(objs: &[Obj], ranks: &[usize]) -> Vec<f64> {
    let n = objs.len();
    let mut dist = vec![0.0f64; n];
    if n == 0 {
        return dist;
    }

    let max_rank = *ranks.iter().max().unwrap_or(&0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for i in 0..n {
        groups[ranks[i]].push(i);
    }

    for group in groups {
        if group.is_empty() {
            continue;
        }
        let g = group.len();
        if g == 1 {
            dist[group[0]] = f64::INFINITY;
            continue;
        }

        let m = objs[group[0]].len();
        #[allow(clippy::needless_range_loop)]
        for obj in 0..m {
            let key = |idx: usize| -> f64 { objs[idx][obj] };

            let mut order = group.clone();
            order.sort_by(|&a, &b| {
                key(a)
                    .partial_cmp(&key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let f_min = key(order[0]);
            let f_max = key(order[g - 1]);
            let span = f_max - f_min;

            dist[order[0]] = f64::INFINITY;
            dist[order[g - 1]] = f64::INFINITY;

            if span <= 0.0 {
                continue;
            }

            for k in 1..(g - 1) {
                if dist[order[k]].is_finite() {
                    dist[order[k]] += (key(order[k + 1]) - key(order[k - 1])) / span;
                }
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::super::sorting::fast_nondominated_sort;
    use super::*;

    #[test]
    fn test_crowding_boundaries_infinite() {
        let objs = vec![vec![1.0, 4.0], vec![2.0, 2.0], vec![4.0, 1.0]];
        let ranks = fast_nondominated_sort(&objs);
        let crowd = crowding_distance(&objs, &ranks);
        assert!(crowd[0].is_infinite());
        assert!(crowd[2].is_infinite());
        assert!(crowd[1].is_finite());
    }
}
