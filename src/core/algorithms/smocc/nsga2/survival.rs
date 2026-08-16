//! Environment selection: fill the next population front by front, breaking the
//! last front by crowding distance.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Obj;
use super::crowding::crowding_distance;
use super::sorting::fast_nondominated_sort;

pub fn environment_selection(objs: &[Obj], keep: usize) -> Vec<usize> {
    let n = objs.len();
    let target = keep.min(n);
    if target == 0 {
        return Vec::new();
    }

    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);

    let max_rank = *ranks.iter().max().unwrap();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for i in 0..n {
        groups[ranks[i]].push(i);
    }

    let mut survivors: Vec<usize> = Vec::with_capacity(target);
    for front in &groups[1..=max_rank] {
        if front.is_empty() {
            continue;
        }
        if survivors.len() + front.len() <= target {
            survivors.extend_from_slice(front);
        } else {
            let remaining = target - survivors.len();
            let mut ordered = front.clone();
            ordered.sort_by(|&a, &b| {
                crowd[b]
                    .partial_cmp(&crowd[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            survivors.extend_from_slice(&ordered[..remaining]);
            break;
        }
        if survivors.len() == target {
            break;
        }
    }

    survivors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_selection_size_and_rank1_priority() {
        let objs = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![4.0, 1.0],
            vec![3.0, 3.0],
        ];
        let surv = environment_selection(&objs, 3);
        assert_eq!(surv.len(), 3);
        assert!(!surv.contains(&3));
        assert!(surv.contains(&0) && surv.contains(&1) && surv.contains(&2));
    }

    #[test]
    fn test_environment_selection_clamps_to_len() {
        let objs = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        assert_eq!(environment_selection(&objs, 10).len(), 2);
        assert_eq!(environment_selection(&objs, 0).len(), 0);
        assert!(environment_selection(&[], 5).is_empty());
    }
}
