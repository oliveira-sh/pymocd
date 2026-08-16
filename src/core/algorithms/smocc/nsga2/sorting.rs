//! Front assignment: the Jensen sweep that ranks a two-objective population
//! by non-domination in O(n log n).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::Obj;

pub fn fast_nondominated_sort(objs: &[Obj]) -> Vec<usize> {
    if objs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(objs[0].len(), 2, "the Jensen sweep is two-objective only");
    two_objective_sort(objs)
}

fn two_objective_sort(objs: &[Obj]) -> Vec<usize> {
    let n = objs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        objs[a][0]
            .partial_cmp(&objs[b][0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                objs[a][1]
                    .partial_cmp(&objs[b][1])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let mut rank = vec![0usize; n];
    let mut front_min: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < n {
        let (f1, f2) = (objs[idx[i]][0], objs[idx[i]][1]);
        let mut j = i + 1;
        while j < n && objs[idx[j]][0] == f1 && objs[idx[j]][1] == f2 {
            j += 1;
        }
        let k = front_min.partition_point(|&m| m <= f2);
        if k == front_min.len() {
            front_min.push(f2);
        } else {
            front_min[k] = f2;
        }
        for &p in &idx[i..j] {
            rank[p] = k + 1;
        }
        i = j;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn dominates(a: &[f64], b: &[f64]) -> bool {
        let le = a.iter().zip(b).all(|(x, y)| x <= y);
        let lt = a.iter().zip(b).any(|(x, y)| x < y);
        le && lt
    }

    fn generic_sort(objs: &[Obj]) -> Vec<usize> {
        let n = objs.len();
        let mut rank = vec![0usize; n];
        if n == 0 {
            return rank;
        }

        let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut dom_count = vec![0usize; n];

        for p in 0..n {
            for q in 0..n {
                if p == q {
                    continue;
                }
                if dominates(&objs[p], &objs[q]) {
                    dominated[p].push(q);
                } else if dominates(&objs[q], &objs[p]) {
                    dom_count[p] += 1;
                }
            }
        }

        let mut front: Vec<usize> = (0..n).filter(|&p| dom_count[p] == 0).collect();
        let mut r = 1usize;

        while !front.is_empty() {
            let mut next: Vec<usize> = Vec::new();
            for &p in &front {
                rank[p] = r;
                for &q in &dominated[p] {
                    dom_count[q] -= 1;
                    if dom_count[q] == 0 {
                        next.push(q);
                    }
                }
            }
            r += 1;
            front = next;
        }

        rank
    }

    #[test]
    fn test_nondominated_sort_two_fronts() {
        let objs = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.0],
            vec![4.0, 1.0],
            vec![3.0, 3.0],
        ];
        let ranks = fast_nondominated_sort(&objs);
        assert_eq!(ranks[0], 1);
        assert_eq!(ranks[1], 1);
        assert_eq!(ranks[2], 1);
        assert_eq!(ranks[3], 2);
    }

    #[test]
    fn two_objective_sort_matches_generic_on_random_data() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as usize
        };
        for case in 0..200 {
            let n = 1 + next() % 120;
            let lattice = 1 + case % 12;
            let objs: Vec<Obj> = (0..n)
                .map(|_| vec![(next() % lattice) as f64, (next() % lattice) as f64])
                .collect();
            assert_eq!(
                two_objective_sort(&objs),
                generic_sort(&objs),
                "diverged on {objs:?}"
            );
        }
    }

    #[test]
    fn two_objective_sort_keeps_exact_duplicates_in_one_front() {
        let objs = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![0.5, 3.0]];
        assert_eq!(fast_nondominated_sort(&objs), vec![1, 1, 1]);
    }
}
