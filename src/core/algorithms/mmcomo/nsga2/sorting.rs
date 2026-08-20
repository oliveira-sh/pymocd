//! Pareto dominance and fast non-dominated sorting.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// Both objectives are minimized, stored as `(KKM, RC)`.
#[inline]
fn dominates(a: (f64, f64), b: (f64, f64)) -> bool {
    let le = a.0 <= b.0 && a.1 <= b.1;
    let lt = a.0 < b.0 || a.1 < b.1;
    le && lt
}

/// Fast non-dominated sort (Deb et al. 2002). Ranks are 1-based: rank 1 is
/// the Pareto front every phase of Algorithm 1 reads.
pub fn fast_nondominated_sort(objs: &[(f64, f64)]) -> Vec<usize> {
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
            if dominates(objs[p], objs[q]) {
                dominated[p].push(q);
            } else if dominates(objs[q], objs[p]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nondominated_sort_two_fronts() {
        // (1,4),(2,2),(4,1) mutually non-dominating; (3,3) dominated by (2,2).
        let objs = vec![(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0)];
        let ranks = fast_nondominated_sort(&objs);
        assert_eq!(ranks[0], 1);
        assert_eq!(ranks[1], 1);
        assert_eq!(ranks[2], 1);
        assert_eq!(ranks[3], 2);
    }
}
