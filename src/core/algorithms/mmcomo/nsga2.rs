use rand::RngExt; // rand 0.10: random_range/random_bool live on RngExt

// ───────────────────────── domination ─────────────────────────
//
// Both objectives are MINIMIZED and stored as (KKM, RC).
// `a` dominates `b` ⇔ a ≤ b on both objectives and a < b on at least one
// (Deb et al. 2002, NSGA-II).
#[inline]
fn dominates(a: (f64, f64), b: (f64, f64)) -> bool {
    let le = a.0 <= b.0 && a.1 <= b.1;
    let lt = a.0 < b.0 || a.1 < b.1;
    le && lt
}

/// Fast non-dominated sort (NSGA-II). Returns a **1-based** rank per
/// individual: rank 1 is the first (best) Pareto front, rank 2 the next, etc.
///
/// Faithful to Deb et al. 2002, Algorithm "fast-non-dominated-sort": for each
/// `p` compute its domination count `n_p` and the set `S_p` of solutions it
/// dominates; front 1 is everyone with `n_p == 0`; peeling each front
/// decrements the counts of dominated members until empty.
pub fn fast_nondominated_sort(objs: &[(f64, f64)]) -> Vec<usize> {
    let n = objs.len();
    let mut rank = vec![0usize; n];
    if n == 0 {
        return rank;
    }

    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n]; // S_p
    let mut dom_count = vec![0usize; n]; // n_p

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

    // First front.
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

/// Crowding distance (NSGA-II). Computed **per rank-group** (front): within each
/// front, for each objective sort the members, give the two boundary members an
/// infinite distance, and accumulate the normalized gap between each member's
/// two neighbours summed over both objectives.
pub fn crowding_distance(objs: &[(f64, f64)], ranks: &[usize]) -> Vec<f64> {
    let n = objs.len();
    let mut dist = vec![0.0f64; n];
    if n == 0 {
        return dist;
    }

    // Group indices by their (1-based) rank.
    let max_rank = *ranks.iter().max().unwrap_or(&0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for i in 0..n {
        groups[ranks[i]].push(i);
    }

    for group in groups.into_iter() {
        if group.is_empty() {
            continue;
        }
        let g = group.len();
        if g == 1 {
            dist[group[0]] = f64::INFINITY;
            continue;
        }

        // Each of the two objectives contributes.
        for obj in 0..2 {
            let key = |idx: usize| -> f64 {
                if obj == 0 {
                    objs[idx].0
                } else {
                    objs[idx].1
                }
            };

            let mut order = group.clone();
            order.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));

            let f_min = key(order[0]);
            let f_max = key(order[g - 1]);
            let span = f_max - f_min;

            // Boundary members get infinite distance.
            dist[order[0]] = f64::INFINITY;
            dist[order[g - 1]] = f64::INFINITY;

            if span <= 0.0 {
                // All equal on this objective → no contribution.
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

/// Environment selection (NSGA-II μ-survivor step). Fills survivors by ascending
/// rank; when the next whole front would overflow `keep`, that last front is
/// truncated by **descending crowding distance**. Returns survivor indices,
/// of size `min(keep, objs.len())`.
pub fn environment_selection(objs: &[(f64, f64)], keep: usize) -> Vec<usize> {
    let n = objs.len();
    let target = keep.min(n);
    if target == 0 {
        return Vec::new();
    }

    let ranks = fast_nondominated_sort(objs);
    let crowd = crowding_distance(objs, &ranks);

    // Bucket indices per rank.
    let max_rank = *ranks.iter().max().unwrap();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for i in 0..n {
        groups[ranks[i]].push(i);
    }

    let mut survivors: Vec<usize> = Vec::with_capacity(target);
    for r in 1..=max_rank {
        let front = &groups[r];
        if front.is_empty() {
            continue;
        }
        if survivors.len() + front.len() <= target {
            survivors.extend_from_slice(front);
        } else {
            // Partially fill the last needed front: highest crowding first.
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

/// Binary tournament selection (NSGA-II crowded-comparison operator). Picks two
/// random contestants; the winner has the **lower rank**, ties broken by the
/// **higher crowding distance**. Returns the winning individual's index.
/// (Provided for completeness; the genetic operators carry their own copy.)
#[allow(dead_code)]
pub fn tournament(ranks: &[usize], crowd: &[f64], r: &mut impl rand::Rng) -> usize {
    let n = ranks.len();
    debug_assert!(n > 0, "tournament on empty population");
    if n == 1 {
        return 0;
    }

    let a = r.random_range(0..n);
    let mut b = r.random_range(0..n);
    while b == a {
        b = r.random_range(0..n);
    }

    if ranks[a] < ranks[b] {
        a
    } else if ranks[b] < ranks[a] {
        b
    } else if crowd[a] >= crowd[b] {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nondominated_sort_two_fronts() {
        // Front 1: (1,4),(2,2),(4,1) are mutually non-dominating.
        // (3,3) is dominated by (2,2) → front 2.
        let objs = vec![(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0)];
        let ranks = fast_nondominated_sort(&objs);
        assert_eq!(ranks[0], 1);
        assert_eq!(ranks[1], 1);
        assert_eq!(ranks[2], 1);
        assert_eq!(ranks[3], 2);
    }

    #[test]
    fn test_crowding_boundaries_infinite() {
        let objs = vec![(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)];
        let ranks = fast_nondominated_sort(&objs);
        let crowd = crowding_distance(&objs, &ranks);
        // Sorted by obj0: (1,4),(2,2),(4,1) → ends are 0 and 2.
        assert!(crowd[0].is_infinite());
        assert!(crowd[2].is_infinite());
        assert!(crowd[1].is_finite());
    }

    #[test]
    fn test_environment_selection_size_and_rank1_priority() {
        let objs = vec![(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 3.0)];
        let surv = environment_selection(&objs, 3);
        assert_eq!(surv.len(), 3);
        // The three rank-1 members must all survive; the dominated (idx 3) drops.
        assert!(!surv.contains(&3));
        assert!(surv.contains(&0) && surv.contains(&1) && surv.contains(&2));
    }

    #[test]
    fn test_environment_selection_clamps_to_len() {
        let objs = vec![(1.0, 1.0), (2.0, 2.0)];
        assert_eq!(environment_selection(&objs, 10).len(), 2);
        assert_eq!(environment_selection(&objs, 0).len(), 0);
        assert!(environment_selection(&[], 5).is_empty());
    }

    #[test]
    fn test_tournament_prefers_lower_rank() {
        let ranks = vec![1usize, 2usize];
        let crowd = vec![0.0f64, 100.0f64];
        let mut r = rand::rng();
        // Regardless of draw order, the rank-1 individual (idx 0) must win.
        for _ in 0..50 {
            assert_eq!(tournament(&ranks, &crowd, &mut r), 0);
        }
    }

    #[test]
    fn test_tournament_tiebreak_crowding() {
        let ranks = vec![1usize, 1usize];
        let crowd = vec![5.0f64, 1.0f64];
        let mut r = rand::rng();
        for _ in 0..50 {
            // Equal rank → higher crowding (idx 0) wins for either draw order.
            assert_eq!(tournament(&ranks, &crowd, &mut r), 0);
        }
    }
}