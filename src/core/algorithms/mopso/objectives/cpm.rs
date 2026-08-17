//! The Constant Potts Model split into a cut fraction and a pair coverage.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::Obj;

/// The two extensive counts every CPM quantity is built from: the number of
/// intra-community edges, and `sum_c C(n_c,2)`.
///
/// They are integers, which is why the swarm can maintain them through
/// individual node moves instead of rescanning the graph: an integer count
/// updated by integer deltas is still exact after any number of moves.
pub type Counts = (i64, i64);

/// Load `size` with the community sizes of `labels`, recording in `live` which
/// entries were made nonzero. `live` is what a later caller clears, so the
/// `O(n)`-sized `size` array is never memset.
pub fn load_sizes(labels: &[i32], size: &mut [u32], live: &mut Vec<u32>) {
    for &c in live.iter() {
        size[c as usize] = 0;
    }
    live.clear();
    for &c in labels {
        debug_assert!((c as usize) < size.len(), "label {c} outside the slot range");
        let c = c as usize;
        if size[c] == 0 {
            live.push(c as u32);
        }
        size[c] += 1;
    }
}

/// Both counts of a partition, from scratch. Leaves `size`/`live` holding that
/// partition's community sizes, which is what the swarm goes on to maintain.
pub fn measure(g: &CsrGraph, labels: &[i32], size: &mut [u32], live: &mut Vec<u32>) -> Counts {
    load_sizes(labels, size, live);
    let mut pair_sum = 0i64;
    for &c in live.iter() {
        let s = i64::from(size[c as usize]);
        pair_sum += s * (s - 1) / 2;
    }
    let mut internal = 0i64;
    for &(u, v) in &g.edges {
        internal += i64::from(labels[u as usize] == labels[v as usize]);
    }
    (internal, pair_sum)
}

/// The two minimised objectives:
///
/// ```text
/// cut  = 1 - (sum_c e_c) / m           the fraction of edges leaving their community
/// pair = (sum_c C(n_c,2)) / C(n,2)     the fraction of node pairs sharing a community
/// ```
///
/// CPM at resolution `gamma` is `H(gamma) = sum_c [e_c - gamma C(n_c,2)]`, so
///
/// ```text
/// H(gamma) / m = 1 - cut - (gamma / gamma_d) * pair,    gamma_d = m / C(n,2)
/// ```
///
/// Every resolution is a weighted sum of the same two numbers. Minimising them
/// jointly therefore optimises CPM at every `gamma` at once, and the Pareto
/// front is the graph's complete resolution profile rather than one slice of
/// it. `gamma_d` is the graph's density, the resolution at which the two
/// objectives carry equal weight.
///
/// This is HP-MOCD's split of modularity with one term changed: `cut` is the
/// same edge term, and CPM's constant pair penalty replaces modularity's
/// degree null model `sum_c (d_c/2m)^2` — which is precisely the substitution
/// that removes the resolution limit.
///
/// `cut` is also exactly the realised mixing parameter of the partition.
pub fn obj_of(g: &CsrGraph, (internal, pair_sum): Counts) -> Obj {
    let m = g.m as f64;
    let n = g.n as f64;
    let cut = if m > 0.0 { 1.0 - internal as f64 / m } else { 0.0 };
    let pair = if n > 1.0 {
        2.0 * pair_sum as f64 / (n * (n - 1.0))
    } else {
        0.0
    };
    [cut, pair]
}

/// Number of distinct communities in `labels`. `seen` must arrive all-false and
/// is left that way.
pub fn community_count(labels: &[i32], seen: &mut [bool]) -> usize {
    let mut k = 0;
    for &c in labels {
        let c = c as usize;
        if !seen[c] {
            seen[c] = true;
            k += 1;
        }
    }
    for &c in labels {
        seen[c as usize] = false;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::utils::fixtures::two_triangles;

    fn eval(g: &CsrGraph, part: &[i32]) -> Obj {
        let mut size = vec![0u32; g.n];
        let mut live = Vec::new();
        obj_of(g, measure(g, part, &mut size, &mut live))
    }

    #[test]
    fn the_split_reproduces_cpm_at_every_resolution() {
        let g = two_triangles();
        let (n, m) = (g.n as f64, g.m as f64);
        let density = m / (n * (n - 1.0) / 2.0);

        for part in [
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 0, 1, 1, 2, 2],
        ] {
            let [cut, pair] = eval(&g, &part);
            let mut internal = 0.0;
            let mut penalty = 0.0;
            for c in 0..6i32 {
                let sz = part.iter().filter(|&&x| x == c).count() as f64;
                penalty += sz * (sz - 1.0) / 2.0;
                internal += g
                    .edges
                    .iter()
                    .filter(|&&(u, v)| part[u as usize] == c && part[v as usize] == c)
                    .count() as f64;
            }
            for gamma in [0.1, 1.0, density, 3.7] {
                let direct = internal - gamma * penalty;
                let split = m * (1.0 - cut - (gamma / density) * pair);
                assert!(
                    (direct - split).abs() < 1e-9,
                    "part {part:?} gamma {gamma}: {direct} != {split}"
                );
            }
        }
    }

    #[test]
    fn degenerate_partitions_sit_at_the_corners() {
        let g = two_triangles();
        assert_eq!(eval(&g, &[0; 6]), [0.0, 1.0]);
        assert_eq!(eval(&g, &(0..6).collect::<Vec<i32>>()), [1.0, 0.0]);
    }

    #[test]
    fn cut_is_the_realised_mixing_parameter() {
        // Six of the seven edges are internal to the two triangles; (2,3) leaves.
        let g = two_triangles();
        let [cut, _] = eval(&g, &[0, 0, 0, 1, 1, 1]);
        assert!((cut - 1.0 / 7.0).abs() < 1e-12, "cut = {cut}");
    }

    #[test]
    fn relabelling_does_not_move_the_point() {
        let g = two_triangles();
        assert_eq!(eval(&g, &[0, 0, 0, 1, 1, 1]), eval(&g, &[5, 5, 5, 2, 2, 2]));
    }

    #[test]
    fn buffers_are_reusable_across_partitions() {
        let g = two_triangles();
        let mut size = vec![0u32; g.n];
        let mut live = Vec::new();
        let a = measure(&g, &[0, 0, 0, 1, 1, 1], &mut size, &mut live);
        let b = measure(&g, &[5, 5, 5, 2, 2, 2], &mut size, &mut live);
        assert_eq!(a, b, "the buffers leaked between calls");
        assert_eq!(measure(&g, &[0, 0, 0, 1, 1, 1], &mut size, &mut live), a);
    }

    #[test]
    fn community_count_counts_distinct_labels_and_leaves_no_residue() {
        let mut seen = vec![false; 6];
        assert_eq!(community_count(&[0, 0, 0, 1, 1, 1], &mut seen), 2);
        assert_eq!(community_count(&[0, 1, 2, 3, 4, 5], &mut seen), 6);
        assert_eq!(community_count(&[3; 6], &mut seen), 1);
        assert!(seen.iter().all(|&b| !b), "seen was left dirty");
    }
}
