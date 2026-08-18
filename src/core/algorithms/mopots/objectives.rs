//! The Constant Potts objective pair (cut, pair) and the exact CPM identity behind it.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

const PARALLEL_EDGE_THRESHOLD: usize = 1024;

/// The two Constant Potts objectives, both **minimized**, read over the `n_a`
/// non-isolated nodes (`n_c` likewise counts only non-isolated members):
/// `cut  = 1 − Σ_c |E(c)|/m`   (fraction of edges leaving their community)
/// `pair = Σ_c C(n_c,2)/C(n_a,2)`   (fraction of non-isolated pairs put together)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub cut: f64,
    pub pair: f64,
}

/// The two integer denominators behind the objectives: the edge count `m` and
/// `all_pairs = C(n_a,2)` over the non-isolated nodes.
pub fn denominators(graph: &Graph) -> (f64, f64) {
    (graph.edges.len() as f64, choose_two(active_nodes(graph)))
}

/// Edge density `2m/(n_a(n_a−1))` over the non-isolated nodes, the exchange
/// rate between `cut` and `pair`. Returns `0.0` when `n_a < 2`.
// the ladder cancels this factor out, so only the CPM identity test still reads it
#[cfg_attr(not(test), allow(dead_code))]
pub fn gamma_d(graph: &Graph) -> f64 {
    let n = active_nodes(graph);
    if n < 2 {
        return 0.0;
    }
    let m = graph.edges.len() as f64;
    2.0 * m / (n as f64 * (n - 1) as f64)
}

/// Evaluates `(cut, pair)` for `partition`; `parallel` only switches the O(m) edge scan.
pub fn calculate_objectives(graph: &Graph, partition: &Partition, parallel: bool) -> Metrics {
    let m = graph.edges.len();
    let intra = count_internal_edges(graph, partition, parallel);

    // integer counting with a single final division keeps both paths bit-identical
    let cut = if m == 0 {
        0.0
    } else {
        1.0 - intra as f64 / m as f64
    };

    // n_a counts partition keys, gamma_d graph nodes: a partition must key every active node
    let (n_active, co_clustered) = co_clustered_pairs(graph, partition);
    // C(n_a,2) is zero for n_a < 2, in which case no pair can be co-clustered
    let all_pairs = choose_two(n_active);
    let pair = if all_pairs == 0.0 {
        0.0
    } else {
        co_clustered / all_pairs
    };

    Metrics { cut, pair }
}

/// `H_gamma(C)/m = 1 − cut − (gamma/gamma_d)·pair`, the CPM value of the metrics.
/// `gamma_d == 0.0` means no node carries an edge, so `pair` is identically `0.0`.
// only the identity test reads this spec of (cut, pair)
#[cfg_attr(not(test), allow(dead_code))]
pub fn cpm(metrics: &Metrics, gamma: f64, gamma_d: f64) -> f64 {
    if gamma_d == 0.0 {
        return 1.0 - metrics.cut;
    }
    1.0 - metrics.cut - (gamma / gamma_d) * metrics.pair
}

// graph.edges holds every undirected edge exactly once, so one pass counts them all; an
// isolated node carries none, so the non-isolated scope needs no filter here
fn count_internal_edges(graph: &Graph, partition: &Partition, parallel: bool) -> usize {
    if parallel && graph.edges.len() > PARALLEL_EDGE_THRESHOLD {
        graph
            .edges
            .par_iter()
            .copied()
            .filter(|&(u, v)| is_internal(partition, u, v))
            .count()
    } else {
        graph
            .edges
            .iter()
            .copied()
            .filter(|&(u, v)| is_internal(partition, u, v))
            .count()
    }
}

fn co_clustered_pairs(graph: &Graph, partition: &Partition) -> (usize, f64) {
    let mut sizes: HashMap<CommunityId, usize> = HashMap::default();
    let mut n_active: usize = 0;
    for (&node, &comm) in partition {
        if graph.degree(&node) == 0 {
            continue;
        }
        n_active += 1;
        *sizes.entry(comm).or_insert(0) += 1;
    }
    (n_active, sizes.values().map(|&size| choose_two(size)).sum())
}

#[inline]
fn is_internal(partition: &Partition, u: NodeId, v: NodeId) -> bool {
    match (partition.get(&u), partition.get(&v)) {
        (Some(cu), Some(cv)) => cu == cv,
        _ => false,
    }
}

// the scope of both objectives and of gamma_d: an isolated node is outside the search,
// mutation cannot move it and the run reports it as community -1
fn active_nodes(graph: &Graph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|&node| graph.degree(node) > 0)
        .count()
}

// C(k,2) in f64, so a huge community cannot overflow the product
#[inline]
fn choose_two(k: usize) -> f64 {
    if k < 2 {
        return 0.0;
    }
    let k = k as f64;
    k * (k - 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopots::fixtures::two_triangles;
    use crate::core::graph::karate::{KARATE_CLUB, karate_club};
    use crate::core::graph::normalize_community_ids;

    fn two_triangles_with_isolated(count: NodeId) -> Graph {
        let mut g = two_triangles();
        for node in 6..(6 + count) {
            g.nodes.insert(node);
        }
        g.finalize();
        g
    }

    fn two_triangles_split() -> Partition {
        [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]
            .into_iter()
            .collect()
    }

    fn one_community(graph: &Graph) -> Partition {
        graph.nodes.iter().map(|&node| (node, 0)).collect()
    }

    fn singletons(graph: &Graph) -> Partition {
        graph.nodes.iter().map(|&node| (node, node)).collect()
    }

    fn karate_ground_truth() -> Partition {
        KARATE_CLUB
            .iter()
            .enumerate()
            .map(|(node, &comm)| (node as NodeId, comm))
            .collect()
    }

    // Σ_c (|E(c)| − gamma·C(n_c,2)) / m over non-isolated nodes, from the CPM definition
    fn direct_cpm(graph: &Graph, partition: &Partition, gamma: f64) -> f64 {
        let m = graph.edges.len() as f64;
        let mut internal = 0.0;
        for &(u, v) in &graph.edges {
            if let (Some(cu), Some(cv)) = (partition.get(&u), partition.get(&v))
                && cu == cv
            {
                internal += 1.0;
            }
        }
        let mut sizes: HashMap<CommunityId, usize> = HashMap::default();
        for (&node, &comm) in partition {
            if graph.degree(&node) == 0 {
                continue;
            }
            *sizes.entry(comm).or_insert(0) += 1;
        }
        let co_clustered: f64 = sizes.values().map(|&size| choose_two(size)).sum();
        (internal - gamma * co_clustered) / m
    }

    #[test]
    fn one_community_leaves_no_edge_and_joins_every_pair() {
        let g = karate_club();
        let metrics = calculate_objectives(&g, &one_community(&g), false);
        assert_eq!(metrics.cut, 0.0, "no edge can leave a single community");
        assert_eq!(metrics.pair, 1.0, "every pair is co-clustered");
    }

    #[test]
    fn singletons_leave_every_edge_and_join_no_pair() {
        let g = karate_club();
        let metrics = calculate_objectives(&g, &singletons(&g), false);
        assert_eq!(metrics.cut, 1.0, "every edge leaves its community");
        assert_eq!(metrics.pair, 0.0, "singletons hold no pair");
    }

    #[test]
    fn two_triangles_match_the_hand_computation() {
        let g = two_triangles();
        // n = 6, m = 7, 6 internal edges, C(3,2)+C(3,2) = 6 of the C(6,2) = 15 pairs
        let metrics = calculate_objectives(&g, &two_triangles_split(), false);
        assert!(
            (metrics.cut - 1.0 / 7.0).abs() < 1e-12,
            "cut = 1 - 6/7, got {}",
            metrics.cut
        );
        assert!(
            (metrics.pair - 6.0 / 15.0).abs() < 1e-12,
            "pair = 6/15, got {}",
            metrics.pair
        );
        assert!(
            (gamma_d(&g) - 14.0 / 30.0).abs() < 1e-12,
            "gamma_d = 2*7/(6*5), got {}",
            gamma_d(&g)
        );
    }

    #[test]
    fn cpm_equals_the_potts_hamiltonian_over_m() {
        let karate = karate_club();
        let triangles = two_triangles();
        let padded = two_triangles_with_isolated(3);
        // the padded case pins the identity on the non-isolated node scope
        let mut padded_split = two_triangles_split();
        for (node, comm) in [(6, 0), (7, 7), (8, 1)] {
            padded_split.insert(node, comm);
        }
        let cases: [(&Graph, Partition); 6] = [
            (&triangles, two_triangles_split()),
            (&triangles, one_community(&triangles)),
            (&triangles, singletons(&triangles)),
            (&karate, karate_ground_truth()),
            (&karate, singletons(&karate)),
            (&padded, padded_split),
        ];
        for (graph, partition) in &cases {
            let density = gamma_d(graph);
            let metrics = calculate_objectives(graph, partition, false);
            for gamma in [0.0, 0.1, 0.5, 1.0, 2.0, density] {
                let expected = direct_cpm(graph, partition, gamma);
                let got = cpm(&metrics, gamma, density);
                assert!(
                    (got - expected).abs() < 1e-12,
                    "the CPM identity must be exact: gamma {gamma}, expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn the_denominators_are_m_and_the_active_pair_count() {
        let g = two_triangles_with_isolated(3);
        let (m, all_pairs) = denominators(&g);
        assert_eq!(m, 7.0, "the three isolated nodes carry no edge");
        assert_eq!(all_pairs, 15.0, "C(6,2) over the six non-isolated nodes");
        assert_eq!(m / all_pairs, gamma_d(&g), "the density is their ratio");
    }

    #[test]
    fn degenerate_graph_has_no_density_and_no_pair_term() {
        let g = Graph::new();
        assert_eq!(gamma_d(&g), 0.0, "C(n,2) vanishes for n < 2");
        let metrics = calculate_objectives(&g, &Partition::default(), false);
        assert_eq!(metrics, Metrics::default(), "no edges, no pairs");
        assert_eq!(cpm(&metrics, 1.0, 0.0), 1.0, "there is no pair term to add");
    }

    #[test]
    fn an_edgeless_graph_holds_no_pair_however_it_is_cut() {
        let mut g = Graph::new();
        for node in 0..3 {
            g.nodes.insert(node);
        }
        g.finalize();
        assert_eq!(gamma_d(&g), 0.0, "not one node carries an edge");
        let blob = calculate_objectives(&g, &one_community(&g), false);
        let alone = calculate_objectives(&g, &singletons(&g), false);
        assert_eq!(blob.pair, 0.0, "an isolated node contributes no pair");
        assert_eq!(alone.pair, 0.0, "an isolated node contributes no pair");
        assert_eq!(cpm(&blob, 1.0, 0.0), cpm(&alone, 1.0, 0.0), "pair is zero");
    }

    #[test]
    fn isolated_nodes_move_neither_objective() {
        let g = two_triangles();
        let before = calculate_objectives(&g, &two_triangles_split(), false);

        let padded = two_triangles_with_isolated(4);
        let mut partition = two_triangles_split();
        for (node, comm) in [(6, 0), (7, 1), (8, 2), (9, 2)] {
            partition.insert(node, comm);
        }
        let after = calculate_objectives(&padded, &partition, false);
        assert_eq!(before, after, "an isolated node has no edge and no pair");
        assert_eq!(gamma_d(&padded), gamma_d(&g), "density ignores them too");
    }

    #[test]
    fn objectives_are_invariant_under_normalize_community_ids() {
        let g = two_triangles_with_isolated(3);
        let mut partition = two_triangles_split();
        for (node, comm) in [(6, 0), (7, 7), (8, 1)] {
            partition.insert(node, comm);
        }
        let normalized = normalize_community_ids(&g, partition.clone());
        assert_eq!(
            calculate_objectives(&g, &partition, false),
            calculate_objectives(&g, &normalized, false),
            "the reported metrics must describe the partition the module returns"
        );
    }

    #[test]
    fn the_parallel_path_matches_the_sequential_one() {
        let karate = karate_club();
        let truth = karate_ground_truth();
        assert_eq!(
            calculate_objectives(&karate, &truth, true),
            calculate_objectives(&karate, &truth, false)
        );

        // K60 has 1770 edges, above the threshold, so this really exercises rayon
        let mut dense = Graph::new();
        for u in 0..60 {
            for v in (u + 1)..60 {
                dense.add_edge(u, v);
            }
        }
        dense.finalize();
        let blocks: Partition = dense.nodes.iter().map(|&node| (node, node % 6)).collect();
        assert!(dense.edges.len() > PARALLEL_EDGE_THRESHOLD);
        assert_eq!(
            calculate_objectives(&dense, &blocks, true),
            calculate_objectives(&dense, &blocks, false)
        );
    }
}
