//! The Constant Potts pair (cut, pair), whose Pareto front is the resolution ladder.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::smocc::Labels;
use crate::core::graph::CsrGraph;

use super::UNSEEN;

/// The Constant Potts pair, both minimised over the `n_a` non-isolated nodes:
/// `cut  = 1 - (internal edges)/m` and `pair = Σ_c C(n_c,2)/C(n_a,2)`, so that
/// `H_gamma(C)/m = 1 - cut - (gamma/gamma_d)·pair` with `gamma_d = 2m/(n_a(n_a-1))`.
pub fn cpm(g: &CsrGraph, labels: &Labels) -> (f64, f64) {
    let mut slot: Vec<u32> = vec![UNSEEN; g.n];
    let mut size: Vec<u32> = Vec::new();
    // n_a: the nodes carrying at least one edge, the scope of both objectives
    let mut n_active: usize = 0;

    for (v, &c) in labels.iter().enumerate().take(g.n) {
        debug_assert!((c as usize) < g.n, "label {c} outside [0,{})", g.n);
        // isolated nodes are skipped: output forces them to -1, so their label is meaningless
        if g.deg[v] == 0 {
            continue;
        }
        n_active += 1;
        let s = slot[c as usize];
        let b = if s == UNSEEN {
            let b = size.len() as u32;
            slot[c as usize] = b;
            size.push(0);
            b
        } else {
            s
        } as usize;
        size[b] += 1;
    }

    // g.edges holds every undirected edge exactly once, so one pass counts them all
    let mut intra: usize = 0;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            intra += 1;
        }
    }

    // integer counting with a single final division keeps the value exactly reproducible
    let cut = if g.m == 0 {
        0.0
    } else {
        1.0 - intra as f64 / g.m as f64
    };

    let co_clustered: f64 = size.iter().map(|&s| choose_two(s as usize)).sum();
    // C(n_a,2) is zero for n_a < 2, in which case no pair can be co-clustered
    let all_pairs = choose_two(n_active);
    let pair = if all_pairs == 0.0 {
        0.0
    } else {
        co_clustered / all_pairs
    };

    (cut, pair)
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
    use super::super::sets::{ObjSet, evaluate, split_mode};
    use super::*;
    use crate::core::algorithms::mopots::calculate_objectives;
    use crate::core::algorithms::smocc::utils::fixtures::two_triangles;
    use crate::core::graph::karate::{KARATE_CLUB, KARATE_EDGES};
    use crate::core::graph::{Graph, Partition};

    // one community per triangle, labelled by a member node id as smocc labels are
    fn two_triangles_split() -> Labels {
        vec![0, 0, 0, 3, 3, 3]
    }

    fn two_triangles_with_isolated(count: i32) -> CsrGraph {
        let nodes: Vec<i32> = (0..(6 + count)).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        CsrGraph::from_edges(&nodes, &edges)
    }

    #[test]
    fn one_community_leaves_no_edge_and_joins_every_pair() {
        let g = two_triangles();
        let one: Labels = vec![0; g.n];
        let (cut, pair) = cpm(&g, &one);
        assert_eq!(cut, 0.0, "no edge can leave a single community");
        assert_eq!(pair, 1.0, "every pair is co-clustered");
    }

    #[test]
    fn singletons_leave_every_edge_and_join_no_pair() {
        let g = two_triangles();
        let labels: Labels = (0..g.n as i32).collect();
        let (cut, pair) = cpm(&g, &labels);
        assert_eq!(cut, 1.0, "every edge leaves its community");
        assert_eq!(pair, 0.0, "singletons hold no pair");
    }

    #[test]
    fn two_triangles_match_the_hand_computation() {
        // n = 6, m = 7, 6 internal edges, C(3,2)+C(3,2) = 6 of the C(6,2) = 15 pairs
        let g = two_triangles();
        let (cut, pair) = cpm(&g, &two_triangles_split());
        assert!((cut - 1.0 / 7.0).abs() < 1e-12, "cut = 1 - 6/7, got {cut}");
        assert!((pair - 6.0 / 15.0).abs() < 1e-12, "pair = 6/15, got {pair}");
    }

    #[test]
    fn isolated_nodes_move_neither_objective() {
        let g = two_triangles();
        let before = cpm(&g, &two_triangles_split());

        let padded = two_triangles_with_isolated(4);
        // assorted labels: one per triangle, then a pair sharing a fresh community
        let mut labels = two_triangles_split();
        labels.extend_from_slice(&[0, 3, 8, 8]);
        assert_eq!(padded.n, 10, "the four extra nodes carry no edge");
        assert_eq!(cpm(&padded, &labels), before, "no edge and no pair");
    }

    #[test]
    fn an_edgeless_graph_divides_by_neither_denominator() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        let one: Labels = vec![0; g.n];
        assert_eq!(cpm(&g, &one), (0.0, 0.0), "no edges, no active pairs");
    }

    // the same node and edge list as a hashmap graph, so both sides read one input
    fn adj_graph(nodes: &[i32], edges: &[(i32, i32)]) -> Graph {
        let mut g = Graph::new();
        for &v in nodes {
            g.nodes.insert(v);
        }
        for &(u, v) in edges {
            g.add_edge(u, v);
        }
        g.finalize();
        g
    }

    // smocc labels are indexed by dense id, mopots keys by the original node id
    fn as_partition(g: &CsrGraph, labels: &Labels) -> Partition {
        labels
            .iter()
            .enumerate()
            .map(|(v, &c)| (g.labels[v], c))
            .collect()
    }

    // the two independent implementations must agree bit for bit on (cut, pair)
    fn assert_agrees(case: &str, nodes: &[i32], edges: &[(i32, i32)], labels: &Labels) {
        let csr = CsrGraph::from_edges(nodes, edges);
        assert_eq!(csr.n, labels.len(), "{case}: one label per dense node");
        let mine = cpm(&csr, labels);
        let theirs =
            calculate_objectives(&adj_graph(nodes, edges), &as_partition(&csr, labels), false);
        assert_eq!(
            mine,
            (theirs.cut, theirs.pair),
            "{case}: smocc {mine:?} != mopots ({}, {})",
            theirs.cut,
            theirs.pair
        );
    }

    #[test]
    fn smocc_and_mopots_read_the_same_cut_and_pair() {
        let six: Vec<i32> = (0..6).collect();
        let ten: Vec<i32> = (0..10).collect();
        let three: Vec<i32> = (0..3).collect();
        let karate: Vec<i32> = (0..34).collect();
        let triangles = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        let karate_edges: Vec<(i32, i32)> = KARATE_EDGES.to_vec();
        let truth: Labels = KARATE_CLUB.to_vec();

        // an isolated node keeps an assorted label: shared, private, or paired off
        assert_agrees("triangles/split", &six, &triangles, &vec![0, 0, 0, 3, 3, 3]);
        assert_agrees("triangles/one", &six, &triangles, &vec![0; 6]);
        assert_agrees("triangles/alone", &six, &triangles, &(0..6).collect());
        assert_agrees("triangles/mixed", &six, &triangles, &vec![0, 0, 3, 3, 5, 5]);
        let padded: Labels = vec![0, 0, 0, 3, 3, 3, 0, 3, 8, 8];
        assert_agrees("isolated/padded", &ten, &triangles, &padded);
        assert_agrees("isolated/alone", &ten, &triangles, &(0..10).collect());
        // an empty edge set leaves every node isolated, so both denominators vanish
        assert_agrees("edgeless/one", &three, &[], &vec![0; 3]);
        assert_agrees("edgeless/alone", &three, &[], &vec![0, 1, 2]);
        assert_agrees("empty", &[], &[], &Vec::new());
        assert_agrees("karate/truth", &karate, &karate_edges, &truth);
        assert_agrees("karate/one", &karate, &karate_edges, &vec![0; 34]);
        assert_agrees("karate/alone", &karate, &karate_edges, &(0..34).collect());
        let blocks: Labels = (0..34).map(|v| v % 5).collect();
        assert_agrees("karate/blocks", &karate, &karate_edges, &blocks);
    }

    #[test]
    fn the_dispatch_reaches_cpm_and_the_modes_decode_to_it() {
        assert_eq!(ObjSet::from_u8(20), ObjSet::Cpm);
        // homogeneous 20, then micro/macro pairs through the two-digit >= 1000 branch
        assert_eq!(split_mode(20), (ObjSet::Cpm, ObjSet::Cpm));
        assert_eq!(split_mode(3020), (ObjSet::Cpm, ObjSet::Cpm));
        assert_eq!(split_mode(3000), (ObjSet::Cpm, ObjSet::KkmRc));
        assert_eq!(split_mode(1020), (ObjSet::KkmRc, ObjSet::Cpm));
        assert_eq!(split_mode(3006), (ObjSet::Cpm, ObjSet::HpIntraInter));
        assert_eq!(split_mode(1620), (ObjSet::HpIntraInter, ObjSet::Cpm));

        let g = two_triangles();
        let labels = two_triangles_split();
        let (cut, pair) = cpm(&g, &labels);
        assert_eq!(evaluate(&g, &labels, ObjSet::Cpm), vec![cut, pair]);
        assert_ne!(
            evaluate(&g, &labels, ObjSet::Cpm),
            evaluate(&g, &labels, ObjSet::KkmRc),
            "a missing from_u8 arm must not pass silently"
        );
    }
}
