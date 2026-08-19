//! The GDPSO entry points and the boundary conversion back to a `Partition`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::algorithms::gdpso::config::Config;
use crate::core::algorithms::gdpso::swarm::run;
use crate::core::graph::{
    CommunityId, CsrGraph, Graph, NodeId, Partition, normalize_community_ids,
};

// must stay in step with normalize_community_ids: dense ids in ascending node
// order, isolated nodes reported as -1
fn to_partition(g: &CsrGraph, labels: &[i32]) -> Partition {
    let mut remap: FxHashMap<i32, CommunityId> = FxHashMap::default();
    let mut partition: Partition = FxHashMap::default();
    let mut next = 0;
    for (dense, &degree) in g.deg.iter().enumerate() {
        let community = if degree == 0 {
            -1
        } else {
            *remap.entry(labels[dense]).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        };
        partition.insert(g.labels[dense], community);
    }
    partition
}

/// Run GDPSO over an explicit node list and edge list and return the highest
/// modularity partition the swarm ever held.
///
/// `nodes` pins the isolated vertices, which never appear in `edges`; every
/// operator leaves them alone and they come back as community `-1`.
#[allow(clippy::too_many_arguments)]
pub fn gdpso(
    nodes: &[NodeId],
    edges: &[(NodeId, NodeId)],
    pop_size: usize,
    num_gens: usize,
    w: f64,
    c1: f64,
    c2: f64,
    mut_rate: f64,
    mut_frac: f64,
    lpa_sweeps: usize,
) -> Partition {
    gdpso_with(
        nodes,
        edges,
        &Config {
            pop_size,
            num_gens,
            w,
            c1,
            c2,
            mut_rate,
            mut_frac,
            lpa_sweeps,
            ..Config::default()
        },
    )
}

/// GDPSO with the full configuration, including the two ablation switches that
/// are not part of the method (`mut_by_index`, `const_move_prob`).
pub fn gdpso_with(nodes: &[NodeId], edges: &[(NodeId, NodeId)], cfg: &Config) -> Partition {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Partition::default();
    }
    to_partition(&g, &run(&g, cfg))
}

/// GDPSO on an already-built `Graph`; the search still runs on a CSR copy.
#[allow(clippy::too_many_arguments)]
pub fn gdpso_on_graph(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    w: f64,
    c1: f64,
    c2: f64,
    mut_rate: f64,
    mut_frac: f64,
    lpa_sweeps: usize,
) -> Partition {
    let g = CsrGraph::from_edges(graph.nodes_vec(), &graph.edges);
    if g.n == 0 {
        return Partition::default();
    }
    let labels = run(
        &g,
        &Config {
            pop_size,
            num_gens,
            w,
            c1,
            c2,
            mut_rate,
            mut_frac,
            lpa_sweeps,
            ..Config::default()
        },
    );
    let partition: Partition = (0..g.n).map(|d| (g.labels[d], labels[d])).collect();
    normalize_community_ids(graph, partition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::gdpso::config::defaults::{
        DEFAULT_C1, DEFAULT_C2, DEFAULT_LPA_SWEEPS, DEFAULT_MUT_FRAC, DEFAULT_MUT_RATE,
        DEFAULT_POP_SIZE, DEFAULT_W,
    };
    use crate::core::algorithms::gdpso::objective::{degree_sums, modularity};
    use rustc_hash::FxHashSet;

    fn two_cliques() -> (Vec<NodeId>, Vec<(NodeId, NodeId)>) {
        let mut edges = vec![(4, 5)];
        for base in [0, 5] {
            for a in base..base + 5 {
                for b in a + 1..base + 5 {
                    edges.push((a, b));
                }
            }
        }
        ((0..10).collect(), edges)
    }

    fn run_small(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
        pop: usize,
        gens: usize,
    ) -> Partition {
        gdpso_with(
            nodes,
            edges,
            &Config {
                pop_size: pop,
                num_gens: gens,
                ..Config::default()
            },
        )
    }

    fn quality(nodes: &[NodeId], edges: &[(NodeId, NodeId)], part: &Partition) -> f64 {
        let g = CsrGraph::from_edges(nodes, edges);
        let labels: Vec<i32> = (0..g.n).map(|d| part[&g.labels[d]]).collect();
        let mut sigma = vec![0i64; g.n + 1];
        // -1 would index out of range, so isolated nodes take the top slot
        let labels: Vec<i32> = labels
            .iter()
            .map(|&c| if c < 0 { g.n as i32 } else { c })
            .collect();
        degree_sums(&g, &labels, &mut sigma);
        modularity(&g, &labels, &sigma)
    }

    #[test]
    fn two_cliques_split_exactly() {
        let (nodes, edges) = two_cliques();
        let part = run_small(&nodes, &edges, 30, 30);
        for i in 1..5 {
            assert_eq!(part[&0], part[&i], "clique A lost node {i}");
        }
        for i in 6..10 {
            assert_eq!(part[&5], part[&i], "clique B lost node {i}");
        }
        assert_ne!(part[&0], part[&5], "the cliques merged");
    }

    #[test]
    fn two_runs_are_byte_identical() {
        let (nodes, edges) = two_cliques();
        let a = run_small(&nodes, &edges, 20, 15);
        let b = run_small(&nodes, &edges, 20, 15);
        assert_eq!(a, b);
    }

    #[test]
    fn karate_reaches_the_reference_modularity() {
        use crate::core::graph::karate::KARATE_EDGES;
        let nodes: Vec<NodeId> = (0..34).collect();
        // the reference reports max Q = 0.41979 over its 30 runs: four groups,
        // karate's modularity optimum and the resolution-limit answer
        let part = run_small(&nodes, &KARATE_EDGES, 300, 100);
        let q = quality(&nodes, &KARATE_EDGES, &part);
        assert!(q > 0.4197, "Q={q}");
        let k: FxHashSet<CommunityId> = part.values().copied().collect();
        assert_eq!(k.len(), 4, "communities={}", k.len());

        // the shipped 100 particles must land strictly short of that optimum:
        // five propagation sweeps leave only ~19 distinct seeds
        let small = run_small(&nodes, &KARATE_EDGES, DEFAULT_POP_SIZE, 100);
        let q_small = quality(&nodes, &KARATE_EDGES, &small);
        assert!(q_small > 0.40 && q_small < q, "Q={q_small}");
    }

    #[test]
    fn empty_graph_is_empty() {
        assert!(run_small(&[], &[], 10, 5).is_empty());
    }

    #[test]
    fn single_node_is_isolated() {
        let part = run_small(&[7], &[], 10, 5);
        assert_eq!(part[&7], -1);
    }

    #[test]
    fn a_lone_edge_is_one_community() {
        let part = run_small(&[0, 1], &[(0, 1)], 10, 5);
        assert_eq!(part[&0], part[&1]);
        assert_ne!(part[&0], -1);
    }

    #[test]
    fn edgeless_graph_is_all_minus_one() {
        let part = run_small(&[0, 1, 2], &[], 10, 5);
        assert!(part.values().all(|&c| c == -1));
    }

    #[test]
    fn isolated_nodes_become_minus_one() {
        let (mut nodes, edges) = two_cliques();
        nodes.push(20);
        nodes.push(21);
        let part = run_small(&nodes, &edges, 20, 10);
        assert_eq!(part[&20], -1);
        assert_eq!(part[&21], -1);
        assert_ne!(part[&0], -1);
    }

    #[test]
    fn disconnected_components_never_merge() {
        let nodes: Vec<NodeId> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let part = run_small(&nodes, &edges, 20, 15);
        assert_eq!(part[&0], part[&2]);
        assert_eq!(part[&3], part[&5]);
        assert_ne!(part[&0], part[&3]);
    }

    #[test]
    fn self_loops_do_not_panic() {
        let (nodes, edges) = two_cliques();
        let mut noisy = edges.clone();
        noisy.push((0, 0));
        noisy.push((5, 5));
        let clean = run_small(&nodes, &edges, 20, 10);
        assert_eq!(run_small(&nodes, &noisy, 20, 10), clean);
    }

    #[test]
    fn degenerate_parameters_do_not_panic() {
        let (nodes, edges) = two_cliques();
        assert_eq!(run_small(&nodes, &edges, 0, 0).len(), 10);
        assert_eq!(run_small(&nodes, &edges, 1, 0).len(), 10);
        assert_eq!(run_small(&nodes, &edges, 1, 5).len(), 10);
        let wild = gdpso(&nodes, &edges, 4, 3, f64::NAN, -2.0, f64::NAN, 5.0, -1.0, 0);
        assert_eq!(wild.len(), 10);
    }

    #[test]
    fn the_graph_entry_point_agrees_with_the_edge_list_one() {
        let (nodes, edges) = two_cliques();
        let mut graph = Graph::new();
        for &node in &nodes {
            graph.nodes.insert(node);
            graph.adjacency_list.entry(node).or_default();
        }
        for &(u, v) in &edges {
            graph.add_edge(u, v);
        }
        graph.finalize();

        let from_graph = gdpso_on_graph(
            &graph,
            20,
            10,
            DEFAULT_W,
            DEFAULT_C1,
            DEFAULT_C2,
            DEFAULT_MUT_RATE,
            DEFAULT_MUT_FRAC,
            DEFAULT_LPA_SWEEPS,
        );
        let from_edges = run_small(&nodes, &edges, 20, 10);
        // the two entry points number communities differently, so compare
        // co-clustering rather than ids
        for &a in &nodes {
            for &b in &nodes {
                assert_eq!(
                    from_graph[&a] == from_graph[&b],
                    from_edges[&a] == from_edges[&b],
                    "nodes {a} and {b} disagree"
                );
            }
        }
    }
}
