//! The CDRME entry points and the boundary conversion back to a `Partition`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::chromosome::{NO_COMMUNITY, Relation};
use super::config::Config;
use super::config::defaults::MERGE_ATTEMPTS_PER_COMMUNITY;
use super::evolve::{Candidate, best, diversify, modularity, mutate};
use super::objective::objective;
use super::primary::compose;
use super::topology::Topology;
use crate::core::graph::{CommunityId, Graph, NodeId, Partition, normalize_community_ids};

// dense ids handed out in ascending node order; isolated nodes report -1
fn to_partition(topology: &Topology, labels: &[u32]) -> Partition {
    let mut remap: FxHashMap<u32, CommunityId> = FxHashMap::default();
    let mut partition = Partition::default();
    let mut next = 0;
    for (dense, &label) in labels.iter().enumerate() {
        let community = if label == NO_COMMUNITY {
            -1
        } else {
            *remap.entry(label).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        };
        partition.insert(topology.labels[dense], community);
    }
    partition
}

/// Run CDRME over an explicit node list and edge list.
///
/// `nodes` is what pins the isolated vertices, which never appear in `edges`;
/// they are outside every equation of the paper and come back as community
/// `-1`. Self-loops and repeated edges are dropped, so `|E|` and every degree
/// are those of the simple graph.
///
/// `alpha_walk` is Eq. (7)'s length coefficient and `alpha_mut` the Sec. 4.4.2
/// mutation threshold; the paper calls both `alpha`. `pop_size` is `N_p` and
/// `elite_size` is `N_sp`. Eq. (12) is one maximised scalar, so there is no
/// Pareto front to return.
#[allow(clippy::too_many_arguments)]
pub fn cdrme(
    nodes: &[NodeId],
    edges: &[(NodeId, NodeId)],
    alpha_walk: f64,
    n_walk: usize,
    pop_size: usize,
    elite_size: usize,
    alpha_mut: f64,
    mut_sweeps: usize,
) -> Partition {
    cdrme_with(
        nodes,
        edges,
        &Config {
            alpha_walk,
            n_walk,
            pop_size,
            elite_size,
            alpha_mut,
            mut_sweeps,
        },
    )
}

/// CDRME with the full configuration block.
pub fn cdrme_with(nodes: &[NodeId], edges: &[(NodeId, NodeId)], cfg: &Config) -> Partition {
    let topology = Topology::from_edges(nodes, edges);
    if topology.n == 0 {
        return Partition::default();
    }
    let labels = search(&topology, cfg);
    to_partition(&topology, &labels)
}

/// CDRME on an already-built `Graph`, the form the hashmap side of the crate
/// carries graphs in.
pub fn cdrme_on_graph(graph: &Graph, cfg: &Config) -> Partition {
    let topology = Topology::from_edges(graph.nodes_vec(), &graph.edges);
    if topology.n == 0 {
        return Partition::default();
    }
    let labels = search(&topology, cfg);
    let partition: Partition = (0..topology.n)
        .map(|d| {
            let community = if labels[d] == NO_COMMUNITY {
                -1
            } else {
                labels[d] as CommunityId
            };
            (topology.labels[d], community)
        })
        .collect();
    normalize_community_ids(graph, partition)
}

// the four components in order: pre-processing and primary composition, then
// one merge chain per population slot, then mutation of the N_sp best, then
// Sec. 4.4.4's selection
fn search(topology: &Topology, cfg: &Config) -> Vec<u32> {
    let cfg = cfg.sanitised();
    if topology.active.is_empty() {
        return vec![NO_COMMUNITY; topology.n];
    }

    let primary = compose(topology, &cfg);
    let (base, k) = primary.communities(topology);

    // Sec. 4.3.2's uniform draw is stratified over the slots instead, so the
    // merge ladder is covered for every k (README, "Divergences").
    let budget = MERGE_ATTEMPTS_PER_COMMUNITY * k;
    let mut chains: Vec<_> = (0..cfg.pop_size)
        .into_par_iter()
        .map(|slot| diversify(topology, &base, k, slot, budget * slot / cfg.pop_size))
        .collect();
    chains.sort_by(|a, b| {
        b.objective
            .total_cmp(&a.objective)
            .then_with(|| a.k.cmp(&b.k))
            .then_with(|| a.labels.cmp(&b.labels))
    });
    chains.truncate(cfg.elite_size);

    let candidates: Vec<Candidate> = chains
        .into_par_iter()
        .map(|chain| {
            let mut labels = chain.labels;
            let mut sim = primary.sim.clone();
            mutate(
                topology,
                &mut labels,
                &mut sim,
                chain.k,
                cfg.alpha_mut,
                cfg.mut_sweeps,
            );
            let quality = modularity(topology, &labels, chain.k);
            let objective = objective(&Relation::from_labels(topology, &labels, chain.k));
            let k = distinct(&labels);
            Candidate {
                labels,
                k,
                quality,
                objective,
            }
        })
        .collect();

    best(candidates).unwrap_or(base)
}

fn distinct(labels: &[u32]) -> usize {
    let mut seen: Vec<u32> = labels.iter().copied().filter(|&c| c != NO_COMMUNITY).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;

    fn run(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> Partition {
        cdrme_with(
            nodes,
            edges,
            &Config {
                pop_size: 20,
                elite_size: 10,
                n_walk: 20,
                ..Config::default()
            },
        )
    }

    // two 5-cliques joined by the single edge (4,5)
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

    #[test]
    fn two_cliques_split_exactly() {
        let (nodes, edges) = two_cliques();
        let part = run(&nodes, &edges);
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
        let a = run(&nodes, &edges);
        let b = run(&nodes, &edges);
        assert_eq!(a, b);
        let full = cdrme(&nodes, &edges, 1.0, 20, 8, 4, 0.5, 10);
        assert_eq!(full, cdrme(&nodes, &edges, 1.0, 20, 8, 4, 0.5, 10));
    }

    #[test]
    fn the_thread_count_does_not_change_the_result() {
        let (nodes, edges) = two_cliques();
        let with = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run(&nodes, &edges))
        };
        assert_eq!(with(1), with(4));
        assert_eq!(with(1), with(7));
    }

    #[test]
    fn the_empty_graph_is_empty() {
        assert!(run(&[], &[]).is_empty());
    }

    #[test]
    fn a_single_node_is_isolated() {
        assert_eq!(run(&[7], &[])[&7], -1);
    }

    #[test]
    fn an_edgeless_graph_is_all_minus_one() {
        let part = run(&[0, 1, 2], &[]);
        assert!(part.values().all(|&c| c == -1));
    }

    #[test]
    fn a_lone_edge_is_one_community() {
        let part = run(&[0, 1], &[(0, 1)]);
        assert_eq!(part[&0], part[&1]);
        assert_ne!(part[&0], -1);
    }

    #[test]
    fn isolated_nodes_become_minus_one() {
        let (mut nodes, edges) = two_cliques();
        nodes.extend([20, 21]);
        let part = run(&nodes, &edges);
        assert_eq!(part[&20], -1);
        assert_eq!(part[&21], -1);
        assert_ne!(part[&0], -1);
    }

    #[test]
    fn disconnected_components_never_merge() {
        let nodes: Vec<NodeId> = (0..6).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let part = run(&nodes, &edges);
        assert_eq!(part[&0], part[&2]);
        assert_eq!(part[&3], part[&5]);
        assert_ne!(part[&0], part[&3]);
    }

    #[test]
    fn self_loops_change_nothing() {
        let (nodes, edges) = two_cliques();
        let mut noisy = edges.clone();
        noisy.extend([(0, 0), (5, 5)]);
        assert_eq!(run(&nodes, &noisy), run(&nodes, &edges));
    }

    #[test]
    fn a_complete_graph_is_one_community() {
        let nodes: Vec<NodeId> = (0..12).collect();
        let mut edges = Vec::new();
        for a in 0..12 {
            for b in a + 1..12 {
                edges.push((a, b));
            }
        }
        let part = run(&nodes, &edges);
        let communities: FxHashSet<CommunityId> = part.values().copied().collect();
        assert_eq!(communities.len(), 1, "a clique was cut into {communities:?}");
    }

    #[test]
    fn a_star_does_not_panic_and_keeps_every_node() {
        let nodes: Vec<NodeId> = (0..20).collect();
        let edges: Vec<(NodeId, NodeId)> = (1..20).map(|v| (0, v)).collect();
        let part = run(&nodes, &edges);
        assert_eq!(part.len(), 20);
        assert!(part.values().all(|&c| c >= 0));
    }

    #[test]
    fn degenerate_parameters_do_not_panic() {
        let (nodes, edges) = two_cliques();
        assert_eq!(cdrme(&nodes, &edges, 1.0, 0, 0, 0, 0.5, 0).len(), 10);
        assert_eq!(
            cdrme(&nodes, &edges, f64::NAN, 1, 1, 99, f64::NAN, 1).len(),
            10
        );
        assert_eq!(cdrme(&nodes, &edges, -5.0, 1, 2, 1, 9.0, 3).len(), 10);
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

        let cfg = Config {
            pop_size: 20,
            elite_size: 10,
            n_walk: 20,
            ..Config::default()
        };
        let from_graph = cdrme_on_graph(&graph, &cfg);
        let from_edges = run(&nodes, &edges);
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

    #[test]
    fn karate_recovers_the_club_split() {
        use crate::core::graph::karate::{KARATE_CLUB, KARATE_EDGES};
        use crate::core::metrics::gt_metrics;

        let nodes: Vec<NodeId> = (0..34).collect();
        let part = cdrme_with(&nodes, &KARATE_EDGES, &Config::default());
        let found: Vec<i64> = nodes.iter().map(|n| i64::from(part[n])).collect();
        let truth: Vec<i64> = KARATE_CLUB.iter().map(|&c| i64::from(c)).collect();
        let (nmi, _, _, _) = gt_metrics(&truth, &found);
        assert!(nmi > 0.6, "karate NMI = {nmi}");
    }
}
