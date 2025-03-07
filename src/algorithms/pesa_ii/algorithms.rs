const NUM_RANDOM_NETWORKS: usize = 0;
const RANDOM_NETWORKS_MAX_GENS: usize = 20;

use crate::graph::{self, Graph, Partition};
use crate::utils::args::AGArgs;
use super::evolutionary::evolutionary_phase;
use super::hypergrid::Solution;
use super::model_selection::{max_q_selection, min_max_selection};

use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use rand::rng;
use rand::seq::SliceRandom as _;

/// Generates multiple random networks and combines their solutions
fn generate_random_networks(original: &Graph, num_networks: usize) -> Vec<Graph> {
    (0..num_networks)
        .map(|_| {
            let mut random_graph = graph::Graph {
                nodes: original.nodes.clone(),
                ..Default::default()
            };

            let node_vec: Vec<_> = random_graph.nodes.iter().cloned().collect();
            let num_nodes = node_vec.len();
            let num_edges = original.edges.len();
            let mut rng = rng();
            let mut possible_pairs = Vec::with_capacity(num_nodes * (num_nodes - 1) / 2);

            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    possible_pairs.push((node_vec[i], node_vec[j]));
                }
            }

            possible_pairs.shuffle(&mut rng);
            let selected_edges = possible_pairs
                .into_iter()
                .take(num_edges)
                .collect::<Vec<_>>();

            for (src, dst) in &selected_edges {
                random_graph.edges.push((*src, *dst));
            }

            for node in &random_graph.nodes {
                random_graph.adjacency_list.insert(*node, Vec::new());
            }

            for (src, dst) in &random_graph.edges {
                random_graph.adjacency_list.get_mut(src).unwrap().push(*dst);
                random_graph.adjacency_list.get_mut(dst).unwrap().push(*src);
            }

            random_graph
        })
        .collect()
}

pub fn pesa_ii_maxq(graph: &Graph, args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Evolutionary algorithm returns the Pareto frontier
    let (archive, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Phase 2: Selection Model> max q 
    let best_solution = max_q_selection(&archive);

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}

pub fn pesa_ii_minimax(graph: &Graph, mut args: AGArgs) -> (Partition, Vec<f64>, f64) {
    let degrees: HashMap<i32, usize, FxBuildHasher> = graph.precompute_degress();

    // Phase 1: Evolutionary algorithm returns the Pareto frontier 
    let (archive, best_fitness_history) = evolutionary_phase(graph, &args, &degrees);

    // Phase 2: Selection Model: mini-max selection
    let best_solution = {
        args.num_gens = RANDOM_NETWORKS_MAX_GENS;
        let random_networks = generate_random_networks(graph, NUM_RANDOM_NETWORKS);
        let random_archives: Vec<Vec<Solution>> = random_networks
            .iter()
            .map(|random_graph| {
                let random_degrees = random_graph.precompute_degress();
                let (random_archive, _) = evolutionary_phase(random_graph, &args, &random_degrees);
                random_archive
            })
            .collect();

        // Use Min-Max selection with random archives
        min_max_selection(&archive, &random_archives)
    };

    (
        best_solution.partition.clone(),
        best_fitness_history,
        best_solution.objectives[0],
    )
}
