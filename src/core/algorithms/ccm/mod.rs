//! NSGA-III-CCM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021): a
//! self-contained, single-threaded reimplementation of the paper's method —
//! locus genome (`locus.rs`), CCM's own NSGA-III loop with paper
//! customizations (`engine.rs`), (Community Score, Community Fitness,
//! Modularity) objectives, max-modularity rank-1 decision rule. Deliberately
//! avoids the shared NSGA-III engine and Rayon so this baseline's cost and
//! behaviour track the paper.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::normalize_community_ids;
use crate::core::graph::{Graph, Partition};
use objectives::{community_objectives, modularity_labels};

use std::cmp::Ordering;

mod defaults;
mod engine;
mod locus;
mod objectives;

pub use defaults::*;

/// (Community Score, Community Fitness, Modularity) are all maximized by the
/// paper; this engine's `Individual::dominates` assumes minimization, so the
/// objective vector fed to it is `(-CS, -CF, -Q)`.
fn evaluate(neighbor_pos: &[Vec<usize>], labels: &[i32], r: f64, alpha: f64) -> Vec<f64> {
    let (cs, cf) = community_objectives(neighbor_pos, labels, r, alpha);
    let q = modularity_labels(neighbor_pos, labels);
    vec![-cs, -cf, -q]
}

/// Run NSGA-III-CCM and return the **max-modularity** member of the rank-1
/// Pareto front (Shaik et al. recommend modularity-only decision-making when no
/// ground truth is available), normalized (isolated nodes → community `-1`).
#[allow(clippy::too_many_arguments)]
pub fn ccm(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> Partition {
    let nodes = graph.nodes_vec().clone();
    let neighbor_pos = locus::neighbor_positions(graph, &nodes);

    let mut pop = engine::evolve(
        &neighbor_pos,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        divisions,
        |labels| evaluate(&neighbor_pos, labels, r, alpha),
    );

    engine::fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (modularity_labels(&neighbor_pos, &ind.labels), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    let partition: Partition = nodes
        .iter()
        .enumerate()
        .map(|(p, &node)| (node, best.labels[p]))
        .collect();
    normalize_community_ids(graph, partition)
}

/// The rank-1 Pareto front `ccm` selects from, as normalized partitions —
/// the paper's Table 1/2 protocol (best-NMI / best-Q over the front) needs it.
#[allow(clippy::too_many_arguments)]
pub fn ccm_fronts(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    r: f64,
    alpha: f64,
    divisions: usize,
) -> Vec<Partition> {
    let nodes = graph.nodes_vec().clone();
    let neighbor_pos = locus::neighbor_positions(graph, &nodes);

    let mut pop = engine::evolve(
        &neighbor_pos,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        divisions,
        |labels| evaluate(&neighbor_pos, labels, r, alpha),
    );

    engine::fast_non_dominated_sort(&mut pop);
    pop.iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| {
            let partition: Partition = nodes
                .iter()
                .enumerate()
                .map(|(p, &node)| (node, ind.labels[p]))
                .collect();
            normalize_community_ids(graph, partition)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Triangle {0,1,2}, triangle {3,4,5}, single bridge edge (2,3).
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn finds_two_community_split() {
        let g = two_triangles();
        let res = ccm(&g, 100, 100, 0.8, 0.2, 1.0, 1.0, 12);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
