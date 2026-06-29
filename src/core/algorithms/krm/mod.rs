//! NSGA-III-KRM (Shaik, Ravi & Deb, SN Computer Science 2:13, 2021): NSGA-III
//! over (Kernel-K-Means, Ratio-Cut, Modularity) — KKM & RC minimized, Modularity
//! maximized. Only the objective wiring is new — the `kkm_ratiocut` objective,
//! the locus-based operators and the `core::metaheuristics::nsga3` engine are
//! reused unchanged.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::objectives::kernel_ratiocut::kkm_ratiocut;
use crate::core::metaheuristics::helpers::operators::get_modularity_from_partition;
use crate::core::metaheuristics::helpers::individual::{
    Individual, TOURNAMENT_SIZE, fast_non_dominated_sort,
};
use crate::core::metaheuristics::nsga3;
use crate::core::utils::normalize_community_ids;

use rayon::prelude::*;
use std::cmp::Ordering;
use std::convert::Infallible;

mod defaults;
pub use defaults::*;

fn evaluate(graph: &Graph, pop: &mut [Individual]) {
    pop.par_iter_mut().for_each(|ind| {
        let (kkm, rc) = kkm_ratiocut(graph, &ind.partition);
        let q = get_modularity_from_partition(&ind.partition, graph);
        // KKM & RC are minimized (fed as-is); Q is maximized → feed negated.
        ind.objectives = vec![kkm, rc, -q];
    });
}

/// Run NSGA-III-KRM and return the **max-modularity** member of the rank-1
/// Pareto front (Shaik et al. recommend modularity-only decision-making when no
/// ground truth is available), normalized (isolated nodes → community `-1`).
pub fn krm(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    divisions: usize,
) -> Partition {
    let mut pop = nsga3::evolve(
        graph,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        TOURNAMENT_SIZE,
        divisions,
        |inds| {
            evaluate(graph, inds);
            Ok::<(), Infallible>(())
        },
        |_, _, _| Ok(()),
    )
    .expect("nsga3::evolve is infallible for KRM");

    fast_non_dominated_sort(&mut pop);
    let best = pop
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| (get_modularity_from_partition(&ind.partition, graph), ind))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .expect("empty Pareto front")
        .1;

    normalize_community_ids(graph, best.partition.clone())
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
        let res = krm(&g, 100, 100, 0.8, 0.2, 12);
        assert_eq!(res[&0], res[&1]);
        assert_eq!(res[&1], res[&2]);
        assert_eq!(res[&3], res[&4]);
        assert_eq!(res[&4], res[&5]);
        assert_ne!(res[&0], res[&3]);
    }
}
