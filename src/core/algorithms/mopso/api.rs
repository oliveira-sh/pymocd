//! Public entry points for MOPSO, re-exported from the crate root.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::config::Cfg;
use super::front::{select_best, select_index};
use super::swarm::run;
use super::utils::to_output;

/// Runs the swarm and returns the selected partition; isolated nodes get community `-1`.
#[allow(clippy::too_many_arguments)]
pub fn mopso(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    seed_rounds: usize,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let cfg = Cfg::new(
        pop,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        seed_rounds,
    );
    let (front, objs) = run(&g, &cfg);
    let best = select_best(&g, front, &objs);
    to_output(&g, &best)
}

pub type Profile = (Vec<Vec<(i32, i32)>>, Vec<[f64; 2]>, usize); // partitions, their objective points, and the selected index.

/// Runs the swarm and returns the whole archive, the graph's resolution profile.
#[allow(clippy::too_many_arguments)]
pub fn mopso_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    seed_rounds: usize,
) -> Profile {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }
    let cfg = Cfg::new(
        pop,
        num_gens,
        inertia,
        cognitive,
        social,
        local_rate,
        archive,
        seed_rounds,
    );
    let (front, objs) = run(&g, &cfg);
    let selected = select_index(&g, &front, &objs);
    (
        front.iter().map(|l| to_output(&g, l)).collect(),
        objs,
        selected,
    )
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::core::algorithms::mopso::config::defaults::*;

    fn defaults() -> (usize, usize, f64, f64, f64, f64, usize, usize) {
        (
            DEFAULT_POP_SIZE,
            DEFAULT_NUM_GENS,
            DEFAULT_INERTIA,
            DEFAULT_COGNITIVE,
            DEFAULT_SOCIAL,
            DEFAULT_LOCAL_RATE,
            DEFAULT_POP_SIZE,
            DEFAULT_SEED_ROUNDS,
        )
    }

    fn small(nodes: &[i32], edges: &[(i32, i32)]) -> FxHashMap<i32, i32> {
        let (_, _, w, c1, c2, lr, _, sr) = defaults();
        mopso(nodes, edges, 32, 25, w, c1, c2, lr, 32, sr)
            .into_iter()
            .collect()
    }

    fn two_clique_edges() -> Vec<(i32, i32)> {
        let mut e = Vec::new();
        for (lo, hi) in [(0, 5), (5, 10)] {
            for a in lo..hi {
                for b in (a + 1)..hi {
                    e.push((a, b));
                }
            }
        }
        e.push((4, 5));
        e
    }

    fn ring_of_cliques(k: i32, s: i32) -> (Vec<i32>, Vec<(i32, i32)>) {
        let mut e = Vec::new();
        for c in 0..k {
            let lo = c * s;
            for a in lo..lo + s {
                for b in (a + 1)..lo + s {
                    e.push((a, b));
                }
            }
            e.push((lo + s - 1, (lo + s) % (k * s)));
        }
        ((0..k * s).collect(), e)
    }

    #[test]
    fn finds_the_two_community_split() {
        let c = small(&(0..10).collect::<Vec<i32>>(), &two_clique_edges());
        for i in 1..5 {
            assert_eq!(c[&0], c[&i], "clique A node {i} split off");
        }
        for i in 6..10 {
            assert_eq!(c[&5], c[&i], "clique B node {i} split off");
        }
        assert_ne!(c[&0], c[&5], "the cliques merged");
    }

    #[test]
    fn beats_the_resolution_limit_on_a_ring_of_cliques() {
        let (nodes, edges) = ring_of_cliques(30, 5);
        let c = small(&nodes, &edges);
        for r in 0..30i32 {
            let base = c[&(r * 5)];
            for j in 1..5 {
                assert_eq!(c[&(r * 5 + j)], base, "clique {r} was split");
            }
        }
        let mut ks: Vec<i32> = (0..150).map(|v| c[&v]).collect();
        ks.sort_unstable();
        ks.dedup();
        assert_eq!(ks.len(), 30, "the ring did not resolve into thirty cliques");
    }

    #[test]
    fn isolated_nodes_get_minus_one() {
        let nodes: Vec<i32> = (0..7).collect();
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        assert_eq!(small(&nodes, &edges)[&6], -1);
    }

    #[test]
    fn the_result_is_deterministic() {
        let (nodes, edges) = ring_of_cliques(8, 5);
        assert_eq!(small(&nodes, &edges), small(&nodes, &edges));
    }

    #[test]
    fn an_empty_graph_returns_nothing() {
        let (p, g, w, c1, c2, lr, a, sr) = defaults();
        assert!(mopso(&[], &[], p, g, w, c1, c2, lr, a, sr).is_empty());
    }

    #[test]
    fn the_front_is_a_profile_and_the_selection_indexes_into_it() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let (_, _, w, c1, c2, lr, _, sr) = defaults();
        let (front, objs, pick) = mopso_fronts(&nodes, &edges, 32, 25, w, c1, c2, lr, 32, sr);
        assert!(!front.is_empty());
        assert_eq!(front.len(), objs.len());
        assert!(pick < front.len());
        assert!(front.iter().all(|f| f.len() == nodes.len()));
        let cuts: Vec<f64> = objs.iter().map(|o| o[0]).collect();
        assert!(
            cuts.iter().copied().fold(f64::MIN, f64::max)
                > cuts.iter().copied().fold(f64::MAX, f64::min),
            "the archive collapsed to one point"
        );
    }
}
