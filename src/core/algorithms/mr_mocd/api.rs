//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::super::mr_mocd::Labels;
use super::config::Cfg;
use super::front::{select_best, select_index};
use super::objectives::{measure, obj_of};
use super::swarm::run;
use super::utils::to_output;

#[allow(clippy::too_many_arguments)]
pub fn mr_mocd(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    ls_period: usize,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let mut cfg = Cfg::new(
        pop, num_gens, inertia, cognitive, social, local_rate, archive,
    );
    cfg.ls_period = ls_period;
    let (front, objs) = run(&g, &cfg);
    let best = select_best(&g, front, &objs);
    to_output(&g, &best)
}

pub type Profile = (Vec<Vec<(i32, i32)>>, Vec<[f64; 2]>, usize);

#[allow(clippy::too_many_arguments)]
pub fn mr_mocd_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    local_rate: f64,
    archive: usize,
    ls_period: usize,
) -> Profile {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }
    let mut cfg = Cfg::new(
        pop, num_gens, inertia, cognitive, social, local_rate, archive,
    );
    cfg.ls_period = ls_period;
    let (front, objs) = run(&g, &cfg);
    let selected = select_index(&g, &front, &objs);
    (
        front.iter().map(|l| to_output(&g, l)).collect(),
        objs,
        selected,
    )
}

pub fn mr_mocd_select(
    nodes: &[i32],
    edges: &[(i32, i32)],
    candidates: &[Vec<(i32, i32)>],
) -> (usize, Vec<[f64; 2]>) {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 || candidates.is_empty() {
        return (0, Vec::new());
    }
    let idx: std::collections::HashMap<i32, usize> =
        nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let front: Vec<Labels> = candidates
        .iter()
        .map(|part| {
            let mut lab: Labels = (0..g.n as i32).collect();
            for &(node, comm) in part {
                if let Some(&i) = idx.get(&node) {
                    lab[i] = comm;
                }
            }
            let mut seen: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
            let mut next = 0i32;
            for c in &mut lab {
                let e = seen.entry(*c).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                });
                *c = *e;
            }
            lab
        })
        .collect();
    let mut size = vec![0u32; g.n];
    let mut live = Vec::new();
    let objs: Vec<[f64; 2]> = front
        .iter()
        .map(|p| obj_of(&g, measure(&g, p, &mut size, &mut live)))
        .collect();
    (select_index(&g, &front, &objs), objs)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::core::algorithms::mr_mocd::config::defaults::*;

    fn small(nodes: &[i32], edges: &[(i32, i32)]) -> FxHashMap<i32, i32> {
        mr_mocd(
            nodes,
            edges,
            32,
            25,
            DEFAULT_INERTIA,
            DEFAULT_COGNITIVE,
            DEFAULT_SOCIAL,
            DEFAULT_LOCAL_RATE,
            32,
            DEFAULT_LS_PERIOD,
        )
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
    fn the_result_is_deterministic() {
        let (nodes, edges) = ring_of_cliques(8, 5);
        assert_eq!(small(&nodes, &edges), small(&nodes, &edges));
    }

    #[test]
    fn an_empty_graph_returns_nothing() {
        assert!(
            mr_mocd(
                &[],
                &[],
                DEFAULT_POP_SIZE,
                DEFAULT_NUM_GENS,
                DEFAULT_INERTIA,
                DEFAULT_COGNITIVE,
                DEFAULT_SOCIAL,
                DEFAULT_LOCAL_RATE,
                DEFAULT_POP_SIZE,
                DEFAULT_LS_PERIOD
            )
            .is_empty()
        );
    }

    #[test]
    fn the_front_is_a_profile_and_the_selection_indexes_into_it() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let (front, objs, pick) = mr_mocd_fronts(
            &nodes,
            &edges,
            32,
            25,
            DEFAULT_INERTIA,
            DEFAULT_COGNITIVE,
            DEFAULT_SOCIAL,
            DEFAULT_LOCAL_RATE,
            32,
            DEFAULT_LS_PERIOD,
        );
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
