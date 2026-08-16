//! Public entry points for SMOCC: the Pareto-front search and the
//! single-partition wrappers re-exported from the crate root.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::config::defaults::{
    DEFAULT_MACRO_CAP, DEFAULT_MICRO_MUT, DEFAULT_OBJ_MODE, DEFAULT_TOPO_MODE,
};
use super::macro_micro::run_fronts;
use super::select::{select_best, to_output};

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn smocc(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
) -> Vec<(i32, i32)> {
    smocc_capped(
        nodes,
        edges,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        DEFAULT_MACRO_CAP,
        DEFAULT_MICRO_MUT,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn smocc_capped(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    macro_cap: f64,
    micro_mut: f64,
) -> Vec<(i32, i32)> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    let front = run_fronts(
        &g,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        true,
        DEFAULT_TOPO_MODE,
        DEFAULT_OBJ_MODE,
        macro_cap,
        micro_mut,
    );
    let best = select_best(&g, front);
    to_output(&g, &best)
}

#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn smocc_fronts(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
) -> Vec<Vec<(i32, i32)>> {
    smocc_fronts_capped(
        nodes,
        edges,
        pop,
        num_gens,
        cross_rate,
        mut_rate,
        gap,
        beta,
        refine,
        topo_mode,
        obj_mode,
        DEFAULT_MACRO_CAP,
        DEFAULT_MICRO_MUT,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn smocc_fronts_capped(
    nodes: &[i32],
    edges: &[(i32, i32)],
    pop: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    gap: usize,
    beta: f64,
    refine: bool,
    topo_mode: u8,
    obj_mode: u16,
    macro_cap: f64,
    micro_mut: f64,
) -> Vec<Vec<(i32, i32)>> {
    let g = CsrGraph::from_edges(nodes, edges);
    if g.n == 0 {
        return Vec::new();
    }
    run_fronts(
        &g, pop, num_gens, cross_rate, mut_rate, gap, beta, refine, topo_mode, obj_mode, macro_cap,
        micro_mut,
    )
    .iter()
    .map(|l| to_output(&g, l))
    .collect()
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::core::algorithms::smocc::config::defaults::*;
    use crate::core::algorithms::smocc::fixtures::two_clique_edges;
    use crate::core::algorithms::smocc::macro_micro::macro_cmax;

    fn two_triangle_edges() -> Vec<(i32, i32)> {
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    }

    fn ring_of_cliques(k: i32, s: i32) -> (Vec<i32>, Vec<(i32, i32)>) {
        let nodes: Vec<i32> = (0..k * s).collect();
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
        (nodes, e)
    }

    fn grid(w: i32, h: i32) -> (Vec<i32>, Vec<(i32, i32)>) {
        let nodes: Vec<i32> = (0..w * h).collect();
        let mut e = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let u = y * w + x;
                if x + 1 < w {
                    e.push((u, u + 1));
                }
                if y + 1 < h {
                    e.push((u, u + w));
                }
            }
        }
        (nodes, e)
    }

    #[test]
    fn finds_two_community_split() {
        let nodes: Vec<i32> = (0..10).collect();
        let out = smocc(
            &nodes,
            &two_clique_edges(),
            60,
            40,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: FxHashMap<i32, i32> = out.into_iter().collect();
        for i in 1..5 {
            assert_eq!(c[&0], c[&i], "clique A node {i} split off");
        }
        for i in 6..10 {
            assert_eq!(c[&5], c[&i], "clique B node {i} split off");
        }
        assert_ne!(c[&0], c[&5], "cliques merged");
    }

    #[test]
    fn isolated_node_gets_minus_one() {
        let nodes: Vec<i32> = (0..7).collect();
        let out = smocc(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        let c: FxHashMap<i32, i32> = out.into_iter().collect();
        assert_eq!(c[&6], -1);
    }

    #[test]
    fn fronts_are_nonempty() {
        let nodes: Vec<i32> = (0..6).collect();
        let fronts = smocc_fronts(
            &nodes,
            &two_triangle_edges(),
            40,
            20,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            true,
            0,
            0,
        );
        assert!(!fronts.is_empty());
        assert!(fronts.iter().all(|f| f.len() == 6));
    }

    #[test]
    fn fronts_are_deterministic() {
        let nodes: Vec<i32> = (0..6).collect();
        let run = || {
            smocc_fronts(
                &nodes,
                &two_triangle_edges(),
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                true,
                0,
                0,
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn heterogeneous_objective_modes_are_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [106u16, 160, 166, 100] {
            let run = || {
                smocc_fronts(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                )
            };
            let a = run();
            assert!(!a.is_empty(), "obj_mode {obj_mode} produced an empty front");
            assert!(a.iter().all(|f| f.len() == 10));
            assert_eq!(a, run(), "obj_mode {obj_mode} is not deterministic");
        }
    }

    #[test]
    fn heterogeneous_encoding_of_equal_sides_matches_homogeneous() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        let run = |obj_mode: u16| {
            smocc_fronts(
                &nodes,
                &edges,
                40,
                15,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                true,
                0,
                obj_mode,
            )
        };
        for v in 0..10u16 {
            assert_eq!(run(v), run(100 + v * 10 + v), "mode {v} != het({v},{v})");
        }
        assert_eq!(run(160), run(1600));
        assert_eq!(run(106), run(1006));
    }

    #[test]
    fn both_objective_sets_search_and_differ() {
        let (nodes, edges) = grid(10, 10);
        let n = nodes.len();
        let run = |obj_mode: u16| {
            smocc_fronts(
                &nodes,
                &edges,
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                5,
                DEFAULT_BETA,
                true,
                0,
                obj_mode,
            )
        };
        let kkm_rc = run(0);
        let intra_inter = run(6);
        for (label, a) in [("0", &kkm_rc), ("6", &intra_inter)] {
            assert!(!a.is_empty(), "obj_mode {label}: empty front");
            assert!(a.iter().all(|f| f.len() == n), "obj_mode {label}: partial");
        }
        assert_eq!(kkm_rc, run(0), "obj_mode 0 is not deterministic");
        assert_eq!(intra_inter, run(6), "obj_mode 6 is not deterministic");
        assert_ne!(
            kkm_rc, intra_inter,
            "the two objective sets are the same arm"
        );
        assert_ne!(run(160), kkm_rc);
        assert_ne!(run(160), intra_inter);
    }

    #[test]
    fn default_macro_cap_matches_the_unparameterized_wrapper() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            let legacy = || {
                smocc_fronts(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                )
            };
            let capped = |cap: f64| {
                smocc_fronts_capped(
                    &nodes,
                    &edges,
                    40,
                    15,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    DEFAULT_GAP,
                    DEFAULT_BETA,
                    true,
                    0,
                    obj_mode,
                    cap,
                    DEFAULT_MICRO_MUT,
                )
            };
            let a = capped(DEFAULT_MACRO_CAP);
            assert_eq!(
                a,
                capped(DEFAULT_MACRO_CAP),
                "obj_mode {obj_mode}: unstable"
            );
            assert_eq!(
                a,
                legacy(),
                "obj_mode {obj_mode}: default changed the front"
            );
            assert_eq!(legacy(), legacy());
        }
        let selected = smocc_capped(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
            DEFAULT_MACRO_CAP,
            DEFAULT_MICRO_MUT,
        );
        let legacy = smocc(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_CROSS_RATE,
            DEFAULT_MUT_RATE,
            DEFAULT_GAP,
            DEFAULT_BETA,
        );
        assert_eq!(
            selected, legacy,
            "smocc/smocc_capped diverged at the default"
        );
    }

    #[test]
    fn macro_cap_variants_deterministic_and_nonempty() {
        let nodes: Vec<i32> = (0..10).collect();
        let edges = two_clique_edges();
        for obj_mode in [0u16, 160] {
            for cap in [1.0f64, 2.0, 4.0] {
                let run = || {
                    smocc_fronts_capped(
                        &nodes,
                        &edges,
                        40,
                        15,
                        DEFAULT_CROSS_RATE,
                        DEFAULT_MUT_RATE,
                        DEFAULT_GAP,
                        DEFAULT_BETA,
                        true,
                        0,
                        obj_mode,
                        cap,
                        DEFAULT_MICRO_MUT,
                    )
                };
                let a = run();
                assert!(!a.is_empty(), "obj_mode {obj_mode} cap {cap}: empty front");
                assert!(
                    a.iter().all(|f| f.len() == 10),
                    "obj_mode {obj_mode} cap {cap}: partial partition"
                );
                assert_eq!(a, run(), "obj_mode {obj_mode} cap {cap}: nondeterministic");
            }
        }
    }

    #[test]
    fn macro_cap_changes_the_front() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        assert_eq!(macro_cmax(n, 1.0), 8);
        assert_eq!(macro_cmax(n, 4.0), 31);
        let run = |cap: f64| {
            smocc_fronts_capped(
                &nodes,
                &edges,
                40,
                15,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                DEFAULT_GAP,
                DEFAULT_BETA,
                true,
                0,
                0,
                cap,
                DEFAULT_MICRO_MUT,
            )
        };
        let one = run(1.0);
        let four = run(4.0);
        assert_eq!(one, run(1.0), "cap 1.0 nondeterministic");
        assert_eq!(four, run(4.0), "cap 4.0 nondeterministic");
        assert_ne!(one, four, "macro_cap is not wired through to init_macro");
    }

    #[test]
    fn every_topo_bit_is_deterministic_and_nonempty() {
        let (nodes, edges) = ring_of_cliques(12, 5);
        let n = nodes.len();
        for topo_mode in [0u8, 2, 128, 130] {
            let run = || {
                smocc_fronts_capped(
                    &nodes,
                    &edges,
                    40,
                    20,
                    DEFAULT_CROSS_RATE,
                    DEFAULT_MUT_RATE,
                    5,
                    DEFAULT_BETA,
                    true,
                    topo_mode,
                    0,
                    DEFAULT_MACRO_CAP,
                    DEFAULT_MICRO_MUT,
                )
            };
            let a = run();
            assert!(!a.is_empty(), "topo {topo_mode}: empty");
            assert!(
                a.iter().all(|f| f.len() == n),
                "topo {topo_mode}: partial partition"
            );
            assert_eq!(a, run(), "topo {topo_mode}: nondeterministic");
        }
    }

    #[test]
    fn only_the_two_live_topo_bits_change_the_front() {
        let (nodes, edges) = grid(10, 10);
        let run = |topo_mode: u8| {
            smocc_fronts_capped(
                &nodes,
                &edges,
                40,
                20,
                DEFAULT_CROSS_RATE,
                DEFAULT_MUT_RATE,
                5,
                DEFAULT_BETA,
                true,
                topo_mode,
                0,
                DEFAULT_MACRO_CAP,
                DEFAULT_MICRO_MUT,
            )
        };
        let base = run(0);
        for bit in [2u8, 128, 130] {
            assert_ne!(base, run(bit), "topo bit {bit} is inert end to end");
        }
        for dead in [1u8, 4, 8, 16, 32, 64] {
            assert_eq!(
                base,
                run(dead),
                "deleted topo bit {dead} still does something"
            );
            assert_eq!(
                run(130),
                run(130 | dead),
                "deleted topo bit {dead} changed the shipped mask"
            );
        }
    }
}
