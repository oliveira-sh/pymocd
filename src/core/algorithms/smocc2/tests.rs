//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use super::mopso::init::macro_cmax;
use crate::core::algorithms::smocc2::mopso::pareto::{Obj, dominates};
use crate::core::algorithms::smocc2::sim::init_weights;
use crate::core::graph::CsrGraph;

use super::api::{smocc2, smocc2_fronts};
use super::config::defaults::*;
use super::config::objectives::Cfg;
use super::config::schedule::inertia;
use super::mopso::archive::{arch_crowd, update_macro_archive, update_micro_archive};
use super::mopso::init::init_macro_swarm;
use super::mopso::particles::{MacElite, MicElite};
use super::mopso::steps::macro_step;

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

#[test]
fn finds_two_community_split() {
    let nodes: Vec<i32> = (0..10).collect();
    let out = smocc2(
        &nodes,
        &two_clique_edges(),
        60,
        40,
        DEFAULT_GAP,
        DEFAULT_TURB,
        DEFAULT_MACRO_CAP,
        false,
    )
    .unwrap();
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
    let out = smocc2(
        &nodes,
        &two_triangle_edges(),
        40,
        20,
        DEFAULT_GAP,
        DEFAULT_TURB,
        DEFAULT_MACRO_CAP,
        false,
    )
    .unwrap();
    let c: FxHashMap<i32, i32> = out.into_iter().collect();
    assert_eq!(c[&6], -1);
}

#[test]
fn fronts_are_nonempty() {
    let nodes: Vec<i32> = (0..6).collect();
    let fronts = smocc2_fronts(
        &nodes,
        &two_triangle_edges(),
        40,
        20,
        DEFAULT_GAP,
        DEFAULT_TURB,
        true,
        0,
        DEFAULT_MACRO_CAP,
        false,
    )
    .unwrap();
    assert!(!fronts.is_empty());
    assert!(fronts.iter().all(|f| f.len() == 6));
}

#[test]
fn objective_modes_produce_full_partitions() {
    let nodes: Vec<i32> = (0..10).collect();
    let edges = two_clique_edges();
    for obj_mode in [106u16, 160, 166, 100, 0, 6] {
        let a = smocc2_fronts(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_GAP,
            DEFAULT_TURB,
            true,
            obj_mode,
            DEFAULT_MACRO_CAP,
            false,
        )
        .unwrap();
        assert!(!a.is_empty(), "obj_mode {obj_mode} produced an empty front");
        assert!(a.iter().all(|f| f.len() == 10));
    }
}

#[test]
fn gpu_fronts_are_valid_and_find_the_split() {
    let nodes: Vec<i32> = (0..10).collect();
    let edges = two_clique_edges();
    let a = match smocc2_fronts(
        &nodes,
        &edges,
        40,
        20,
        DEFAULT_GAP,
        DEFAULT_TURB,
        true,
        DEFAULT_OBJ_MODE,
        DEFAULT_MACRO_CAP,
        true,
    ) {
        Err(e) => {
            eprintln!("skipping GPU test (no usable CUDA device): {e}");
            return;
        }
        Ok(a) => a,
    };
    assert!(!a.is_empty());
    assert!(a.iter().all(|f| f.len() == 10));

    let out = smocc2(
        &nodes,
        &edges,
        60,
        40,
        DEFAULT_GAP,
        DEFAULT_TURB,
        DEFAULT_MACRO_CAP,
        true,
    )
    .unwrap();
    let c: FxHashMap<i32, i32> = out.into_iter().collect();
    for i in 1..5 {
        assert_eq!(c[&0], c[&i], "gpu: clique A node {i} split off");
    }
    for i in 6..10 {
        assert_eq!(c[&5], c[&i], "gpu: clique B node {i} split off");
    }
    assert_ne!(c[&0], c[&5], "gpu: cliques merged");
}

#[test]
fn macro_cardinality_invariant_survives_updates() {
    let (nodes, edges) = ring_of_cliques(8, 5);
    let g = CsrGraph::from_edges(&nodes, &edges);
    let wadj = init_weights(&g);
    let cfg = Cfg::new(0);
    let pop = 12usize;
    let cmax = macro_cmax(g.n, DEFAULT_MACRO_CAP);

    let mut parts = init_macro_swarm(&g, &wadj, pop, &cfg, DEFAULT_MACRO_CAP, None);
    let cards: Vec<usize> = parts
        .iter()
        .map(|p| p.genome.iter().filter(|&&b| b != 0).count())
        .collect();
    for &c in &cards {
        assert!((1..=cmax).contains(&c), "init cardinality {c} out of range");
    }

    let mut arch = update_macro_archive(
        Vec::new(),
        parts
            .iter()
            .map(|p| MacElite {
                genome: p.genome.clone(),
                labels: p.labels.clone(),
                obj: p.obj.clone(),
            })
            .collect(),
        pop,
    );
    let num_gens = 30usize;
    for t in 1..=num_gens {
        let objs: Vec<Obj> = arch.iter().map(|a| a.obj.clone()).collect();
        let crowd = arch_crowd(&objs);
        macro_step(
            &g,
            &wadj,
            &mut parts,
            &arch,
            &crowd,
            &cfg,
            inertia(t, num_gens),
            0.5,
        );
        arch = update_macro_archive(
            arch,
            parts
                .iter()
                .map(|p| MacElite {
                    genome: p.genome.clone(),
                    labels: p.labels.clone(),
                    obj: p.obj.clone(),
                })
                .collect(),
            pop,
        );
        for (k, (p, &c0)) in parts.iter().zip(&cards).enumerate() {
            let c = p.genome.iter().filter(|&&b| b != 0).count();
            assert_eq!(c, c0, "gen {t}: particle {k} drifted {c0} -> {c}");
            let cp = p.pbest.iter().filter(|&&b| b != 0).count();
            assert_eq!(cp, c0, "gen {t}: particle {k} pbest drifted");
        }
    }
}

#[test]
fn archive_never_contains_a_dominated_member() {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as usize
    };
    for case in 0..100 {
        let cap = 1 + case % 12;
        let mut arch: Vec<MicElite> = Vec::new();
        for _round in 0..4 {
            let lattice = 2 + case % 7;
            let fresh: Vec<MicElite> = (0..(1 + next() % 20))
                .map(|_| {
                    let o = vec![(next() % lattice) as f64, (next() % lattice) as f64];
                    let labels = (0..4).map(|_| (next() % 3) as i32).collect();
                    MicElite { labels, obj: o }
                })
                .collect();
            arch = update_micro_archive(arch, fresh, cap);
            assert!(arch.len() <= cap, "case {case}: archive over capacity");
            for a in 0..arch.len() {
                for b in 0..arch.len() {
                    if a != b {
                        assert!(
                            !dominates(&arch[b].obj, &arch[a].obj),
                            "case {case}: dominated member survived"
                        );
                        assert!(
                            !(arch[a].obj == arch[b].obj && arch[a].labels == arch[b].labels),
                            "case {case}: exact duplicate survived"
                        );
                    }
                }
            }
        }
    }
}
