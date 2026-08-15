//! SMOCC: Sparse Multi-Objective Co-evolutionary Community detection,
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::algorithms::smocc2::mopso::pareto::dominates;
use crate::core::graph::CsrGraph;

use super::api::{smocc2, smocc2_fronts};
use super::config::defaults::*;
use super::config::schedule::inertia;
use super::gpu::Gpu;
use super::mopso::archive::update_micro_archive;
use super::mopso::init::macro_cmax;
use super::mopso::particles::{MacElite, MicParticle, MicElite};
use super::mopso::steps::macro_move;
use super::mopso::particles::MacParticle;

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
    let out = match smocc2(
        &nodes,
        &two_clique_edges(),
        60,
        40,
        DEFAULT_GAP,
        DEFAULT_TURB,
        DEFAULT_MACRO_CAP,
    ) {
        Err(e) => {
            eprintln!("skipping GPU-only test (no usable CUDA device): {e}");
            return;
        }
        Ok(o) => o,
    };
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
    let out = match smocc2(
        &nodes,
        &two_triangle_edges(),
        40,
        20,
        DEFAULT_GAP,
        DEFAULT_TURB,
        DEFAULT_MACRO_CAP,
    ) {
        Err(e) => {
            eprintln!("skipping GPU-only test (no usable CUDA device): {e}");
            return;
        }
        Ok(o) => o,
    };
    let c: FxHashMap<i32, i32> = out.into_iter().collect();
    assert_eq!(c[&6], -1);
}

#[test]
fn fronts_are_nonempty() {
    let nodes: Vec<i32> = (0..6).collect();
    let fronts = match smocc2_fronts(
        &nodes,
        &two_triangle_edges(),
        40,
        20,
        DEFAULT_GAP,
        DEFAULT_TURB,
        true,
        0,
        DEFAULT_MACRO_CAP,
    ) {
        Err(e) => {
            eprintln!("skipping GPU-only test (no usable CUDA device): {e}");
            return;
        }
        Ok(o) => o,
    };
    assert!(!fronts.is_empty());
    assert!(fronts.iter().all(|f| f.len() == 6));
}

#[test]
fn objective_modes_produce_full_partitions() {
    let nodes: Vec<i32> = (0..10).collect();
    let edges = two_clique_edges();
    for obj_mode in [106u16, 160, 166, 100, 0, 6] {
        let a = match smocc2_fronts(
            &nodes,
            &edges,
            40,
            15,
            DEFAULT_GAP,
            DEFAULT_TURB,
            true,
            obj_mode,
            DEFAULT_MACRO_CAP,
        ) {
            Err(e) => {
                eprintln!("skipping GPU-only test (no usable CUDA device): {e}");
                return;
            }
            Ok(o) => o,
        };
        assert!(!a.is_empty(), "obj_mode {obj_mode} produced an empty front");
        assert!(a.iter().all(|f| f.len() == 10));
    }
}

#[test]
fn macro_cardinality_invariant_survives_updates() {
    let (nodes, edges) = ring_of_cliques(8, 5);
    let g = CsrGraph::from_edges(&nodes, &edges);
    let n = g.n;
    let cmax = macro_cmax(n, DEFAULT_MACRO_CAP);
    let pop = 12usize;

    let mut r_state = 0x243f6a8885a308d3u64;
    let mut next = move || {
        r_state = r_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (r_state >> 33) as usize
    };
    let mut parts: Vec<MacParticle> = (0..pop)
        .map(|_| {
            let c = 1 + next() % cmax;
            let mut genome = vec![0u8; n];
            let mut placed = 0;
            while placed < c {
                let i = next() % n;
                if genome[i] == 0 {
                    genome[i] = 1;
                    placed += 1;
                }
            }
            MacParticle {
                pbest: genome.clone(),
                pbest_obj: vec![0.0, 0.0],
                genome,
                labels: vec![0; n],
                obj: vec![0.0, 0.0],
            }
        })
        .collect();
    let cards: Vec<usize> = parts
        .iter()
        .map(|p| p.genome.iter().filter(|&&b| b != 0).count())
        .collect();

    let arch: Vec<MacElite> = parts
        .iter()
        .map(|p| MacElite {
            genome: p.genome.clone(),
            labels: p.labels.clone(),
            obj: vec![next() as f64 % 7.0, next() as f64 % 7.0],
        })
        .collect();
    let crowd = vec![1.0f64; arch.len()];

    let num_gens = 30usize;
    for t in 1..=num_gens {
        let w = inertia(t, num_gens);
        for p in parts.iter_mut() {
            macro_move(n, p, &arch, &crowd, w, 0.5);
        }
        for (k, (p, &c0)) in parts.iter().zip(&cards).enumerate() {
            let c = p.genome.iter().filter(|&&b| b != 0).count();
            assert_eq!(c, c0, "gen {t}: particle {k} drifted {c0} -> {c}");
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
