//! The swarm loop: seed the ladder, fly, archive, repeat.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rayon::prelude::*;

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::config::Cfg;
use crate::core::algorithms::mopso::objectives::Obj;
use crate::core::algorithms::mopso::pareto::Archive;
use crate::core::algorithms::mopso::utils::sampling::slot_rng;
use crate::core::graph::CsrGraph;

use super::init::seed;
use super::ladder::ladder;
use super::motion::advance;
use super::particle::Scratch;

/// Runs the swarm and returns its archive: the partitions and the objective point each
/// occupies. The archive is the deliverable; choosing one member out of it is the selector's job.
pub fn run(g: &CsrGraph, cfg: &Cfg) -> (Vec<Labels>, Vec<Obj>) {
    if g.n == 0 {
        return (Vec::new(), Vec::new());
    }
    let gammas = ladder(g, cfg.pop);
    let mut swarm = seed(g, cfg, &gammas);

    let mut archive = Archive::with_capacity(cfg.archive);
    for p in &swarm {
        archive.offer(p.objective(g), &p.pos);
    }
    archive.prune();

    for t in 1..=cfg.gens as u64 {
        let view = &archive; // read-only for the whole flight, so every particle sees the same front.
        swarm.par_iter_mut().enumerate().for_each_init(
            || Scratch::new(g.n),
            |s, (i, p)| {
                let mut r = slot_rng(t, i);
                let leader = view.leader(&mut r);
                advance(g, p, view.position(leader), cfg, s, &mut r);
                let score = p.score();
                if score > p.best_score {
                    p.best_score = score;
                    p.best.copy_from_slice(&p.pos);
                }
            },
        );

        // Slot order, so the archive never depends on which particle a worker finished first.
        for p in &swarm {
            archive.offer(p.objective(g), &p.pos);
        }
        archive.prune();
    }

    archive.into_parts()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::objectives::{measure, obj_of};
    use crate::core::algorithms::mopso::pareto::dominates;
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_cliques};

    fn small() -> Cfg {
        Cfg::new(24, 20, 0.4, 0.7, 0.7, 0.35, 24, 3)
    }

    #[test]
    fn the_archive_is_a_nonempty_valid_front() {
        let g = ring_of_cliques(10, 5);
        let (front, objs) = run(&g, &small());
        assert!(!front.is_empty());
        assert_eq!(front.len(), objs.len());
        assert!(front.iter().all(|p| p.len() == g.n));
        for i in 0..objs.len() {
            for j in 0..objs.len() {
                assert!(i == j || !dominates(&objs[i], &objs[j]));
            }
        }
    }

    #[test]
    fn the_reported_objectives_match_the_partitions() {
        let g = ring_of_cliques(8, 6);
        let (front, objs) = run(&g, &small());
        let mut size = vec![0u32; g.n];
        let mut live = Vec::new();
        for (p, o) in front.iter().zip(&objs) {
            let direct = obj_of(&g, measure(&g, p, &mut size, &mut live));
            assert_eq!(&direct, o, "the archived objective is not the partition's");
        }
    }

    #[test]
    fn the_run_is_deterministic() {
        let g = ring_of_cliques(8, 5);
        let a = run(&g, &small());
        let b = run(&g, &small());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn the_front_spans_granularity() {
        let g = ring_of_cliques(16, 6);
        let (front, _) = run(&g, &small());
        let ks: Vec<usize> = front.iter().map(distinct).collect();
        assert!(
            ks.iter().max().unwrap() > &(2 * ks.iter().min().unwrap()),
            "the archive holds a single granularity: {ks:?}"
        );
    }

    #[test]
    fn the_planted_partition_reaches_the_front() {
        let g = ring_of_cliques(12, 5);
        let (front, _) = run(&g, &small());
        let recovers_the_cliques =
            |p: &Labels| (0..g.n).all(|i| p[i] == p[i / 5 * 5]) && distinct(p) == 12;
        assert!(
            front.iter().any(recovers_the_cliques),
            "the twelve cliques never made the archive"
        );
    }

    fn distinct(p: &Labels) -> usize {
        let mut c = p.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    }

    #[test]
    fn an_edgeless_graph_still_returns_a_partition() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[]);
        let (front, objs) = run(&g, &small());
        assert!(!front.is_empty());
        assert!(front.iter().all(|p| p.len() == 3));
        assert_eq!(objs.len(), front.len());
    }

    #[test]
    fn a_two_clique_graph_is_split() {
        let g = two_cliques();
        let (front, _) = run(&g, &small());
        assert!(
            front.iter().any(|p| {
                (1..5).all(|i| p[i] == p[0]) && (6..10).all(|i| p[i] == p[5]) && p[0] != p[5]
            }),
            "the two cliques were never separated"
        );
    }
}

/// Randomised checks that the archive's invariants survive parameter values and graph
/// shapes the shipped defaults never produce.
#[cfg(test)]
mod fuzz {
    use super::*;
    use crate::core::algorithms::mopso::front::select_index;
    use crate::core::algorithms::mopso::objectives::{measure, obj_of};

    fn rnd_graph(n: i32, e: usize, seed: u64) -> CsrGraph {
        let mut st = seed;
        let mut next = |m: u64| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            (st >> 33) % m
        };
        let mut edges = Vec::new();
        for _ in 0..e {
            let a = next(n as u64) as i32;
            let b = next(n as u64) as i32;
            edges.push((a, b));
        }
        CsrGraph::from_edges(&(0..n).collect::<Vec<i32>>(), &edges)
    }

    #[test]
    fn invariants_hold_under_extreme_parameters() {
        let cfgs = vec![
            Cfg::new(8, 12, 1.0, 1.0, 1.0, 1.0, 8, 8),
            Cfg::new(8, 12, 0.0, 0.0, 0.0, 1.0, 8, 0),
            Cfg::new(8, 12, 1.0, 0.0, 1.0, 0.0, 2, 1),
            Cfg::new(2, 30, 0.9, 0.9, 0.9, 0.9, 2, 3),
            Cfg::new(16, 5, 0.5, 1.0, 1.0, 0.5, 3, 20),
        ];
        for seed in [1u64, 7, 99, 12345] {
            for (n, e) in [
                (1i32, 0usize),
                (2, 1),
                (3, 0),
                (5, 3),
                (20, 10),
                (40, 200),
                (60, 60),
            ] {
                let g = rnd_graph(n.max(1), e, seed);
                for cfg in &cfgs {
                    let (front, objs) = run(&g, cfg);
                    for (p, o) in front.iter().zip(&objs) {
                        assert_eq!(p.len(), g.n);
                        assert!(
                            p.iter().all(|&c| c >= 0 && (c as usize) < g.n),
                            "label out of range"
                        );
                        let mut size = vec![0u32; g.n];
                        let mut live = Vec::new();
                        let direct = obj_of(&g, measure(&g, p, &mut size, &mut live));
                        assert_eq!(&direct, o, "objective drifted from the partition");
                    }
                }
            }
        }
    }

    #[test]
    fn the_archive_stays_valid_over_three_hundred_random_graphs() {
        let mut st = 0xDEAD_BEEF_1234_5678u64;
        let mut next = |m: u64| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            (st >> 33) % m.max(1)
        };
        for trial in 0..300u32 {
            let n = 1 + next(60) as i32;
            let e = next(150) as usize;
            let g = {
                let mut edges = Vec::new();
                for _ in 0..e {
                    let a = next(n as u64) as i32;
                    let b = next(n as u64) as i32;
                    edges.push((a, b));
                }
                CsrGraph::from_edges(&(0..n).collect::<Vec<i32>>(), &edges)
            };
            let cfg = Cfg::new(
                (2 + next(20)) as usize,
                next(12) as usize,
                next(11) as f64 / 10.0,
                next(11) as f64 / 10.0,
                next(11) as f64 / 10.0,
                next(11) as f64 / 10.0,
                (1 + next(10)) as usize,
                next(10) as usize,
            );
            let (front, objs) = run(&g, &cfg);
            assert!(!front.is_empty(), "trial {trial}: empty archive");
            for (p, o) in front.iter().zip(&objs) {
                assert_eq!(p.len(), g.n);
                assert!(
                    p.iter().all(|&c| c >= 0 && (c as usize) < g.n),
                    "trial {trial}: label out of range"
                );
                let mut size = vec![0u32; g.n];
                let mut live = Vec::new();
                assert_eq!(
                    &obj_of(&g, measure(&g, p, &mut size, &mut live)),
                    o,
                    "trial {trial}: objective drift"
                );
            }
            let idx = select_index(&g, &front, &objs);
            assert!(
                idx < front.len(),
                "trial {trial}: selected {idx} of {}",
                front.len()
            );
        }
    }

    #[test]
    fn an_isolated_vertex_never_joins_a_community() {
        let mut edges = Vec::new();
        for c in 0..6i32 {
            let lo = c * 10;
            for a in lo..lo + 5 {
                for b in (a + 1)..lo + 5 {
                    edges.push((a, b));
                }
            }
            edges.push((lo, (lo + 10) % 60));
        }
        let g = CsrGraph::from_edges(&(0..60).collect::<Vec<i32>>(), &edges);
        let cfg = Cfg::new(24, 30, 0.4, 0.7, 0.7, 0.35, 24, 8);
        let (front, _) = run(&g, &cfg);
        for p in &front {
            for u in 0..g.n {
                if g.deg[u] == 0 {
                    let shared = (0..g.n).filter(|&v| p[v] == p[u]).count();
                    assert_eq!(
                        shared, 1,
                        "isolated node {u} joined a community: label {}",
                        p[u]
                    );
                }
            }
        }
    }
}
