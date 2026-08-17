//! Seeding the swarm: one particle per rung of the resolution ladder.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rayon::prelude::*;

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::config::Cfg;
use crate::core::algorithms::mopso::utils::sampling::slot_rng;
use crate::core::graph::CsrGraph;

use super::local::best_move;
use super::merge::merge_sweep;
use super::particle::{Particle, Scratch};

/// Salt reserved for seeding, so no iteration can collide with it.
const SEED_SALT: u64 = u64::MAX;

/// Every node takes a random neighbour's label. The result is far from a
/// partition but it is locally coherent, which is a much better start than
/// random labels: the local move has structure to sharpen rather than to
/// invent.
fn scatter(g: &CsrGraph, slot: usize) -> Labels {
    let mut r = slot_rng(SEED_SALT, slot);
    (0..g.n)
        .map(|i| {
            let nbrs = g.neighbors(i);
            if nbrs.is_empty() {
                i as i32
            } else {
                nbrs[r.random_range(0..nbrs.len())] as i32
            }
        })
        .collect()
}

/// The seeded swarm, one particle per rung of `gammas`.
///
/// Each particle is driven to its own rung before the first iteration, so the
/// archive starts out holding a spread of granularities rather than a hundred
/// copies of the same coarse partition.
pub fn seed(g: &CsrGraph, cfg: &Cfg, gammas: &[f64]) -> Vec<Particle> {
    gammas
        .par_iter()
        .enumerate()
        .map(|(k, &gamma)| {
            let mut s = Scratch::new(g.n);
            let pos = scatter(g, k);
            let (internal, pair_sum) = s.measure(g, &pos);
            let mut p = Particle {
                vel: vec![0.0; g.n],
                best: pos.clone(),
                pos,
                internal,
                pair_sum,
                gamma,
                best_score: f64::NEG_INFINITY,
            };
            // Node moves sharpen boundaries, merges coarsen; neither reaches
            // the other's granularity alone, so a rung is only reached by
            // alternating them until the particle stops moving.
            for _ in 0..cfg.seed_rounds {
                let mut moved = false;
                for u in 0..g.n {
                    moved |= best_move(g, &mut p, u, &mut s);
                }
                moved |= merge_sweep(g, &mut p, &mut s);
                if !moved {
                    break;
                }
            }
            p.canonicalize(&mut s);
            p.best.copy_from_slice(&p.pos);
            p.best_score = p.score();
            p
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::swarm::ladder::ladder;
    use crate::core::algorithms::mopso::utils::fixtures::ring_of_cliques;

    fn counts(p: &Particle) -> usize {
        let mut c = p.pos.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    }

    #[test]
    fn seeding_spans_granularity_from_coarse_to_fine() {
        let g = ring_of_cliques(16, 6);
        let cfg = Cfg::default();
        let gammas = ladder(&g, 24);
        let swarm = seed(&g, &cfg, &gammas);
        let ks: Vec<usize> = swarm.iter().map(counts).collect();
        assert!(
            ks[0] < ks[ks.len() - 1],
            "the ladder produced no granularity spread: {ks:?}"
        );
        assert!(
            ks.iter().max().unwrap() - ks.iter().min().unwrap() > 4,
            "the spread is too narrow to be a profile: {ks:?}"
        );
    }

    #[test]
    fn a_mid_rung_particle_lands_on_the_planted_cliques() {
        let g = ring_of_cliques(16, 6);
        let cfg = Cfg::default();
        let gammas = ladder(&g, 40);
        let swarm = seed(&g, &cfg, &gammas);
        assert!(
            swarm.iter().any(|p| counts(p) == 16),
            "no rung of the ladder found the sixteen cliques: {:?}",
            swarm.iter().map(counts).collect::<Vec<_>>()
        );
    }

    #[test]
    fn seeding_is_reproducible_and_the_counts_are_exact() {
        let g = ring_of_cliques(8, 5);
        let cfg = Cfg::default();
        let gammas = ladder(&g, 12);
        let a = seed(&g, &cfg, &gammas);
        let b = seed(&g, &cfg, &gammas);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.pos, y.pos, "seeding is not reproducible");
        }
        let mut check = Scratch::new(g.n);
        for p in &a {
            assert_eq!(check.measure(&g, &p.pos), (p.internal, p.pair_sum));
            assert_eq!(p.best, p.pos);
            assert_eq!(p.best_score, p.score());
        }
    }

    #[test]
    fn labels_stay_inside_the_slot_range() {
        let g = ring_of_cliques(8, 5);
        let swarm = seed(&g, &Cfg::default(), &ladder(&g, 12));
        for p in &swarm {
            assert!(p.pos.iter().all(|&c| c >= 0 && (c as usize) < g.n));
        }
    }
}
