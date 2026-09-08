//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rayon::prelude::*;

use crate::core::algorithms::rimpso::Labels;
use crate::core::algorithms::rimpso::utils::sampling::slot_rng;
use crate::core::graph::CsrGraph;

use super::particle::{Particle, ScratchPool};

const SEED_SALT: u64 = u64::MAX;

fn scatter(g: &CsrGraph, rng_seed: u64, slot: usize) -> Labels {
    let mut r = slot_rng(rng_seed, SEED_SALT, slot);
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

pub fn seed(g: &CsrGraph, gammas: &[f64], rng_seed: u64) -> Vec<Particle> {
    let pool = ScratchPool::new(g.n);
    gammas
        .par_iter()
        .enumerate()
        .map(|(k, &gamma)| {
            let mut held = pool.get();
            let s = &mut *held;
            let pos = scatter(g, rng_seed, k);
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
            p.canonicalize(s);
            p.best.copy_from_slice(&p.pos);
            p.best_score = p.score();
            p
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::rimpso::config::defaults::DEFAULT_SEED;
    use crate::core::algorithms::rimpso::swarm::ladder::ladder;
    use crate::core::algorithms::rimpso::swarm::particle::Scratch;
    use crate::core::algorithms::rimpso::utils::fixtures::ring_of_cliques;

    fn counts(p: &Particle) -> usize {
        let mut c = p.pos.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    }

    #[test]
    fn every_particle_starts_at_an_unoptimised_scatter() {
        let g = ring_of_cliques(16, 6);
        let gammas = ladder(&g, 24);
        let swarm = seed(&g, &gammas, DEFAULT_SEED);
        for p in &swarm {
            assert!(
                counts(p) > 16,
                "a particle arrived pre-optimised at {} communities",
                counts(p)
            );
            for u in 0..g.n {
                let c = p.pos[u];
                assert!(c >= 0 && (c as usize) < g.n, "label {c} out of range");
            }
        }
    }

    #[test]
    fn different_seeds_start_the_same_rung_somewhere_else() {
        let g = ring_of_cliques(16, 6);
        let a = scatter(&g, DEFAULT_SEED, 3);
        assert_eq!(a, scatter(&g, DEFAULT_SEED, 3), "one seed, two scatters");
        let moved = (1..8u64).filter(|&s| scatter(&g, s, 3) != a).count();
        assert!(
            moved >= 6,
            "only {moved} of 7 seeds moved rung 3 off its default start, so the \
             seed barely reaches the initial population"
        );
    }

    #[test]
    fn the_rungs_start_from_different_scatters() {
        let g = ring_of_cliques(16, 6);
        let swarm = seed(&g, &ladder(&g, 24), DEFAULT_SEED);
        let distinct = {
            let mut v: Vec<&Vec<i32>> = swarm.iter().map(|p| &p.pos).collect();
            v.sort();
            v.dedup();
            v.len()
        };
        assert!(
            distinct > swarm.len() / 2,
            "only {distinct} of {} particles have distinct starts",
            swarm.len()
        );
    }

    #[test]
    fn seeding_is_reproducible_and_the_counts_are_exact() {
        let g = ring_of_cliques(8, 5);
        let gammas = ladder(&g, 12);
        let a = seed(&g, &gammas, DEFAULT_SEED);
        let b = seed(&g, &gammas, DEFAULT_SEED);
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
}
