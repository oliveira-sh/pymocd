//! One particle's flight: the velocity update and the position move it drives.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rngs::StdRng;

use crate::core::algorithms::mopso::config::Cfg;
use crate::core::algorithms::mopso::utils::sampling::unit;
use crate::core::graph::CsrGraph;

use super::local::best_move;
use super::particle::{Particle, Scratch};

/// Advance one particle by one iteration.
///
/// The continuous update `v <- w v + c1 r1 (pbest - x) + c2 r2 (leader - x)`
/// carries over term by term once "distance" is read per node: a node whose
/// label differs from an attractor's contributes that attractor's pull, a node
/// that already agrees contributes nothing. Velocity is then the probability
/// that the node is unstable this iteration, clamped to `[0,1]` so it stays a
/// probability, and `x <- x + v` becomes: an unstable node adopts one of the
/// two attractors' labels, chosen in proportion to their pulls.
///
/// A node that agrees with both attractors but still carries momentum has no
/// direction to follow, so it drifts to a neighbour's label — the discrete
/// remnant of coasting.
///
/// The stable nodes are where the resolution-directed local move runs. The two
/// branches are drawn from the same uniform and are therefore mutually
/// exclusive, which is deliberate: a node that has just taken an attractor's
/// label should not be overwritten in the same iteration. It also anneals the
/// swarm for free — velocity is high early, so the swarm explores, and decays
/// as the attractors are reached, so late iterations are almost all local
/// refinement.
///
/// Costs one uniform per node, plus two per particle.
pub fn advance(
    g: &CsrGraph,
    p: &mut Particle,
    leader: &[i32],
    cfg: &Cfg,
    s: &mut Scratch,
    r: &mut StdRng,
) {
    s.load(&p.pos);
    let r1 = unit(r) * cfg.cognitive;
    let r2 = unit(r) * cfg.social;
    let local_floor = 1.0 - cfg.local_rate;

    #[allow(clippy::needless_range_loop)]
    for j in 0..g.n {
        // An isolated vertex belongs to no community. Following an attractor
        // into one would cost `pair` and return no edge, and the local move can
        // never undo it because it has no neighbour to reason from.
        if g.neighbors(j).is_empty() {
            continue;
        }
        let here = p.pos[j];
        let pull_p = if p.best[j] == here { 0.0 } else { r1 };
        let pull_l = if leader[j] == here { 0.0 } else { r2 };

        let v = (cfg.inertia * f64::from(p.vel[j]) + pull_p + pull_l).clamp(0.0, 1.0);
        p.vel[j] = v as f32;

        let u = unit(r);
        if u < v {
            // `u / v` is uniform on [0,1) given the node is unstable, so the
            // attractor split and the drift target cost no further draws.
            let t = u / v;
            let total = pull_p + pull_l;
            let target = if total > 0.0 {
                if t * total < pull_p {
                    p.best[j]
                } else {
                    leader[j]
                }
            } else {
                let nbrs = g.neighbors(j);
                let pick = (t * nbrs.len() as f64) as usize;
                p.pos[nbrs[pick.min(nbrs.len() - 1)] as usize]
            };
            if target != here {
                let (from_links, to_links) = p.link_pair(g, j, target);
                p.relocate(j, target, from_links, to_links, s);
            }
        } else if u > local_floor {
            best_move(g, p, j, s);
        }
    }
    // Both attractors are read as labels next iteration, so the flight has to
    // end in the shared naming or the reading is meaningless.
    p.canonicalize(s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::Labels;
    use crate::core::algorithms::mopso::swarm::particle::seeded;
    use crate::core::algorithms::mopso::utils::fixtures::ring_of_cliques;
    use crate::core::algorithms::mopso::utils::sampling::slot_rng;

    fn particle(g: &CsrGraph, pos: Labels, best: Labels, gamma: f64) -> Particle {
        let (mut p, _) = seeded(g, pos, gamma);
        p.best = best;
        p
    }

    fn setup() -> (CsrGraph, Labels, Labels) {
        let g = ring_of_cliques(6, 5);
        let pos: Labels = (0..g.n as i32).collect();
        let best: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        (g, pos, best)
    }

    #[test]
    fn the_counts_stay_exact_across_a_flight() {
        let (g, pos, best) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let mut p = particle(&g, pos, best, 0.3);
        let mut s = Scratch::new(g.n);
        let cfg = Cfg::default();
        for t in 1..=25u64 {
            let mut r = slot_rng(t, 3);
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r);
            let mut check = Scratch::new(g.n);
            assert_eq!(
                check.measure(&g, &p.pos),
                (p.internal, p.pair_sum),
                "iteration {t}: the incremental counts left the partition"
            );
        }
    }

    #[test]
    fn a_flight_is_reproducible() {
        let (g, pos, best) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let cfg = Cfg::default();
        let fly = || {
            let mut p = particle(&g, pos.clone(), best.clone(), 0.3);
            let mut s = Scratch::new(g.n);
            for t in 1..=10u64 {
                let mut r = slot_rng(t, 3);
                advance(&g, &mut p, &leader, &cfg, &mut s, &mut r);
            }
            p.pos
        };
        assert_eq!(fly(), fly());
    }

    #[test]
    fn zero_velocity_and_no_local_search_leaves_the_particle_where_it_was() {
        let (g, pos, best) = setup();
        let leader: Labels = vec![0; g.n];
        let cfg = Cfg::new(10, 10, 0.0, 0.0, 0.0, 0.0, 10, 0);
        let mut p = particle(&g, pos.clone(), best, 0.3);
        let mut s = Scratch::new(g.n);
        let mut r = slot_rng(1, 0);
        advance(&g, &mut p, &leader, &cfg, &mut s, &mut r);
        assert_eq!(p.pos, pos, "the particle moved with every force switched off");
    }

    #[test]
    fn the_social_term_alone_pulls_toward_the_leader() {
        let (g, pos, _) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        // No cognition, no local search, no inertia: only the leader acts.
        let cfg = Cfg::new(10, 10, 0.0, 0.0, 1.0, 0.0, 10, 0);
        let mut p = particle(&g, pos.clone(), pos.clone(), 0.3);
        let mut s = Scratch::new(g.n);
        let before = p.pos.iter().zip(&leader).filter(|(a, b)| a == b).count();
        for t in 1..=8u64 {
            let mut r = slot_rng(t, 1);
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r);
        }
        let after = p.pos.iter().zip(&leader).filter(|(a, b)| a == b).count();
        assert!(after > before, "the swarm did not converge on its leader");
    }

    #[test]
    fn every_label_stays_a_valid_slot() {
        let (g, pos, best) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let mut p = particle(&g, pos, best, 0.3);
        let mut s = Scratch::new(g.n);
        let cfg = Cfg::default();
        for t in 1..=15u64 {
            let mut r = slot_rng(t, 7);
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r);
            assert!(
                p.pos.iter().all(|&c| c >= 0 && (c as usize) < g.n),
                "iteration {t} produced a label outside [0,n)"
            );
        }
    }
}
