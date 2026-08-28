//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rngs::StdRng;

use crate::core::algorithms::mr_mocd::config::Cfg;
use crate::core::algorithms::mr_mocd::utils::sampling::unit;
use crate::core::graph::CsrGraph;

use super::local::best_move;
use super::merge::merge_sweep;
use super::particle::{Particle, Scratch};

#[inline]
fn push(s: &mut Scratch, j: u32) {
    if s.inq[j as usize] == 0 {
        s.inq[j as usize] = 1;
        s.queue.push(j);
    }
}

fn repair(g: &CsrGraph, p: &mut Particle, s: &mut Scratch) {
    let mut head = 0usize;
    while head < s.queue.len() {
        let j = s.queue[head] as usize;
        head += 1;
        s.inq[j] = 0;
        let nbrs = g.neighbors(j);
        if nbrs.is_empty() {
            continue;
        }
        if best_move(nbrs, p, j, s) {
            push(s, j as u32);
            for &w in nbrs {
                push(s, w);
            }
        }
        if head >= 4096 && head * 2 >= s.queue.len() {
            s.queue.drain(..head);
            head = 0;
        }
    }
    for &j in &s.queue {
        s.inq[j as usize] = 0;
    }
    s.queue.clear();
}

pub fn advance(
    g: &CsrGraph,
    p: &mut Particle,
    leader: &[i32],
    cfg: &Cfg,
    s: &mut Scratch,
    r: &mut StdRng,
    local_search: bool,
) {
    s.load(&p.pos);
    let r1 = unit(r) * cfg.cognitive;
    let r2 = unit(r) * cfg.social;
    let local_floor = 1.0 - cfg.local_rate;

    #[allow(clippy::needless_range_loop)]
    for j in 0..g.n {
        let nbrs = g.neighbors(j);
        if nbrs.is_empty() {
            continue;
        }
        let here = p.pos[j];
        let pull_p = if p.best[j] == here { 0.0 } else { r1 };
        let pull_l = if leader[j] == here { 0.0 } else { r2 };

        let v = (cfg.inertia * f64::from(p.vel[j]) + pull_p + pull_l).clamp(0.0, 1.0);
        p.vel[j] = v as f32;

        let u = unit(r);
        if u < v {
            let t = u / v;
            let total = pull_p + pull_l;
            let target = if total > 0.0 {
                if t * total < pull_p {
                    p.best[j]
                } else {
                    leader[j]
                }
            } else {
                let pick = (t * nbrs.len() as f64) as usize;
                p.pos[nbrs[pick.min(nbrs.len() - 1)] as usize]
            };
            if target != here {
                let (from_links, to_links) = p.link_pair(nbrs, here, target);
                p.relocate(j, target, from_links, to_links, s);
                if local_search {
                    push(s, j as u32);
                    for &w in nbrs {
                        push(s, w);
                    }
                }
            }
        } else if !local_search && u > local_floor {
            best_move(nbrs, p, j, s);
        }
    }
    if local_search {
        repair(g, p, s);
        merge_sweep(g, p, s);
        s.load(&p.pos);
    }
    p.canonicalize(s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mr_mocd::Labels;
    use crate::core::algorithms::mr_mocd::swarm::particle::seeded;
    use crate::core::algorithms::mr_mocd::utils::fixtures::ring_of_cliques;
    use crate::core::algorithms::mr_mocd::utils::sampling::slot_rng;

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
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r, true);
            let mut check = Scratch::new(g.n);
            assert_eq!(
                check.measure(&g, &p.pos),
                (p.internal, p.pair_sum),
                "iteration {t}: the incremental counts left the partition"
            );
        }
    }

    #[test]
    fn zero_velocity_and_no_local_search_leaves_the_particle_where_it_was() {
        let (g, pos, best) = setup();
        let leader: Labels = vec![0; g.n];
        let cfg = Cfg::new(10, 10, 0.0, 0.0, 0.0, 0.0, 10);
        let mut p = particle(&g, pos.clone(), best, 0.3);
        let mut s = Scratch::new(g.n);
        let mut r = slot_rng(1, 0);
        advance(&g, &mut p, &leader, &cfg, &mut s, &mut r, false);
        assert_eq!(
            p.pos, pos,
            "the particle moved with every force switched off"
        );
    }

    #[test]
    fn the_social_term_alone_pulls_toward_the_leader() {
        let (g, pos, _) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 5) * 5).collect();
        let cfg = Cfg::new(10, 10, 0.0, 0.0, 1.0, 0.0, 10);
        let mut p = particle(&g, pos.clone(), pos, 0.3);
        let mut s = Scratch::new(g.n);
        let before = p.pos.iter().zip(&leader).filter(|(a, b)| a == b).count();
        for t in 1..=8u64 {
            let mut r = slot_rng(t, 1);
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r, true);
        }
        let after = p.pos.iter().zip(&leader).filter(|(a, b)| a == b).count();
        assert!(after > before, "the swarm did not converge on its leader");
    }

    #[test]
    fn a_repaired_flight_ends_at_a_local_optimum() {
        let (g, pos, best) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let cfg = Cfg::default();
        let mut p = particle(&g, pos, best, 0.3);
        let mut s = Scratch::new(g.n);
        for t in 1..=10u64 {
            let mut r = slot_rng(t, 3);
            advance(&g, &mut p, &leader, &cfg, &mut s, &mut r, true);
        }
        let mut check = Scratch::new(g.n);
        check.load(&p.pos);
        for u in 0..g.n {
            let mut probe = particle(&g, p.pos.clone(), p.best.clone(), 0.3);
            let mut ps = Scratch::new(g.n);
            ps.load(&probe.pos);
            assert!(
                !best_move(g.neighbors(u), &mut probe, u, &mut ps),
                "vertex {u} still had a better move after the repair"
            );
        }
    }

    #[test]
    fn the_local_search_schedule_changes_where_the_particle_lands() {
        let (g, pos, best) = setup();
        let leader: Labels = (0..g.n).map(|i| (i as i32 / 10) * 10).collect();
        let cfg = Cfg::default();
        let fly = |ls: bool| {
            let mut p = particle(&g, pos.clone(), best.clone(), 0.3);
            let mut s = Scratch::new(g.n);
            for t in 1..=8u64 {
                let mut r = slot_rng(t, 3);
                advance(&g, &mut p, &leader, &cfg, &mut s, &mut r, ls);
            }
            p.pos
        };
        assert_eq!(
            fly(false),
            fly(false),
            "swarm-only motion is not reproducible"
        );
        assert_eq!(
            fly(true),
            fly(true),
            "the scheduled sweep is not reproducible"
        );
        assert_ne!(
            fly(true),
            fly(false),
            "the local search made no difference to where the particle landed"
        );
    }
}
