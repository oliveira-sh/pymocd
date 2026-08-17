//! The resolution-directed local move: the single best CPM step for one node,
//! at the particle's own resolution.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::graph::CsrGraph;

use super::particle::{Particle, Scratch};

/// Move `u` to whichever adjacent community maximises the CPM gain at the
/// particle's resolution, staying put if none beats where it already is.
///
/// Moving `u` from `a` to `c` changes CPM by
/// `(k_uc - k_ua) - gamma * (s_c - (s_a - 1))`, so the best target maximises
/// `k_uc - gamma * s_c` with `u` itself removed from its own community's size.
/// This is the counterpart of the majority-neighbour move the other detectors
/// here use, except that the size penalty is what stops it from sliding into
/// one giant community — which is exactly how a majority move fails at high
/// mixing.
///
/// Returns whether `u` moved. Costs one scan of `u`'s neighbours.
pub fn best_move(g: &CsrGraph, p: &mut Particle, u: usize, s: &mut Scratch) -> bool {
    let nbrs = g.neighbors(u);
    if nbrs.is_empty() {
        return false;
    }
    let from = p.pos[u];

    for &v in nbrs {
        let c = p.pos[v as usize] as usize;
        if s.link[c] == 0 {
            s.touched.push(c as u32);
        }
        s.link[c] += 1;
    }

    // Staying put is scored with `u` already removed from its community, so the
    // comparison against every alternative is like for like.
    let from_links = s.link[from as usize];
    let mut best = from;
    let mut best_gain = f64::from(from_links) - p.gamma * f64::from(s.size[from as usize] - 1);
    let mut best_links = from_links;
    for &c in &s.touched {
        let ci = c as usize;
        let gain = f64::from(s.link[ci]) - p.gamma * f64::from(s.size[ci]);
        // Ties keep the lower label, so the outcome does not depend on the
        // order the neighbours happened to be laid out in.
        if gain > best_gain || (gain == best_gain && (c as i32) < best) {
            best_gain = gain;
            best = c as i32;
            best_links = s.link[ci];
        }
    }

    for &c in &s.touched {
        s.link[c as usize] = 0;
    }
    s.touched.clear();

    if best == from {
        return false;
    }
    p.relocate(u, best, from_links, best_links, s);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::swarm::particle::seeded as particle;
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_triangles};

    #[test]
    fn a_free_node_joins_the_community_it_has_most_edges_into() {
        let g = two_triangles();
        // Node 2 is a singleton; it has two edges into {0,1} and one into {3,4,5}.
        let (mut p, mut s) = particle(&g, vec![0, 0, 2, 3, 3, 3], 1e-9);
        assert!(best_move(&g, &mut p, 2, &mut s));
        assert_eq!(p.pos[2], 0);
    }

    #[test]
    fn the_size_penalty_overrides_the_majority() {
        // Same graph and node, but at a resolution where a community of two
        // already costs more than the extra edge is worth.
        let g = two_triangles();
        let (mut p, mut s) = particle(&g, vec![0, 0, 2, 3, 3, 3], 4.0);
        assert!(
            !best_move(&g, &mut p, 2, &mut s),
            "a high resolution still merged into the larger side"
        );
    }

    #[test]
    fn repeated_moves_only_improve_cpm_and_reach_a_fixed_point() {
        let g = ring_of_cliques(6, 5);
        let (mut p, mut s) = particle(&g, (0..g.n as i32).collect(), 0.2);
        let mut prev = p.score();
        let mut moved = true;
        let mut sweeps = 0;
        while moved && sweeps < 100 {
            moved = false;
            sweeps += 1;
            for u in 0..g.n {
                if best_move(&g, &mut p, u, &mut s) {
                    moved = true;
                    let now = p.score();
                    assert!(now > prev - 1e-9, "a move lowered CPM: {prev} -> {now}");
                    prev = now;
                }
            }
        }
        assert!(!moved, "the local search never settled");

        let mut check = Scratch::new(g.n);
        assert_eq!(check.measure(&g, &p.pos), (p.internal, p.pair_sum));
    }

    #[test]
    fn a_mid_resolution_recovers_the_planted_cliques() {
        let g = ring_of_cliques(8, 6);
        let mean_deg = 2.0 * g.m as f64 / g.n as f64;
        let (mut p, mut s) = particle(&g, (0..g.n as i32).collect(), mean_deg / 6.0);
        for _ in 0..20 {
            for u in 0..g.n {
                best_move(&g, &mut p, u, &mut s);
            }
        }
        for c in 0..8usize {
            let base = p.pos[c * 6];
            for j in 1..6 {
                assert_eq!(p.pos[c * 6 + j], base, "clique {c} was split");
            }
        }
        let mut ks: Vec<i32> = p.pos.clone();
        ks.sort_unstable();
        ks.dedup();
        assert_eq!(ks.len(), 8, "the ring did not resolve into eight cliques");
    }

    #[test]
    fn an_isolated_node_never_moves() {
        let g = CsrGraph::from_edges(&[0, 1, 2], &[(0, 1)]);
        let (mut p, mut s) = particle(&g, vec![0, 0, 2], 0.1);
        assert!(!best_move(&g, &mut p, 2, &mut s));
        assert_eq!(p.pos[2], 2);
    }
}
