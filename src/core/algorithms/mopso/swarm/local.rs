//! The resolution-directed local move: the best CPM step for one node at its own resolution.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use super::particle::{Particle, Scratch};

/// Moves `u` to whichever adjacent community maximises `k_uc - gamma * s_c`, the CPM gain
/// at the particle's resolution, and returns whether it moved. The size penalty is what
/// stops this from sliding into one giant community the way a majority move does.
pub fn best_move(nbrs: &[u32], p: &mut Particle, u: usize, s: &mut Scratch) -> bool {
    if nbrs.is_empty() {
        return false;
    }
    let from = p.pos[u];

    for &v in nbrs {
        let c = p.pos[v as usize] as usize;
        let e = s.link[c];
        if e == 0 {
            s.touched.push(c as u32);
        }
        s.link[c] = e + 1;
    }

    // Staying put is scored with `u` already removed from its community, so every
    // comparison is like for like.
    let from_links = s.link[from as usize];
    let mut best = from;
    let mut best_gain = f64::from(from_links) - p.gamma * f64::from(s.size[from as usize] - 1);
    let mut best_links = from_links;
    // The count is taken and cleared in one visit, which is the difference between one
    // scattered write per neighbouring community and two.
    for &c in &s.touched {
        let ci = c as usize;
        let e = s.link[ci];
        s.link[ci] = 0;
        let gain = f64::from(e) - p.gamma * f64::from(s.size[ci]);
        // Ties keep the lower label, so the outcome does not follow the neighbour layout.
        if gain > best_gain || (gain == best_gain && (c as i32) < best) {
            best_gain = gain;
            best = c as i32;
            best_links = e;
        }
    }
    s.touched.clear();

    if best == from {
        return false;
    }
    p.relocate(u, best, from_links, best_links, s);
    true
}

/// Test-only shim: the callers hold the adjacency already, the tests do not.
#[cfg(test)]
fn best_move_at(
    g: &crate::core::graph::CsrGraph,
    p: &mut Particle,
    u: usize,
    s: &mut Scratch,
) -> bool {
    best_move(g.neighbors(u), p, u, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::swarm::particle::seeded as particle;
    use crate::core::graph::CsrGraph;
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_triangles};

    #[test]
    fn a_free_node_joins_the_community_it_has_most_edges_into() {
        // Node 2 is a singleton with two edges into {0,1} and one into {3,4,5}.
        let g = two_triangles();
        let (mut p, mut s) = particle(&g, vec![0, 0, 2, 3, 3, 3], 1e-9);
        assert!(best_move_at(&g, &mut p, 2, &mut s));
        assert_eq!(p.pos[2], 0);
    }

    #[test]
    fn the_size_penalty_overrides_the_majority() {
        // Same graph and node, at a resolution where a community of two already costs
        // more than the extra edge is worth.
        let g = two_triangles();
        let (mut p, mut s) = particle(&g, vec![0, 0, 2, 3, 3, 3], 4.0);
        assert!(
            !best_move_at(&g, &mut p, 2, &mut s),
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
                if best_move_at(&g, &mut p, u, &mut s) {
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
                best_move_at(&g, &mut p, u, &mut s);
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
        assert!(!best_move_at(&g, &mut p, 2, &mut s));
        assert_eq!(p.pos[2], 2);
    }
}
