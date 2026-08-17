//! A particle and the exact bookkeeping that makes its objective free.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::objectives::{Counts, Obj, load_sizes, measure, obj_of};
use crate::core::graph::CsrGraph;

/// One particle of the swarm.
///
/// `internal` and `pair_sum` are the two extensive counts the CPM split is
/// built from. They are maintained through every single node move instead of
/// being recomputed, which is what removes the `O(m)` objective pass from the
/// iteration; being integers, they cannot drift, so after ten thousand moves
/// they still hold exactly what a full rescan would produce.
pub struct Particle {
    pub pos: Labels,
    pub vel: Vec<f32>,
    pub best: Labels,
    /// Intra-community edges of `pos`, each counted once.
    pub internal: i64,
    /// `sum_c C(n_c, 2)` over `pos`.
    pub pair_sum: i64,
    /// The particle's niche on the resolution profile. Only its local move
    /// reads it; the objective it is archived on stays resolution-free.
    pub gamma: f64,
    /// CPM at `gamma` of `best`, the value the cognitive memory is ranked by.
    pub best_score: f64,
}

/// Per-worker buffers, reused across every particle a rayon task handles so an
/// iteration allocates nothing.
pub struct Scratch {
    /// Community sizes of the particle currently being advanced.
    pub size: Vec<u32>,
    /// The communities `size` holds a nonzero count for, so the next particle
    /// clears only those entries instead of the whole array.
    pub live: Vec<u32>,
    /// Edges from the node being moved into each community.
    pub link: Vec<u32>,
    /// The communities `link` holds a nonzero count for.
    pub touched: Vec<u32>,
    /// Lowest-numbered member of each community, for `canonicalize`.
    pub rep: Vec<i32>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Self {
            size: vec![0; n],
            live: Vec::new(),
            link: vec![0; n],
            touched: Vec::new(),
            rep: Vec::new(),
        }
    }

    pub fn load(&mut self, pos: &[i32]) {
        load_sizes(pos, &mut self.size, &mut self.live);
    }

    /// Both counts of `pos` from scratch, leaving the sizes loaded. Used once
    /// per particle at seeding; after that the counts are maintained by
    /// `relocate` and this is only ever called by tests, as an oracle.
    pub fn measure(&mut self, g: &CsrGraph, pos: &[i32]) -> Counts {
        measure(g, pos, &mut self.size, &mut self.live)
    }
}

impl Particle {
    /// The point this particle occupies in objective space.
    pub fn objective(&self, g: &CsrGraph) -> Obj {
        obj_of(g, (self.internal, self.pair_sum))
    }

    /// CPM at the particle's own resolution — the scalar its personal best is
    /// ranked by. A total order, so the memory update never has to break a tie
    /// by which iteration happened to arrive first.
    pub fn score(&self) -> f64 {
        self.internal as f64 - self.gamma * self.pair_sum as f64
    }

    /// Move `u` into community `to`, given how many of `u`'s neighbours sit in
    /// its current community and in `to`. The caller has just scanned those
    /// neighbours, and rescanning them here would double the only part of a
    /// move that costs anything.
    ///
    /// `to` may be a label no node currently carries — the attractors hand out
    /// labels from other partitions — so a community coming into existence is
    /// registered as live, or the next particle would inherit its count.
    pub fn relocate(&mut self, u: usize, to: i32, from_links: u32, to_links: u32, s: &mut Scratch) {
        let from = self.pos[u];
        if from == to {
            return;
        }
        let (fi, ti) = (from as usize, to as usize);
        if s.size[ti] == 0 {
            s.live.push(to as u32);
        }
        self.internal += i64::from(to_links) - i64::from(from_links);
        self.pair_sum += i64::from(s.size[ti]) - i64::from(s.size[fi]) + 1;
        s.size[fi] -= 1;
        s.size[ti] += 1;
        self.pos[u] = to;
    }

    /// Relabel every community by its lowest-numbered member.
    ///
    /// Without this the attractors transmit noise. A particle's labels are node
    /// ids inherited from the scatter, so two particles that found the *same*
    /// community usually call it different things, and copying a leader's label
    /// at one node moves that node into whichever unrelated community the
    /// receiving particle happens to have parked on that number. Canonical
    /// labels align the two label spaces: agreeing particles agree on the
    /// number too, so `pos[j] = leader[j]` means what it says, and the social
    /// term becomes a graft of the leader's community rather than a shuffle.
    ///
    /// A pure relabelling, so neither count changes.
    pub fn canonicalize(&mut self, s: &mut Scratch) {
        s.rep.clear();
        s.rep.resize(self.pos.len(), -1);
        for u in 0..self.pos.len() {
            let c = self.pos[u] as usize;
            if s.rep[c] < 0 {
                s.rep[c] = u as i32;
            }
        }
        for c in self.pos.iter_mut() {
            *c = s.rep[*c as usize];
        }
    }

    /// Neighbours of `u` in its own community and in `to`, in one scan.
    pub fn link_pair(&self, g: &CsrGraph, u: usize, to: i32) -> (u32, u32) {
        let from = self.pos[u];
        let (mut f, mut t) = (0u32, 0u32);
        for &v in g.neighbors(u) {
            let c = self.pos[v as usize];
            f += u32::from(c == from);
            t += u32::from(c == to);
        }
        (f, t)
    }
}

#[cfg(test)]
pub fn seeded(g: &CsrGraph, pos: Labels, gamma: f64) -> (Particle, Scratch) {
    let mut s = Scratch::new(g.n);
    let (internal, pair_sum) = s.measure(g, &pos);
    (
        Particle {
            vel: vec![0.0; g.n],
            best: pos.clone(),
            pos,
            internal,
            pair_sum,
            gamma,
            best_score: f64::NEG_INFINITY,
        },
        s,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::mopso::objectives::{measure, obj_of};
    use crate::core::algorithms::mopso::utils::fixtures::{ring_of_cliques, two_triangles};

    fn rescan(g: &CsrGraph, pos: &[i32]) -> Counts {
        let mut size = vec![0u32; g.n];
        let mut live = Vec::new();
        measure(g, pos, &mut size, &mut live)
    }

    #[test]
    fn incremental_counts_match_a_full_rescan_after_many_moves() {
        let g = ring_of_cliques(6, 5);
        let (mut p, mut s) = seeded(&g, (0..g.n as i32).collect(), 0.1);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..4000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((state >> 33) as usize) % g.n;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let nb = g.neighbors(u);
            let to = p.pos[nb[((state >> 33) as usize) % nb.len()] as usize];
            let (f, t) = p.link_pair(&g, u, to);
            p.relocate(u, to, f, t, &mut s);
        }
        assert_eq!(rescan(&g, &p.pos), (p.internal, p.pair_sum));
    }

    #[test]
    fn moving_into_a_community_no_node_holds_is_still_exact() {
        // The attractors hand out labels absent from the current partition;
        // the size array must not carry them over to the next particle.
        let g = two_triangles();
        let (mut p, mut s) = seeded(&g, vec![0, 0, 0, 3, 3, 3], 0.1);
        let (f, t) = p.link_pair(&g, 2, 5);
        p.relocate(2, 5, f, t, &mut s);
        assert_eq!(rescan(&g, &p.pos), (p.internal, p.pair_sum));

        let (mut q, _) = seeded(&g, vec![1, 1, 1, 1, 1, 1], 0.1);
        s.load(&q.pos);
        assert!(s.size.iter().all(|&x| x == 0 || x == 6), "{:?}", s.size);
        let (f, t) = q.link_pair(&g, 0, 4);
        q.relocate(0, 4, f, t, &mut s);
        assert_eq!(rescan(&g, &q.pos), (q.internal, q.pair_sum));
    }

    #[test]
    fn sizes_track_the_moves() {
        let g = ring_of_cliques(4, 5);
        let (mut p, mut s) = seeded(&g, (0..g.n as i32).collect(), 0.1);
        for u in 1..5 {
            let (f, t) = p.link_pair(&g, u, 0);
            p.relocate(u, 0, f, t, &mut s);
        }
        assert_eq!(s.size[0], 5);
        assert!((1..5).all(|c| s.size[c] == 0));
        // The first clique is whole: all ten of its edges are internal.
        assert_eq!((p.internal, p.pair_sum), (10, 10));
    }

    #[test]
    fn the_objective_agrees_with_a_direct_evaluation() {
        let g = two_triangles();
        for pos in [
            vec![0, 0, 0, 3, 3, 3],
            vec![0; 6],
            (0..6).collect::<Labels>(),
            vec![0, 0, 1, 1, 4, 4],
        ] {
            let (p, _) = seeded(&g, pos.clone(), 0.1);
            assert_eq!(p.objective(&g), obj_of(&g, rescan(&g, &pos)));
        }
    }

    #[test]
    fn canonical_labels_name_a_community_by_its_lowest_member() {
        let g = ring_of_cliques(4, 5);
        let (mut p, mut s) = seeded(&g, vec![7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 19, 19, 19, 19, 19, 11, 11, 11, 11, 11], 0.1);
        let before = (p.internal, p.pair_sum);
        p.canonicalize(&mut s);
        assert_eq!(&p.pos[..5], &[0; 5]);
        assert_eq!(&p.pos[5..10], &[5; 5]);
        assert_eq!(&p.pos[10..15], &[10; 5]);
        assert_eq!(&p.pos[15..], &[15; 5]);
        assert_eq!((p.internal, p.pair_sum), before, "relabelling changed a count");
        // Idempotent, and it aligns two particles that found the same grouping.
        let mut q = p.pos.clone();
        p.canonicalize(&mut s);
        assert_eq!(p.pos, q);
        q.rotate_left(0);
        let (mut r, mut s2) = seeded(&g, vec![2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17], 0.1);
        r.canonicalize(&mut s2);
        assert_eq!(r.pos, p.pos, "two particles disagreed on the same partition");
    }

    #[test]
    fn relocating_into_the_same_community_is_a_no_op() {
        let g = two_triangles();
        let (mut p, mut s) = seeded(&g, vec![0, 0, 0, 3, 3, 3], 0.1);
        let before = (p.internal, p.pair_sum);
        p.relocate(1, 0, 2, 2, &mut s);
        assert_eq!((p.internal, p.pair_sum), before);
        assert_eq!(s.size[0], 3);
    }
}
