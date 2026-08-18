//! A particle and the exact bookkeeping that makes its objective free.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mopso::Labels;
use crate::core::algorithms::mopso::objectives::{Counts, Obj, load_sizes, measure, obj_of};
use crate::core::graph::CsrGraph;

/// One particle of the swarm. The counts are maintained through every node move rather
/// than recomputed, and being integers they cannot drift from a full rescan.
pub struct Particle {
    pub pos: Labels,
    pub vel: Vec<f32>,
    pub best: Labels,
    pub internal: i64,   // intra-community edges of `pos`, each counted once.
    pub pair_sum: i64,   // `sum_c C(n_c, 2)` over `pos`.
    pub gamma: f64, // the particle's niche on the resolution profile; only its local move reads it.
    pub best_score: f64, // CPM at `gamma` of `best`, what the cognitive memory is ranked by.
}

/// One `Scratch` per worker, reused by every particle that worker handles. Rayon hands each
/// particle out as its own task, so a scratch built per task is a scratch built per particle
/// per iteration; holding them here builds `n`-sized buffers once for the whole run instead.
pub struct ScratchPool {
    slots: Vec<Slot>,
}

/// A slot of its own cache line. The buffer headers live inside the slot and a push in the
/// innermost loop writes one, so packing two workers' slots together would have them
/// fighting over the line.
#[repr(align(128))]
struct Slot(std::sync::Mutex<Scratch>);

impl ScratchPool {
    pub fn new(n: usize) -> Self {
        Self {
            slots: (0..rayon::current_num_threads().max(1))
                .map(|_| Slot(std::sync::Mutex::new(Scratch::new(n))))
                .collect(),
        }
    }

    /// The calling worker's own slot. No two workers reach the same one, so the lock is
    /// there to carry the borrow rather than to serialise anything. What a particle leaves
    /// behind cannot reach the next one: `load` rebuilds the sizes, `link` is restored to
    /// zero by every path that touches it, and the rest is rebuilt or cleared before use.
    pub fn get(&self) -> std::sync::MutexGuard<'_, Scratch> {
        let i = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        self.slots[i]
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Per-worker buffers, reused across every particle a rayon task handles.
pub struct Scratch {
    pub size: Vec<u32>,    // community sizes of the particle being advanced.
    pub live: Vec<u32>,    // the communities `size` holds a nonzero count for.
    pub link: Vec<u32>,    // edges from the node being moved into each community.
    pub touched: Vec<u32>, // the communities `link` holds a nonzero count for.
    pub rep: Vec<i32>,     // lowest-numbered member of each community, for `canonicalize`.
    pub bucket: Vec<u32>,  // vertices ordered by community, then the merge relabelling.
    pub start: Vec<u32>,   // where each community's run of `bucket` begins.
    pub cand: Vec<u128>,   // the merges one sweep is choosing between, in sweep order.
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Self {
            size: vec![0; n],
            live: Vec::new(),
            link: vec![0; n],
            touched: Vec::new(),
            rep: Vec::new(),
            bucket: Vec::new(),
            start: Vec::new(),
            cand: Vec::new(),
        }
    }

    pub fn load(&mut self, pos: &[i32]) {
        load_sizes(pos, &mut self.size, &mut self.live);
    }

    /// Groups the vertices by the community they are in, leaving community `c` holding the
    /// `size[c]` entries of `bucket` from `start[c]`. A counting sort off the sizes that are
    /// already maintained, so it costs one pass and no comparison.
    pub fn bucket_by_community(&mut self, pos: &[i32]) {
        let n = pos.len();
        if self.start.len() != n {
            self.start.clear();
            self.start.resize(n, 0);
            self.bucket.clear();
            self.bucket.resize(n, 0);
        }
        let mut acc = 0u32;
        for c in 0..n {
            acc += self.size[c];
            self.start[c] = acc; // one past the end of the community, for now.
        }
        debug_assert_eq!(
            acc as usize, n,
            "the community sizes do not cover the vertices"
        );
        // Filling from the back turns each end into the start it will be read as.
        for (u, &c) in pos.iter().enumerate().rev() {
            let c = c as usize;
            self.start[c] -= 1;
            self.bucket[self.start[c] as usize] = u as u32;
        }
    }

    /// Both counts of `pos` from scratch, leaving the sizes loaded. Used once per particle
    /// at seeding; after that `relocate` maintains them.
    pub fn measure(&mut self, g: &CsrGraph, pos: &[i32]) -> Counts {
        measure(g, pos, &mut self.size, &mut self.live)
    }
}

impl Particle {
    pub fn objective(&self, g: &CsrGraph) -> Obj {
        obj_of(g, (self.internal, self.pair_sum))
    }

    /// CPM at the particle's own resolution, the scalar its personal best is ranked by.
    pub fn score(&self) -> f64 {
        self.internal as f64 - self.gamma * self.pair_sum as f64
    }

    /// Moves `u` into community `to`, given the neighbour counts the caller has just
    /// scanned. `to` may be a label no node currently carries, so it is registered as live.
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

    /// Relabels every community by its lowest-numbered member, which aligns the label
    /// spaces of two particles so that copying an attractor's label means what it says.
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
        for c in &mut self.pos {
            *c = s.rep[*c as usize];
        }
    }

    /// How many of `nbrs` are in `from`, the vertex's own community, and how many in `to`,
    /// counted in one scan. Both labels are the caller's, which already holds them.
    pub fn link_pair(&self, nbrs: &[u32], from: i32, to: i32) -> (u32, u32) {
        let (mut f, mut t) = (0u32, 0u32);
        for &v in nbrs {
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
            let (f, t) = p.link_pair(g.neighbors(u), p.pos[u], to);
            p.relocate(u, to, f, t, &mut s);
        }
        assert_eq!(rescan(&g, &p.pos), (p.internal, p.pair_sum));
    }

    #[test]
    fn moving_into_a_community_no_node_holds_is_still_exact() {
        let g = two_triangles();
        let (mut p, mut s) = seeded(&g, vec![0, 0, 0, 3, 3, 3], 0.1);
        let (f, t) = p.link_pair(g.neighbors(2), p.pos[2], 5);
        p.relocate(2, 5, f, t, &mut s);
        assert_eq!(rescan(&g, &p.pos), (p.internal, p.pair_sum));

        let (mut q, _) = seeded(&g, vec![1, 1, 1, 1, 1, 1], 0.1);
        s.load(&q.pos);
        assert!(s.size.iter().all(|&x| x == 0 || x == 6), "{:?}", s.size);
        let (f, t) = q.link_pair(g.neighbors(0), q.pos[0], 4);
        q.relocate(0, 4, f, t, &mut s);
        assert_eq!(rescan(&g, &q.pos), (q.internal, q.pair_sum));
    }

    #[test]
    fn sizes_track_the_moves() {
        let g = ring_of_cliques(4, 5);
        let (mut p, mut s) = seeded(&g, (0..g.n as i32).collect(), 0.1);
        for u in 1..5 {
            let (f, t) = p.link_pair(g.neighbors(u), p.pos[u], 0);
            p.relocate(u, 0, f, t, &mut s);
        }
        assert_eq!(s.size[0], 5);
        assert!((1..5).all(|c| s.size[c] == 0));
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
        let (mut p, mut s) = seeded(
            &g,
            vec![
                7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 19, 19, 19, 19, 19, 11, 11, 11, 11, 11,
            ],
            0.1,
        );
        let before = (p.internal, p.pair_sum);
        p.canonicalize(&mut s);
        assert_eq!(&p.pos[..5], &[0; 5]);
        assert_eq!(&p.pos[5..10], &[5; 5]);
        assert_eq!(&p.pos[10..15], &[10; 5]);
        assert_eq!(&p.pos[15..], &[15; 5]);
        assert_eq!(
            (p.internal, p.pair_sum),
            before,
            "relabelling changed a count"
        );
        let mut q = p.pos.clone();
        p.canonicalize(&mut s);
        assert_eq!(p.pos, q);
        q.rotate_left(0);
        let (mut r, mut s2) = seeded(
            &g,
            vec![
                2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17,
            ],
            0.1,
        );
        r.canonicalize(&mut s2);
        assert_eq!(
            r.pos, p.pos,
            "two particles disagreed on the same partition"
        );
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
