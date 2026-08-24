//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mr_mocd::Labels;
use crate::core::algorithms::mr_mocd::objectives::{Counts, Obj, load_sizes, measure, obj_of};
use crate::core::graph::CsrGraph;

pub struct Particle {
    pub pos: Labels,
    pub vel: Vec<f32>,
    pub best: Labels,
    pub internal: i64,
    pub pair_sum: i64,
    pub gamma: f64,
    pub best_score: f64,
}

pub struct ScratchPool {
    slots: Vec<Slot>,
}

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

    pub fn get(&self) -> std::sync::MutexGuard<'_, Scratch> {
        let i = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        self.slots[i]
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub struct Scratch {
    pub size: Vec<u32>,
    pub live: Vec<u32>,
    pub link: Vec<u32>,
    pub touched: Vec<u32>,
    pub rep: Vec<i32>,
    pub bucket: Vec<u32>,
    pub start: Vec<u32>,
    pub cand: Vec<u128>,
    pub queue: Vec<u32>,
    pub inq: Vec<u8>,
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
            queue: Vec::new(),
            inq: vec![0; n],
        }
    }

    pub fn load(&mut self, pos: &[i32]) {
        load_sizes(pos, &mut self.size, &mut self.live);
    }

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
            self.start[c] = acc;
        }
        debug_assert_eq!(
            acc as usize, n,
            "the community sizes do not cover the vertices"
        );
        for (u, &c) in pos.iter().enumerate().rev() {
            let c = c as usize;
            self.start[c] -= 1;
            self.bucket[self.start[c] as usize] = u as u32;
        }
    }

    pub fn measure(&mut self, g: &CsrGraph, pos: &[i32]) -> Counts {
        measure(g, pos, &mut self.size, &mut self.live)
    }
}

impl Particle {
    pub fn objective(&self, g: &CsrGraph) -> Obj {
        obj_of(g, (self.internal, self.pair_sum))
    }

    pub fn score(&self) -> f64 {
        self.internal as f64 - self.gamma * self.pair_sum as f64
    }

    pub fn relocate(&mut self, u: usize, to: i32, from_links: u32, to_links: u32, s: &mut Scratch) {
        let from = self.pos[u];
        debug_assert_ne!(from, to, "relocating a vertex into its own community");
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
    use crate::core::algorithms::mr_mocd::objectives::measure;
    use crate::core::algorithms::mr_mocd::utils::fixtures::{ring_of_cliques, two_triangles};

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
            if to == p.pos[u] {
                continue;
            }
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
        let q = p.pos.clone();
        p.canonicalize(&mut s);
        assert_eq!(p.pos, q);
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
}
