//! Eq. (7) and Algorithm 1: softmax-weighted walks and the primWalk they leave.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;

use crate::core::algorithms::cdrme::topology::Topology;

/// Eq. (7): `RandomWalkLength(v) = Degree(v) + alpha * |V|/ENC`.
pub fn walk_length(topology: &Topology, v: u32, alpha_walk: f64) -> usize {
    // |V|/ENC is AvgDegree by Eq. (6), so ENC cancels out of Eq. (7) entirely
    let raw = topology.degree(v) as f64 + alpha_walk * topology.avg_degree;
    raw.round().max(1.0) as usize
}

/// Algorithm 1, with the transition law of line 10 cached per node.
///
/// `softmax(|Neighbor(current) INTERSECT Neighbor(v)|)` is a property of
/// `current` alone, so each row is built once and reused across every walk of
/// every centre. The exponent is shifted by the row maximum before `exp`, the
/// accurate-softmax form of the paper's own citation, without which a dense
/// row overflows on raw counts.
pub struct Walker {
    transitions: Vec<Option<Box<[f64]>>>,
    freq: Vec<u32>,
    visits: Vec<u32>,
    stamp: Vec<u32>,
    touched: Vec<u32>,
}

fn softmax_cumulative(topology: &Topology, u: u32) -> Box<[f64]> {
    let row = topology.neighbors(u);
    let counts: Vec<u32> = row
        .iter()
        .map(|&v| topology.common_neighbors(u, v))
        .collect();
    let peak = counts.iter().copied().max().unwrap_or(0);
    let mut cumulative = Vec::with_capacity(row.len());
    let mut running = 0.0;
    for &c in &counts {
        running += (f64::from(c) - f64::from(peak)).exp();
        cumulative.push(running);
    }
    cumulative.into_boxed_slice()
}

impl Walker {
    pub fn new(topology: &Topology) -> Self {
        Self {
            transitions: (0..topology.n).map(|_| None).collect(),
            freq: vec![0; topology.n],
            visits: vec![0; topology.n],
            stamp: vec![0; topology.n],
            touched: Vec::new(),
        }
    }

    fn step(&mut self, topology: &Topology, u: u32, rng: &mut StdRng) -> u32 {
        if self.transitions[u as usize].is_none() {
            self.transitions[u as usize] = Some(softmax_cumulative(topology, u));
        }
        let cumulative = self.transitions[u as usize].as_ref().unwrap();
        let Some(&total) = cumulative.last() else {
            return u;
        };
        let target = rng.random_range(0.0..total);
        let index = cumulative.partition_point(|&c| c <= target);
        topology.neighbors(u)[index.min(cumulative.len() - 1)]
    }

    /// Runs `n_walk` walks of `length` steps from `start` and returns the
    /// `length` most frequent nodes with their frequencies (lines 15-17).
    ///
    /// Line 9 unions each node into a per-walk set and line 15 counts "its
    /// repetition in different walks", so a frequency is the number of walks
    /// that contained the node, not the number of visits. Visits survive only
    /// as the tie-break of line 16, which the paper leaves open and which would
    /// otherwise fall through to the node id and bias the primWalk low.
    pub fn run(
        &mut self,
        topology: &Topology,
        start: u32,
        length: usize,
        n_walk: usize,
        rng: &mut StdRng,
    ) -> Vec<(u32, u32)> {
        self.touched.clear();
        for walk in 0..n_walk {
            let mark = walk as u32 + 1;
            let mut current = start;
            for _ in 0..length {
                let slot = current as usize;
                if self.stamp[slot] != mark {
                    if self.freq[slot] == 0 {
                        self.touched.push(current);
                    }
                    self.stamp[slot] = mark;
                    self.freq[slot] += 1;
                }
                self.visits[slot] += 1;
                current = self.step(topology, current, rng);
            }
        }

        let mut prim: Vec<(u32, u32)> = self
            .touched
            .iter()
            .map(|&v| (v, self.freq[v as usize]))
            .collect();
        prim.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| self.visits[b.0 as usize].cmp(&self.visits[a.0 as usize]))
                .then_with(|| a.0.cmp(&b.0))
        });
        prim.truncate(length);

        for &v in &self.touched {
            self.freq[v as usize] = 0;
            self.visits[v as usize] = 0;
            self.stamp[v as usize] = 0;
        }
        prim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::cdrme::sampling::{SALT_WALK, slot_rng};

    fn barbell() -> Topology {
        let mut edges = vec![(4, 5)];
        for base in [0, 5] {
            for a in base..base + 5 {
                for b in a + 1..base + 5 {
                    edges.push((a, b));
                }
            }
        }
        Topology::from_edges(&(0..10).collect::<Vec<i32>>(), &edges)
    }

    #[test]
    fn equation_seven_adds_the_average_degree() {
        let topology = barbell();
        // AvgDegree = 2*21/10 = 4.2, deg(0) = 4  ->  round(4 + 4.2) = 8
        assert_eq!(walk_length(&topology, 0, 1.0), 8);
        assert_eq!(walk_length(&topology, 0, 2.0), 12);
    }

    #[test]
    fn a_walk_stays_in_its_own_clique_and_leads_with_the_centre() {
        let topology = barbell();
        let mut walker = Walker::new(&topology);
        let prim = walker.run(&topology, 0, 8, 50, &mut slot_rng(SALT_WALK, 0));
        assert_eq!(prim[0].1, 50, "the centre is not among the most frequent nodes");
        assert!(prim.len() <= 8);
        let inside: u32 = prim.iter().filter(|&&(v, _)| v < 5).map(|&(_, f)| f).sum();
        let outside: u32 = prim.iter().filter(|&&(v, _)| v >= 5).map(|&(_, f)| f).sum();
        assert!(inside > 4 * outside, "in {inside} out {outside}");
    }

    #[test]
    fn a_frequency_never_exceeds_the_number_of_walks() {
        let topology = barbell();
        let mut walker = Walker::new(&topology);
        let prim = walker.run(&topology, 3, 8, 7, &mut slot_rng(SALT_WALK, 1));
        assert!(prim.iter().all(|&(_, f)| (1..=7).contains(&f)));
        assert_eq!(prim.iter().find(|&&(v, _)| v == 3).unwrap().1, 7);
    }

    #[test]
    fn the_scratch_buffers_are_reset_between_runs() {
        let topology = barbell();
        let mut walker = Walker::new(&topology);
        let first = walker.run(&topology, 0, 8, 20, &mut slot_rng(SALT_WALK, 2));
        let second = walker.run(&topology, 0, 8, 20, &mut slot_rng(SALT_WALK, 2));
        assert_eq!(first, second);
    }

    #[test]
    fn a_lone_edge_walks_without_stalling() {
        let topology = Topology::from_edges(&[0, 1], &[(0, 1)]);
        let mut walker = Walker::new(&topology);
        let prim = walker.run(&topology, 0, 3, 5, &mut slot_rng(SALT_WALK, 3));
        assert_eq!(prim.len(), 2);
    }
}
