//! Sec. 4.2.1 community centre selection and its per-walk frequency decay.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::rngs::StdRng;

use crate::core::algorithms::cdrme::sampling::Wheel;
use crate::core::algorithms::cdrme::topology::Topology;

/// Degree-weighted centre draws (Sec. 4.2.1) whose weights are divided by a
/// node's `primWalk` frequency after each walk (Sec. 4.2.3). The weight decays,
/// it never excludes, so one node can be drawn as a centre twice.
pub struct Centers {
    wheel: Wheel,
}

impl Centers {
    pub fn new(topology: &Topology) -> Self {
        let mut weight = vec![0.0; topology.n];
        for &v in &topology.active {
            weight[v as usize] = topology.degree(v) as f64;
        }
        Self {
            wheel: Wheel::new(weight),
        }
    }

    pub fn draw(&self, rng: &mut StdRng) -> Option<u32> {
        self.wheel.draw(rng).map(|i| i as u32)
    }

    pub fn decay(&mut self, prim: &[(u32, u32)]) {
        for &(node, freq) in prim {
            if freq > 1 {
                let decayed = self.wheel.weight(node as usize) / f64::from(freq);
                self.wheel.set(node as usize, decayed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::cdrme::sampling::{SALT_CENTER, slot_rng};

    #[test]
    fn only_active_nodes_are_drawn_and_hubs_lead() {
        // star on 0 plus an isolated node 5
        let topology = Topology::from_edges(&[0, 1, 2, 3, 4, 5], &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let centers = Centers::new(&topology);
        let mut rng = slot_rng(SALT_CENTER, 0);
        let mut hub = 0;
        for _ in 0..400 {
            let drawn = centers.draw(&mut rng).unwrap();
            assert_ne!(drawn, 5, "an isolated node was drawn as a centre");
            hub += usize::from(drawn == 0);
        }
        assert!(hub > 150, "degree weighting is not in force: {hub}");
    }

    #[test]
    fn decay_divides_by_the_walk_frequency() {
        let topology = Topology::from_edges(&[0, 1, 2], &[(0, 1), (0, 2), (1, 2)]);
        let mut centers = Centers::new(&topology);
        centers.decay(&[(0, 4), (1, 1)]);
        assert!((centers.wheel.weight(0) - 0.5).abs() < 1e-12);
        assert!((centers.wheel.weight(1) - 2.0).abs() < 1e-12);
    }
}
