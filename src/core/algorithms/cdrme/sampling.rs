//! The deterministic per-slot RNG contract for cdrme and its roulette wheel.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

// shared with mr_mocd and gdpso so every module documents one contract
const RNG_BASE: u64 = 0x5CA1_E5EED;

pub const SALT_CENTER: u64 = 0x0CD_0001;
pub const SALT_WALK: u64 = 0x0CD_0002;
pub const SALT_MERGE: u64 = 0x0CD_0003;

/// One independent stream per `(salt, slot)` pair; never depends on the thread count.
pub fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

/// Fenwick-indexed roulette over non-negative weights.
///
/// Sec. 4.2.1 draws `ENC` centres with `P(v) proportional to degree(v)` and
/// rewrites weights after every walk, so a flat cumulative scan would cost
/// `O(|V|^2 / AvgDegree)` on its own; both operations here are `O(log |V|)`.
pub struct Wheel {
    weight: Vec<f64>,
    tree: Vec<f64>,
    span: usize,
}

impl Wheel {
    pub fn new(weight: Vec<f64>) -> Self {
        let n = weight.len();
        let mut tree = vec![0.0; n + 1];
        for (i, &w) in weight.iter().enumerate() {
            tree[i + 1] += w;
            let parent = (i + 1) + ((i + 1) & (i + 1).wrapping_neg());
            if parent <= n {
                let carried = tree[i + 1];
                tree[parent] += carried;
            }
        }
        Self {
            weight,
            tree,
            span: n.next_power_of_two(),
        }
    }

    pub fn weight(&self, i: usize) -> f64 {
        self.weight[i]
    }

    pub fn set(&mut self, i: usize, w: f64) {
        let delta = w - self.weight[i];
        self.weight[i] = w;
        let mut k = i + 1;
        while k < self.tree.len() {
            self.tree[k] += delta;
            k += k & k.wrapping_neg();
        }
    }

    pub fn total(&self) -> f64 {
        let n = self.weight.len();
        let mut sum = 0.0;
        let mut k = n;
        while k > 0 {
            sum += self.tree[k];
            k -= k & k.wrapping_neg();
        }
        sum
    }

    /// Draws an index with probability proportional to its weight, or `None`
    /// when nothing carries positive weight.
    pub fn draw(&self, rng: &mut StdRng) -> Option<usize> {
        let total = self.total();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let mut target = rng.random_range(0.0..total);
        let mut pos = 0usize;
        let mut step = self.span;
        while step > 0 {
            let next = pos + step;
            if next < self.tree.len() && self.tree[next] <= target {
                target -= self.tree[next];
                pos = next;
            }
            step >>= 1;
        }
        // rounding can land one slot past the intended one on a zero weight
        while pos < self.weight.len() && self.weight[pos] <= 0.0 {
            pos += 1;
        }
        if pos < self.weight.len() {
            Some(pos)
        } else {
            self.weight.iter().position(|&w| w > 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wheel_only_ever_draws_positive_weights() {
        let mut wheel = Wheel::new(vec![0.0, 3.0, 0.0, 1.0, 0.0]);
        let mut rng = slot_rng(SALT_CENTER, 0);
        let mut hits = [0usize; 5];
        for _ in 0..2000 {
            hits[wheel.draw(&mut rng).unwrap()] += 1;
        }
        assert_eq!(hits[0] + hits[2] + hits[4], 0);
        assert!(hits[1] > hits[3], "{hits:?}");
        wheel.set(1, 0.0);
        assert!((wheel.total() - 1.0).abs() < 1e-12);
        for _ in 0..50 {
            assert_eq!(wheel.draw(&mut rng), Some(3));
        }
    }

    #[test]
    fn an_empty_wheel_draws_nothing() {
        let wheel = Wheel::new(vec![0.0, 0.0]);
        assert_eq!(wheel.draw(&mut slot_rng(SALT_CENTER, 0)), None);
        assert_eq!(Wheel::new(Vec::new()).draw(&mut slot_rng(SALT_CENTER, 0)), None);
    }
}
