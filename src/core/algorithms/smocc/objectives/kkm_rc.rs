//! Kernel k-means and ratio-cut: the community-size-normalised objective pair.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

use crate::core::algorithms::smocc::Labels;
use crate::core::graph::CsrGraph;

use super::UNSEEN;

pub fn kkm_rc(g: &CsrGraph, labels: &Labels) -> (f64, f64) {
    let mut slot: Vec<u32> = vec![UNSEEN; g.n];
    let mut order: FxHashMap<i32, u64> = FxHashMap::default();
    let mut size: Vec<f64> = Vec::new();
    let mut deg_sum: Vec<f64> = Vec::new();

    for (v, &c) in labels.iter().enumerate().take(g.n) {
        debug_assert!((c as usize) < g.n, "label {c} outside [0,{})", g.n);
        let s = slot[c as usize];
        let b = if s == UNSEEN {
            let b = size.len() as u32;
            slot[c as usize] = b;
            order.entry(c).or_insert_with(|| u64::from(b));
            size.push(0.0);
            deg_sum.push(0.0);
            b
        } else {
            s
        } as usize;
        size[b] += 1.0;
        deg_sum[b] += f64::from(g.deg[v]);
    }

    let mut l_in: Vec<u64> = vec![0; size.len()];
    for &(u, v) in &g.edges {
        let cu = labels[u as usize];
        if cu == labels[v as usize] {
            debug_assert!(slot[cu as usize] != UNSEEN, "edge label {cu} has no slot");
            l_in[slot[cu as usize] as usize] += 2;
        }
    }

    let n = g.n as f64;
    let k = order.len() as f64;

    let mut kkm_internal = 0.0;
    let mut rc = 0.0;
    for &b in order.values() {
        let b = b as usize;
        let sz = size[b];
        if sz == 0.0 {
            continue;
        }
        let li = l_in[b] as f64;
        let ds = deg_sum[b];
        kkm_internal += li / sz;
        rc += (ds - li) / sz;
    }

    (2.0 * (n - k) - kkm_internal, rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_map_iterates_like_the_f64_map_it_replaced() {
        assert_eq!(size_of::<(i32, u64)>(), size_of::<(i32, f64)>());
        assert_eq!(align_of::<(i32, u64)>(), align_of::<(i32, f64)>());
        let mut r = 0x9E37_79B9_7F4A_7C15u64;
        for k in [1usize, 2, 3, 7, 8, 13, 16, 31, 64, 257] {
            let mut slots: FxHashMap<i32, u64> = FxHashMap::default();
            let mut sums: FxHashMap<i32, f64> = FxHashMap::default();
            for _ in 0..k * 4 {
                r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key = ((r >> 33) % k as u64) as i32;
                let next = slots.len() as u64;
                slots.entry(key).or_insert(next);
                *sums.entry(key).or_insert(0.0) += 1.0;
            }
            let a: Vec<i32> = slots.keys().copied().collect();
            let b: Vec<i32> = sums.keys().copied().collect();
            assert_eq!(a, b, "iteration order diverged for {k} distinct labels");
        }
    }
}
