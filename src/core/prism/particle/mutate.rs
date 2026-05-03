//! Neighbor-majority dense mutation (turbulence step).
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CommunityId;
use crate::core::prism::dense::{DenseGraph, Scratch};
use rand::RngExt;

pub fn dense_mutate(
    partition: &mut [CommunityId],
    dg: &DenseGraph,
    rate: f64,
    scratch: &mut Scratch,
) {
    if rate <= 0.0 || dg.n == 0 {
        return;
    }
    let mut rng = rand::rng();
    let freq = &mut scratch.freq_u;
    let touched = &mut scratch.touched;
    let n = dg.n;

    // SAFETY: partition.len() == n; nb < n by DenseGraph construction.
    unsafe {
        for j in 0..n {
            if rng.random::<f64>() >= rate {
                continue;
            }
            let nbs = dg.neighbors(j);
            if nbs.is_empty() {
                continue;
            }
            touched.clear();
            let current = *partition.get_unchecked(j);
            let mut best = current;
            let mut max_count: u32 = 0;
            for &nb in nbs {
                let c = *partition.get_unchecked(nb as usize) as usize;
                let f = freq.get_unchecked_mut(c);
                if *f == 0 {
                    touched.push(c as u32);
                }
                *f += 1;
                if *f > max_count {
                    max_count = *f;
                    best = c as CommunityId;
                }
            }
            for &c in touched.iter() {
                *freq.get_unchecked_mut(c as usize) = 0;
            }
            if max_count > 0 && best != current {
                *partition.get_unchecked_mut(j) = best;
            }
        }
    }
}
