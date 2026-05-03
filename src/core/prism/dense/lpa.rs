//! Label Propagation seed for the swarm.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::{DenseGraph, DensePartition};
use crate::core::graph::CommunityId;
use rand::RngExt;

pub fn lpa_partition(dg: &DenseGraph, iters: usize) -> DensePartition {
    let mut rng = rand::rng();
    let n = dg.n;
    let mut p: DensePartition = (0..n).map(|i| i as CommunityId).collect();
    let mut order: Vec<usize> = (0..n).collect();

    let mut freq: Vec<u32> = vec![0; n];
    let mut touched: Vec<u32> = Vec::with_capacity(64);
    let mut tied: Vec<CommunityId> = Vec::with_capacity(8);

    let min_k = ((n as f64).sqrt().floor() as usize).max(4);
    for _ in 0..iters {
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        {
            let mut seen = rustc_hash::FxHashSet::default();
            for &c in p.iter() {
                seen.insert(c);
                if seen.len() > min_k {
                    break;
                }
            }
            if seen.len() <= min_k {
                break;
            }
        }
        let mut changed = false;
        for &j in &order {
            let nbs = dg.neighbors(j);
            if nbs.is_empty() {
                continue;
            }
            touched.clear();
            let mut best = 0u32;
            for &nb in nbs {
                let c = p[nb as usize] as usize;
                if freq[c] == 0 {
                    touched.push(c as u32);
                }
                freq[c] += 1;
                if freq[c] > best {
                    best = freq[c];
                }
            }
            tied.clear();
            for &c in &touched {
                if freq[c as usize] == best {
                    tied.push(c as CommunityId);
                }
            }
            let new_c = if tied.len() == 1 {
                tied[0]
            } else {
                tied[rng.random_range(0..tied.len())]
            };
            for &c in &touched {
                freq[c as usize] = 0;
            }
            if new_c != p[j] {
                p[j] = new_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    p
}
