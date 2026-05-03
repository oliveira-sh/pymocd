//! Greedy delta-Q local optimization (Louvain-style).
//!
//! `scratch.ctd` holds the per-community total degree; `scratch.freq_f`
//! is the per-node `edges_to` buffer, reset via the touched log.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::CommunityId;
use crate::core::prism::dense::{DenseGraph, Scratch};

pub fn local_optimization(
    partition: &mut [CommunityId],
    dg: &DenseGraph,
    iters: usize,
    scratch: &mut Scratch,
) {
    let n = dg.n;
    if n == 0 || dg.total_edges == 0 {
        return;
    }
    let m2 = 2.0 * dg.total_edges as f64;
    let inv_m2 = 1.0 / m2;

    let ctd = &mut scratch.ctd;
    ctd[..n].fill(0.0);
    let degrees = dg.degrees.as_slice();
    // SAFETY: partition.len() == n; partition entries are in [0, n);
    // degrees indexed 0..n.
    unsafe {
        for i in 0..n {
            *ctd.get_unchecked_mut(*partition.get_unchecked(i) as usize) +=
                *degrees.get_unchecked(i) as f64;
        }
    }

    let edges_to = &mut scratch.freq_f;
    let touched = &mut scratch.touched;

    unsafe {
        for _ in 0..iters {
            let mut moved = false;
            for i in 0..n {
                let deg_i = *degrees.get_unchecked(i) as f64;
                if deg_i == 0.0 {
                    continue;
                }
                let cur = *partition.get_unchecked(i);
                let nbs = dg.neighbors(i);

                touched.clear();
                for &nb in nbs {
                    let c = *partition.get_unchecked(nb as usize) as usize;
                    let e = edges_to.get_unchecked_mut(c);
                    if *e == 0.0 {
                        touched.push(c as u32);
                    }
                    *e += 1.0;
                }
                let self_edges = *edges_to.get_unchecked(cur as usize);

                let tot_cur_removed = *ctd.get_unchecked(cur as usize) - deg_i;
                let mut best_gain = 0.0f64;
                let mut best_c = cur;
                for &c_u32 in touched.iter() {
                    let c = c_u32 as CommunityId;
                    if c == cur {
                        continue;
                    }
                    let k_ic = *edges_to.get_unchecked(c as usize);
                    let tot_c = *ctd.get_unchecked(c as usize);
                    let gain = (k_ic - self_edges) - deg_i * (tot_c - tot_cur_removed) * inv_m2;
                    if gain > best_gain {
                        best_gain = gain;
                        best_c = c;
                    }
                }

                for &c_u32 in touched.iter() {
                    *edges_to.get_unchecked_mut(c_u32 as usize) = 0.0;
                }

                if best_c != cur {
                    *ctd.get_unchecked_mut(cur as usize) -= deg_i;
                    *ctd.get_unchecked_mut(best_c as usize) += deg_i;
                    *partition.get_unchecked_mut(i) = best_c;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }
}
