//! Multi-resolution weighted modularity kernel.
//!
//! Returns [-Q_gamma05, -Q_gamma10, -Q_gamma20] for a given partition.
//! See `../README.md` for the hot-loop design.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::{DenseGraph, Scratch};
use crate::graph::CommunityId;

pub fn evaluate_q_gamma(dg: &DenseGraph, p: &[CommunityId], s: &mut Scratch) -> [f64; 3] {
    let n = dg.n;
    if n == 0 || dg.total_weight <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    debug_assert_eq!(p.len(), n);
    debug_assert!(s.intra.len() >= n);
    debug_assert!(s.wdeg.len() >= n);
    debug_assert!(s.last_version_q.len() >= n);
    debug_assert!(s.idx_of_c.len() >= n);
    debug_assert!(s.p_dense.len() >= n);

    let m_w = dg.total_weight;
    let inv_m = 1.0 / m_w;
    let m2 = 2.0 * m_w;
    let inv_m2_sq = 1.0 / (m2 * m2);

    s.version_q = s.version_q.wrapping_add(1);
    let v_tag = s.version_q;
    let last_version = &mut s.last_version_q;
    let touched_q = &mut s.touched_q;
    let idx_of_c = &mut s.idx_of_c;
    let p_dense = &mut s.p_dense;
    let wdeg = &mut s.wdeg;
    let intra = &mut s.intra;
    let wd_src = &dg.weighted_deg;

    // SAFETY: partition IDs are non-negative throughout the pipeline
    // (initial partitions in [0, n); all relabel paths copy existing
    // labels). Reinterpreting i32 to u32 yields the same bits.
    let p_u = unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u32, p.len()) };

    touched_q.clear();

    // Pass 1: build dense partition, accumulate wdeg, init intra slots
    // on first touch.
    unsafe {
        for i in 0..n {
            let c = *p_u.get_unchecked(i) as usize;
            let lv = *last_version.get_unchecked(c);
            let wsrc = *wd_src.get_unchecked(i);
            let dense_id: u32;
            if lv != v_tag {
                *last_version.get_unchecked_mut(c) = v_tag;
                dense_id = touched_q.len() as u32;
                *idx_of_c.get_unchecked_mut(c) = dense_id;
                touched_q.push(c as u32);
                *wdeg.get_unchecked_mut(dense_id as usize) = wsrc;
                *intra.get_unchecked_mut(dense_id as usize) = 0.0;
            } else {
                dense_id = *idx_of_c.get_unchecked(c);
                *wdeg.get_unchecked_mut(dense_id as usize) += wsrc;
            }
            *p_dense.get_unchecked_mut(i) = dense_id;
        }

        // Pass 2: intra-edge sum, indexed by dense id so intra stays in L1.
        // 4-edge unroll keeps the OoO core saturated on the partition-load,
        // compare, conditional-RMW chain.
        let edge_uv = dg.edge_uv.as_slice();
        let edge_w = dg.edge_w.as_slice();
        let m = edge_uv.len();
        let mut k = 0usize;
        while k + 4 <= m {
            let e0 = *edge_uv.get_unchecked(k);
            let e1 = *edge_uv.get_unchecked(k + 1);
            let e2 = *edge_uv.get_unchecked(k + 2);
            let e3 = *edge_uv.get_unchecked(k + 3);

            let u0 = (e0 & 0xFFFF_FFFF) as usize;
            let v0 = (e0 >> 32) as usize;
            let u1 = (e1 & 0xFFFF_FFFF) as usize;
            let v1 = (e1 >> 32) as usize;
            let u2 = (e2 & 0xFFFF_FFFF) as usize;
            let v2 = (e2 >> 32) as usize;
            let u3 = (e3 & 0xFFFF_FFFF) as usize;
            let v3 = (e3 >> 32) as usize;

            let cu0 = *p_dense.get_unchecked(u0);
            let cv0 = *p_dense.get_unchecked(v0);
            let cu1 = *p_dense.get_unchecked(u1);
            let cv1 = *p_dense.get_unchecked(v1);
            let cu2 = *p_dense.get_unchecked(u2);
            let cv2 = *p_dense.get_unchecked(v2);
            let cu3 = *p_dense.get_unchecked(u3);
            let cv3 = *p_dense.get_unchecked(v3);

            if cu0 == cv0 {
                *intra.get_unchecked_mut(cu0 as usize) += *edge_w.get_unchecked(k);
            }
            if cu1 == cv1 {
                *intra.get_unchecked_mut(cu1 as usize) += *edge_w.get_unchecked(k + 1);
            }
            if cu2 == cv2 {
                *intra.get_unchecked_mut(cu2 as usize) += *edge_w.get_unchecked(k + 2);
            }
            if cu3 == cv3 {
                *intra.get_unchecked_mut(cu3 as usize) += *edge_w.get_unchecked(k + 3);
            }
            k += 4;
        }
        while k < m {
            let e = *edge_uv.get_unchecked(k);
            let u = (e & 0xFFFF_FFFF) as usize;
            let v = (e >> 32) as usize;
            let cu = *p_dense.get_unchecked(u);
            if cu == *p_dense.get_unchecked(v) {
                *intra.get_unchecked_mut(cu as usize) += *edge_w.get_unchecked(k);
            }
            k += 1;
        }
    }

    // Pass 3: reduction over the touched dense ids only.
    let k_count = touched_q.len();
    let mut q05 = 0.0;
    let mut q10 = 0.0;
    let mut q20 = 0.0;
    for k in 0..k_count {
        let vol = unsafe { *wdeg.get_unchecked(k) };
        if vol <= 0.0 {
            continue;
        }
        let ec = unsafe { *intra.get_unchecked(k) };
        let l = ec * inv_m;
        let v2 = vol * vol * inv_m2_sq;
        q05 += l - 0.5 * v2;
        q10 += l - v2;
        q20 += l - 2.0 * v2;
    }
    [-q05, -q10, -q20]
}
