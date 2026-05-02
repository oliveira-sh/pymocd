//! Per-particle preallocated work buffers, reused across generations.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

#[derive(Debug)]
pub struct Scratch {
    pub intra: Vec<f64>,
    pub wdeg: Vec<f64>,
    pub freq_f: Vec<f64>,
    pub freq_u: Vec<u32>,
    pub ctd: Vec<f64>,
    pub touched: Vec<u32>,
    /// Touched-community list for evaluate_q_gamma. Index in this list
    /// is the per-call dense community id used to address `intra`/`wdeg`.
    pub touched_q: Vec<u32>,
    /// Per-community version stamp. version_q is bumped per call; a slot
    /// whose stamp differs from version_q is stale, overwritten on first
    /// touch. Removes the per-call O(n) fill of intra/wdeg.
    pub last_version_q: Vec<u64>,
    pub version_q: u64,
    /// Original-community-id to dense-id map. Valid only while the
    /// matching last_version_q slot equals version_q.
    pub idx_of_c: Vec<u32>,
    /// Node-id to dense-community-id, recomputed per evaluate_q_gamma call.
    pub p_dense: Vec<u32>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            intra: vec![0.0; n],
            wdeg: vec![0.0; n],
            freq_f: vec![0.0; n],
            freq_u: vec![0; n],
            ctd: vec![0.0; n],
            touched: Vec::with_capacity(n),
            touched_q: Vec::with_capacity(n),
            last_version_q: vec![0; n],
            version_q: 0,
            idx_of_c: vec![0; n],
            p_dense: vec![0; n],
        }
    }
}
