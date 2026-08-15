//! GMOCS: GPU-accelerated Multiobjective Co-evolutionary Swarm particle
//! optimization for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at <https://www.gnu.org/licenses/gpl-3.0.html>

use rustc_hash::FxHashMap;

use crate::core::algorithms::gmocs::Labels;
use crate::core::graph::CsrGraph;

const UNSEEN: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ObjSet {
    #[default]
    KkmRc = 0,

    HpIntraInter = 6,
}

impl ObjSet {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            6 => Self::HpIntraInter,
            _ => Self::KkmRc,
        }
    }
}

pub const fn split_mode(v: u16) -> (ObjSet, ObjSet) {
    if v < 100 {
        let s = ObjSet::from_u8(v as u8);
        (s, s)
    } else if v < 1000 {
        let h = v - 100;
        (
            ObjSet::from_u8((h / 10) as u8),
            ObjSet::from_u8((h % 10) as u8),
        )
    } else {
        let h = v - 1000;
        (
            ObjSet::from_u8((h / 100) as u8),
            ObjSet::from_u8((h % 100) as u8),
        )
    }
}

pub fn evaluate(g: &CsrGraph, labels: &Labels, set: ObjSet) -> Vec<f64> {
    match set {
        ObjSet::HpIntraInter => {
            let (intra, inter) = intra_inter(g, labels);
            vec![intra, inter]
        }
        ObjSet::KkmRc => {
            let (kkm, rc) = kkm_rc(g, labels);
            vec![kkm, rc]
        }
    }
}

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
            order.entry(c).or_insert(b as u64);
            size.push(0.0);
            deg_sum.push(0.0);
            b
        } else {
            s
        } as usize;
        size[b] += 1.0;
        deg_sum[b] += g.deg[v] as f64;
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

pub fn intra_inter(g: &CsrGraph, labels: &Labels) -> (f64, f64) {
    let m = g.m as f64;
    if m == 0.0 {
        return (0.0, 0.0);
    }
    let two_m = 2.0 * m;

    let mut slot: Vec<u32> = vec![UNSEEN; g.n];
    let mut d_c: Vec<f64> = Vec::new();
    for (&c, &k) in labels.iter().zip(g.deg.iter()) {
        debug_assert!((c as usize) < g.n, "label {c} outside [0,{})", g.n);
        let s = slot[c as usize];
        let b = if s == UNSEEN {
            let b = d_c.len() as u32;
            slot[c as usize] = b;
            d_c.push(0.0);
            b
        } else {
            s
        } as usize;
        d_c[b] += k as f64;
    }

    let mut l_intra = 0.0f64;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            l_intra += 1.0;
        }
    }

    let inter: f64 = d_c.iter().map(|&d| (d / two_m).powi(2)).sum();
    (1.0 - l_intra / m, inter)
}
