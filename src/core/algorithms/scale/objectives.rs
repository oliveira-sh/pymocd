use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;

use super::Labels;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ObjSet {
    #[default]
    KkmRc = 0,

    HpIntraInter = 6,
}

impl ObjSet {
    /// Every id other than the two live ones maps to the default, exactly as
    /// out-of-range ids always did.
    pub fn from_u8(v: u8) -> Self {
        match v {
            6 => ObjSet::HpIntraInter,
            _ => ObjSet::KkmRc,
        }
    }
}

pub fn split_mode(v: u16) -> (ObjSet, ObjSet) {
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

/// Evaluate the configured objective vector (all entries minimized).
///
/// The two sets share no axis, so neither arm pays for the other's pass.
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
    let mut size: FxHashMap<i32, f64> = FxHashMap::default();
    let mut l_in: FxHashMap<i32, f64> = FxHashMap::default();
    let mut deg_sum: FxHashMap<i32, f64> = FxHashMap::default();

    for v in 0..g.n {
        let c = labels[v];
        *size.entry(c).or_insert(0.0) += 1.0;
        *deg_sum.entry(c).or_insert(0.0) += g.deg[v] as f64;
        let mut internal = 0.0;
        for &u in g.neighbors(v) {
            if labels[u as usize] == c {
                internal += 1.0;
            }
        }
        *l_in.entry(c).or_insert(0.0) += internal;
    }

    let n = g.n as f64;
    let k = size.len() as f64;

    let mut kkm_internal = 0.0;
    let mut rc = 0.0;
    for (c, &sz) in size.iter() {
        if sz == 0.0 {
            continue;
        }
        let li = l_in[c];
        let ds = deg_sum[c];
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

    // Dense-remap communities to 0..k so the volume sum has a fixed order.
    let mut remap: FxHashMap<i32, usize> = FxHashMap::default();
    let mut d_c: Vec<f64> = Vec::new();
    for (&c, &k) in labels.iter().zip(g.deg.iter()) {
        let b = *remap.entry(c).or_insert_with(|| {
            d_c.push(0.0);
            d_c.len() - 1
        });
        d_c[b] += k as f64;
    }

    // `g.edges` holds each undirected edge once, so this is Σ_c l_c directly.
    let mut l_intra = 0.0f64;
    for &(u, v) in &g.edges {
        if labels[u as usize] == labels[v as usize] {
            l_intra += 1.0;
        }
    }

    let inter: f64 = d_c.iter().map(|&d| (d / two_m).powi(2)).sum();
    (1.0 - l_intra / m, inter)
}
