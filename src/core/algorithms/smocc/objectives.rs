use crate::core::graph::CsrGraph;
use rustc_hash::FxHashMap;

use super::Labels;

/// Slot-array sentinel: this community label has not been seen yet.
const UNSEEN: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ObjSet {
    #[default]
    KkmRc = 0,

    HpIntraInter = 6,
}

impl ObjSet {
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
    // Accumulate into dense slot-indexed vectors: labels are node ids in
    // [0, n), so `slot` replaces three hash lookups per node.
    //
    // `order` exists only to fix the sequence of the final reduction: it sums
    // li/sz and (ds-li)/sz, both inexact, so the order is observable in the
    // front. Its keys are inserted on exactly the first-touch subsequence of
    // the label stream (repeat `entry()` hits never mutate the table), and its
    // value is u64 so that (K, V) has the same size and alignment as the
    // `FxHashMap<i32, f64>` it replaced -- hashbrown sizes its buckets from the
    // table layout, so identical layout plus identical key sequence gives
    // identical iteration order. `order_map_iterates_like_the_f64_map_it_replaced`
    // guards that; widening or shrinking this value type can move the fronts.
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

    // L(V_i, V_i) over the unique-edge list instead of over every node's
    // neighbour list. `g.edges` holds each undirected edge exactly once as
    // `(u, v)` with `u < v` and never a self-loop (`CsrGraph::from_edges`
    // skips `du == dv`), while `adj` carries both directions of exactly those
    // edges. The node scan added 1.0 per same-community half-edge, so its
    // total for a community is 2 * (its intra-edge count) -- which is what
    // `+= 2` per intra edge accumulates here, over the same multiset of edges.
    //
    // Reassociating is bit-exact. Every summand is a small non-negative
    // integer, and a community's total is bounded by 2m = adj.len(), which is
    // addressed by the u32 `xadj` and so is below 2^32. Every partial sum is
    // therefore an integer below 2^53, where both integer addition and the
    // final u64 -> f64 conversion are exact, and an exact integer sum does not
    // depend on order. What is *not* reassociated is the reduction below:
    // li/sz and (ds-li)/sz are inexact, and the node pass above still assigns
    // every slot in node order, so `order`'s insertion sequence -- which pins
    // that reduction -- is untouched.
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
    for (_, &b) in order.iter() {
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

    // Community labels are dense node ids in [0, n), so a slot array replaces
    // the hash remap. First-touch order is unchanged, so `d_c` is
    // element-for-element identical and the reduction below sums in the same
    // sequence.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `kkm_rc` reduces in `order`'s iteration order, and the fronts are only
    /// byte-reproducible against pre-existing runs while that order matches the
    /// `FxHashMap<i32, f64>` this map replaced. hashbrown derives its bucket
    /// count from the table layout, so equal `(K, V)` size and alignment plus an
    /// equal key sequence is what makes the two agree. Guard both halves.
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
