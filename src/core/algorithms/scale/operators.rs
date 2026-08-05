use crate::core::graph::CsrGraph;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::{Genome, Labels};

pub const TOPO_MAJORITY_MUT: u8 = 1 << 1;
pub const TOPO_HPMOCD_CROSS: u8 = 1 << 7;

pub const MICRO_BITS: u8 = TOPO_MAJORITY_MUT | TOPO_HPMOCD_CROSS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MicroOps {
    pub majority_mut: bool,
    pub hpmocd_cross: bool,
}

impl MicroOps {
    pub fn from_topo(topo_mode: u8) -> Self {
        let t = topo_mode & MICRO_BITS;
        MicroOps {
            majority_mut: t & TOPO_MAJORITY_MUT != 0,
            hpmocd_cross: t & TOPO_HPMOCD_CROSS != 0,
        }
    }

    pub fn any(self) -> bool {
        self != MicroOps::default()
    }
}

const RNG_BASE: u64 = 0x5CA1_E5EED;

pub(super) fn slot_rng(salt: u64, slot: usize) -> StdRng {
    StdRng::seed_from_u64(
        RNG_BASE ^ salt.rotate_left(32) ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

#[inline]
fn tournament(ranks: &[usize], crowd: &[f64], r: &mut impl Rng) -> usize {
    let len = ranks.len();
    let i = r.random_range(0..len);
    let j = r.random_range(0..len);
    if ranks[i] < ranks[j] || (ranks[i] == ranks[j] && crowd[i] >= crowd[j]) {
        i
    } else {
        j
    }
}

pub fn macro_offspring(
    parents: &[Genome],
    ranks: &[usize],
    crowd: &[f64],
    p_m: f64,
    salt: u64,
) -> Vec<Genome> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = parents[0].len();
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let b = tournament(ranks, crowd, &mut r);
            let (pa, pb) = (&parents[a], &parents[b]);

            let mut child: Genome = Vec::with_capacity(n);
            for i in 0..n {
                let mut bit = if r.random_bool(0.5) { pa[i] } else { pb[i] };
                if r.random_bool(p_m) {
                    bit ^= 1;
                }
                child.push(bit);
            }

            if child.iter().all(|&b| b == 0) && n > 0 {
                let k = r.random_range(0..n);
                child[k] = 1;
            }
            child
        })
        .collect()
}

pub(super) fn micro_mut_rate(n: usize, rate: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if rate > 0.0 {
        rate.min(1.0)
    } else {
        1.0 / n as f64
    }
}

pub fn micro_offspring(
    g: &CsrGraph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
    micro_mut: f64,
    salt: u64,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = micro_mut_rate(n, micro_mut);
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let mut child: Labels = parents[a].clone();

            if r.random_bool(p_c) && n > 0 {
                let b = tournament(ranks, crowd, &mut r);
                let pb = &parents[b];
                let j = r.random_range(0..n);
                let donor = pb[j];
                for u in 0..n {
                    if pb[u] == donor {
                        child[u] = donor;
                    }
                }
            }

            for i in 0..n {
                let nbrs = g.neighbors(i);
                if !nbrs.is_empty() && r.random_bool(p_mut) {
                    let t = nbrs[r.random_range(0..nbrs.len())] as usize;
                    child[i] = child[t];
                }
            }
            child
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn micro_offspring_topo(
    g: &CsrGraph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
    micro_mut: f64,
    salt: u64,
    ops: MicroOps,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = micro_mut_rate(n, micro_mut);
    let topo_mut = ops.majority_mut;
    (0..pop)
        .into_par_iter()
        .map(|k| {
            let mut r = slot_rng(salt, k);
            let a = tournament(ranks, crowd, &mut r);
            let mut child: Labels;
            if r.random_bool(p_c) && n > 0 {
                if ops.hpmocd_cross {
                    const ENSEMBLE: usize = 4;
                    let mut idx: Vec<usize> = Vec::with_capacity(ENSEMBLE);
                    idx.push(a);
                    let mut tries = 0;
                    while idx.len() < ENSEMBLE.min(parents.len()) && tries < 64 {
                        let cand = tournament(ranks, crowd, &mut r);
                        if !idx.contains(&cand) {
                            idx.push(cand);
                        }
                        tries += 1;
                    }
                    child = Vec::with_capacity(n);
                    let mut counts: FxHashMap<i32, u32> = FxHashMap::default();
                    let mut tied: Vec<i32> = Vec::with_capacity(ENSEMBLE);
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..n {
                        counts.clear();
                        let mut max_c = 0u32;
                        for &pi in &idx {
                            let l = parents[pi][i];
                            let e = counts.entry(l).or_insert(0);
                            *e += 1;
                            if *e > max_c {
                                max_c = *e;
                            }
                        }
                        tied.clear();
                        for (&l, &c) in counts.iter() {
                            if c == max_c {
                                tied.push(l);
                            }
                        }
                        // FxHashMap iteration order is unstable across runs;
                        // sorting keeps the fronts byte-reproducible.
                        tied.sort_unstable();
                        let lab = if tied.len() == 1 {
                            tied[0]
                        } else {
                            tied[r.random_range(0..tied.len())]
                        };
                        child.push(lab);
                    }
                } else {
                    child = parents[a].clone();
                    let b = tournament(ranks, crowd, &mut r);
                    let pb = &parents[b];
                    let j = r.random_range(0..n);
                    let donor = pb[j];
                    for u in 0..n {
                        if pb[u] == donor {
                            child[u] = donor;
                        }
                    }
                }
            } else {
                child = parents[a].clone();
            }
            let mut freq: FxHashMap<i32, u32> = FxHashMap::default();
            for i in 0..n {
                let nbrs = g.neighbors(i);
                if !nbrs.is_empty() && r.random_bool(p_mut) {
                    if topo_mut {
                        freq.clear();
                        let mut best = child[i];
                        let mut bestc = 0u32;
                        for &v in nbrs {
                            let l = child[v as usize];
                            let e = freq.entry(l).or_insert(0);
                            *e += 1;
                            if *e > bestc {
                                bestc = *e;
                                best = l;
                            }
                        }
                        child[i] = best;
                    } else {
                        let t = nbrs[r.random_range(0..nbrs.len())] as usize;
                        child[i] = child[t];
                    }
                }
            }
            child
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::CsrGraph;

    const QUIET_MICRO_MUT: f64 = 0.0;

    fn ring_of_cliques(k: i32, s: i32) -> CsrGraph {
        let nodes: Vec<i32> = (0..k * s).collect();
        let mut e = Vec::new();
        for c in 0..k {
            let lo = c * s;
            for a in lo..lo + s {
                for b in (a + 1)..lo + s {
                    e.push((a, b));
                }
            }
            e.push((lo + s - 1, (lo + s) % (k * s)));
        }
        CsrGraph::from_edges(&nodes, &e)
    }

    #[test]
    fn hpmocd_crossover_is_distinct_and_deterministic() {
        let g = ring_of_cliques(8, 5);
        let n = g.n;
        let parents: Vec<Labels> = (0..8)
            .map(|k| {
                let mut r = slot_rng(77, k);
                (0..n).map(|_| r.random_range(0..6i32)).collect()
            })
            .collect();
        let ranks = vec![1usize; parents.len()];
        let crowd = vec![1.0f64; parents.len()];
        let run = |ops: MicroOps| {
            micro_offspring_topo(&g, &parents, &ranks, &crowd, 1.0, QUIET_MICRO_MUT, 5, ops)
        };
        let hp = run(MicroOps::from_topo(TOPO_HPMOCD_CROSS));
        let graft = run(MicroOps::default());
        assert_eq!(
            hp,
            run(MicroOps::from_topo(TOPO_HPMOCD_CROSS)),
            "bit 7 is not deterministic"
        );
        assert_ne!(
            hp, graft,
            "bit 7 produced the baseline graft's offspring — is it wired up?"
        );
        assert!(hp.iter().all(|c| c.len() == n));
    }

    #[test]
    fn majority_mutation_is_distinct_and_deterministic() {
        let g = ring_of_cliques(8, 5);
        let n = g.n;
        let parents: Vec<Labels> = (0..8)
            .map(|k| {
                let mut r = slot_rng(21, k);
                (0..n).map(|_| r.random_range(0..6i32)).collect()
            })
            .collect();
        let ranks = vec![1usize; parents.len()];
        let crowd = vec![1.0f64; parents.len()];
        let run =
            |ops: MicroOps| micro_offspring_topo(&g, &parents, &ranks, &crowd, 0.0, 0.5, 13, ops);
        let maj = run(MicroOps::from_topo(TOPO_MAJORITY_MUT));
        let base = run(MicroOps::default());
        assert_eq!(
            maj,
            run(MicroOps::from_topo(TOPO_MAJORITY_MUT)),
            "bit 1 is not deterministic"
        );
        assert_ne!(
            maj, base,
            "bit 1 produced the baseline mutation's offspring"
        );
        assert!(maj.iter().all(|c| c.len() == n));
    }

    #[test]
    fn micro_ops_decodes_only_the_two_live_bits() {
        assert_eq!(TOPO_MAJORITY_MUT, 1 << 1);
        assert_eq!(TOPO_HPMOCD_CROSS, 1 << 7);
        assert_eq!(MICRO_BITS, 0b1000_0010);

        assert_eq!(MicroOps::from_topo(0), MicroOps::default());
        assert!(!MicroOps::from_topo(0).any());
        assert!(MicroOps::from_topo(TOPO_MAJORITY_MUT).majority_mut);
        assert!(MicroOps::from_topo(TOPO_HPMOCD_CROSS).hpmocd_cross);

        let shipped = MicroOps::from_topo(130);
        assert!(shipped.majority_mut && shipped.hpmocd_cross);

        for dead in [1u8, 4, 8, 16, 32, 64] {
            assert!(
                !MicroOps::from_topo(dead).any(),
                "deleted bit {dead} still routes micro"
            );
            assert_eq!(
                MicroOps::from_topo(130 | dead),
                shipped,
                "deleted bit {dead} changed the shipped mask"
            );
        }
        assert_eq!(MicroOps::from_topo(0xFF), shipped);
    }
}
