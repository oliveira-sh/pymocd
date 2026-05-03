//! EM-momentum velocity update + discrete relabel.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use super::Particle;
use crate::core::graph::CommunityId;
use crate::core::prism::dense::{DenseGraph, Solution};
use rand::RngExt;

pub fn update_particle(
    particle: &mut Particle,
    leader: &Solution,
    dg: &DenseGraph,
    inertia_low: f64,
    inertia_high: f64,
    v_max: f64,
    phi_scale: f64,
    beta: f64,
    chi: f64,
) {
    let n = dg.n;
    if n == 0 {
        return;
    }
    let Particle {
        current,
        em_velocity,
        pbest,
        scratch,
    } = particle;
    let x = &mut current.partition;
    let pb = &pbest.partition;
    let ld = &leader.partition;

    let mut rng = rand::rng();
    let w = rng.random_range(inertia_low..=inertia_high);
    let c1 = rng.random_range(1.5..=2.5);
    let c2 = rng.random_range(1.5..=2.5);
    let r1: f64 = rng.random();
    let r2: f64 = rng.random();

    let one_minus_beta = 1.0 - beta;
    let phi_c1r1 = phi_scale * c1 * r1;
    let phi_c2r2 = phi_scale * c2 * r2;
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(pb.len(), n);
    debug_assert_eq!(ld.len(), n);
    debug_assert!(em_velocity.len() >= n);
    // SAFETY: x, pb, ld, em_velocity all have length n (asserted above);
    // disjoint particle fields and leader reference, no aliasing.
    unsafe {
        for j in 0..n {
            let xj = *x.get_unchecked(j);
            let d_pb = (*pb.get_unchecked(j) != xj) as u8 as f64;
            let d_ld = (*ld.get_unchecked(j) != xj) as u8 as f64;
            let em = *em_velocity.get_unchecked(j);
            let v_raw = chi * (w * em + phi_c1r1 * d_pb + phi_c2r2 * d_ld);
            let v_new = v_raw.clamp(0.0, v_max);
            *em_velocity.get_unchecked_mut(j) = beta * em + one_minus_beta * v_new;
        }
    }

    let freq = &mut scratch.freq_f;
    let touched = &mut scratch.touched;
    let bias_pb = c1 * r1;
    let bias_ld = c2 * r2;
    let mut any_flip = false;

    // SAFETY: x, pb, ld, em_velocity all length n; nb < n by DenseGraph
    // construction; freq sized n.
    unsafe {
        for j in 0..n {
            let v = (*em_velocity.get_unchecked(j)).min(v_max);
            if v <= 0.0 || rng.random::<f64>() >= v {
                continue;
            }
            let cur = *x.get_unchecked(j);
            let nbs = dg.neighbors(j);
            if nbs.is_empty() {
                continue;
            }

            touched.clear();
            for &nb in nbs {
                let c = *x.get_unchecked(nb as usize) as usize;
                let f = freq.get_unchecked_mut(c);
                if *f == 0.0 {
                    touched.push(c as u32);
                }
                *f += 1.0;
            }

            // Bias only labels already present in the neighborhood, never
            // inject foreign labels (load-bearing for topology awareness).
            let pb_j = *pb.get_unchecked(j);
            if pb_j != cur {
                let f = freq.get_unchecked_mut(pb_j as usize);
                if *f > 0.0 {
                    *f += bias_pb;
                }
            }
            let ld_j = *ld.get_unchecked(j);
            if ld_j != cur {
                let f = freq.get_unchecked_mut(ld_j as usize);
                if *f > 0.0 {
                    *f += bias_ld;
                }
            }

            let mut best_score = f64::NEG_INFINITY;
            let mut tied_count: u32 = 0;
            let mut last_tied: u32 = 0;
            for &c in touched.iter() {
                let s = *freq.get_unchecked(c as usize);
                if s > best_score + 1e-9 {
                    best_score = s;
                    tied_count = 1;
                    last_tied = c;
                } else if s > best_score - 1e-9 {
                    tied_count += 1;
                    last_tied = c;
                }
            }
            let best = if tied_count <= 1 {
                last_tied as CommunityId
            } else {
                let mut pick = rng.random_range(0..tied_count);
                let mut chosen = last_tied;
                for &c in touched.iter() {
                    if (*freq.get_unchecked(c as usize) - best_score).abs() < 1e-9 {
                        if pick == 0 {
                            chosen = c;
                            break;
                        }
                        pick -= 1;
                    }
                }
                chosen as CommunityId
            };

            for &c in touched.iter() {
                *freq.get_unchecked_mut(c as usize) = 0.0;
            }

            if best != cur {
                *x.get_unchecked_mut(j) = best;
                any_flip = true;
            }
        }
    }

    // Invalidate cached objectives only on actual flip; evaluate_swarm
    // skips re-evaluation when objectives stay valid.
    if any_flip {
        current.objectives.clear();
        current.rank = usize::MAX;
        current.crowding_distance = f64::MAX;
    }
}
