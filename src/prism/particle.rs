//! prism/particle.rs
//! PRISM particle: EM-momentum velocity, neighbor-majority relabel,
//! Louvain local search. All community-keyed state uses flat Vec<T> buffers
//! (via `Scratch`) — no FxHashMap on the hot path.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::graph::CommunityId;
use crate::prism::dense::{DenseGraph, Scratch, Solution};
use rand::RngExt;

#[derive(Debug)]
pub struct Particle {
    pub current: Solution,
    pub velocity: Vec<f64>,
    /// Exponentially-averaged momentum of prior velocities.
    pub em_velocity: Vec<f64>,
    pub pbest: Solution,
    pub scratch: Scratch,
}

impl Particle {
    pub fn new(partition: Vec<CommunityId>, initial_v: f64) -> Self {
        let n = partition.len();
        let current = Solution::new(partition);
        let pbest = current.clone();
        Particle {
            current,
            velocity: vec![initial_v; n],
            em_velocity: vec![initial_v; n],
            pbest,
            scratch: Scratch::new(n),
        }
    }
}

/// Velocity + discrete relabel in one pass over nodes. Uses `scratch.freq_f`
/// as a flat counter over communities (size n) and `scratch.touched` as the
/// reset log. No FxHashMap; O(deg) per node both for accumulate and reset.
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
        velocity,
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

    // Stage 1: EM velocity update.
    let one_minus_beta = 1.0 - beta;
    let phi_c1r1 = phi_scale * c1 * r1;
    let phi_c2r2 = phi_scale * c2 * r2;
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(pb.len(), n);
    debug_assert_eq!(ld.len(), n);
    debug_assert!(velocity.len() >= n && em_velocity.len() >= n);
    // SAFETY: x, pb, ld, velocity, em_velocity all have length n — asserted
    // above. No aliasing: all disjoint Particle fields / leader reference.
    unsafe {
        for j in 0..n {
            let xj = *x.get_unchecked(j);
            let d_pb = (*pb.get_unchecked(j) != xj) as u8 as f64;
            let d_ld = (*ld.get_unchecked(j) != xj) as u8 as f64;
            let em = *em_velocity.get_unchecked(j);
            let v_raw = chi * (w * em + phi_c1r1 * d_pb + phi_c2r2 * d_ld);
            let v_new = v_raw.clamp(0.0, v_max);
            *velocity.get_unchecked_mut(j) = v_new;
            *em_velocity.get_unchecked_mut(j) = beta * em + one_minus_beta * v_new;
        }
    }

    // Stage 2: discrete move via neighbor-majority + swarm bias.
    let freq = &mut scratch.freq_f;
    let touched = &mut scratch.touched;
    let bias_pb = c1 * r1;
    let bias_ld = c2 * r2;

    for j in 0..n {
        let v = em_velocity[j].min(v_max);
        if v <= 0.0 || rng.random::<f64>() >= v {
            continue;
        }
        let cur = x[j];
        let nbs = dg.neighbors(j);
        if nbs.is_empty() {
            continue;
        }

        touched.clear();
        // Accumulate neighbor-label counts (flat, first-touch logging).
        for &nb in nbs {
            let c = x[nb as usize] as usize;
            if freq[c] == 0.0 {
                touched.push(c as u32);
            }
            freq[c] += 1.0;
        }

        // Swarm bias: only boost labels already present in neighbors
        // (load-bearing — never inject foreign labels).
        let pb_j = pb[j];
        if pb_j != cur && freq[pb_j as usize] > 0.0 {
            freq[pb_j as usize] += bias_pb;
        }
        let ld_j = ld[j];
        if ld_j != cur && freq[ld_j as usize] > 0.0 {
            freq[ld_j as usize] += bias_ld;
        }

        // Argmax with tie collection over the touched set.
        let mut best_score = f64::MIN;
        for &c in touched.iter() {
            let s = freq[c as usize];
            if s > best_score {
                best_score = s;
            }
        }
        let mut tied_count: u32 = 0;
        let mut last_tied: u32 = 0;
        for &c in touched.iter() {
            if (freq[c as usize] - best_score).abs() < 1e-9 {
                tied_count += 1;
                last_tied = c;
            }
        }
        let best = if tied_count == 1 {
            last_tied as CommunityId
        } else {
            // uniform pick among tied
            let mut pick = rng.random_range(0..tied_count);
            let mut chosen = last_tied;
            for &c in touched.iter() {
                if (freq[c as usize] - best_score).abs() < 1e-9 {
                    if pick == 0 {
                        chosen = c;
                        break;
                    }
                    pick -= 1;
                }
            }
            chosen as CommunityId
        };

        // Reset freq via touched log (O(deg)).
        for &c in touched.iter() {
            freq[c as usize] = 0.0;
        }

        if best != cur {
            x[j] = best;
        }
    }

    current.objectives.clear();
    current.rank = usize::MAX;
    current.crowding_distance = f64::MAX;
}

pub fn maybe_update_pbest(particle: &mut Particle) {
    let curr = &particle.current;
    let pb = &particle.pbest;
    if curr.dominates(pb) {
        particle.pbest = curr.clone();
        return;
    }
    if pb.dominates(curr) {
        return;
    }
    if rand::rng().random_bool(0.5) {
        particle.pbest = curr.clone();
    }
}

/// Flat-freq neighbor-majority mutation. `scratch.freq_u` is the counter;
/// touched slots are reset via the log.
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

    for j in 0..dg.n {
        if rng.random::<f64>() >= rate {
            continue;
        }
        let nbs = dg.neighbors(j);
        if nbs.is_empty() {
            continue;
        }
        touched.clear();
        let current = partition[j];
        let mut best = current;
        let mut max_count: u32 = 0;
        for &nb in nbs {
            let c = partition[nb as usize] as usize;
            if freq[c] == 0 {
                touched.push(c as u32);
            }
            freq[c] += 1;
            if freq[c] > max_count {
                max_count = freq[c];
                best = c as CommunityId;
            }
        }
        for &c in touched.iter() {
            freq[c as usize] = 0;
        }
        if max_count > 0 && best != current {
            partition[j] = best;
        }
    }
}

/// Louvain-style greedy ΔQ refinement. `scratch.ctd` holds `comm_tot_deg`
/// across passes; `scratch.freq_f` is the per-node `edges_to` buffer,
/// reset via touched log.
pub fn louvain_refine(
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

    // comm_tot_deg = Σ deg[i] for i ∈ c, flat over all labels.
    let ctd = &mut scratch.ctd;
    ctd[..n].fill(0.0);
    for i in 0..n {
        ctd[partition[i] as usize] += dg.degrees[i] as f64;
    }

    let edges_to = &mut scratch.freq_f;
    let touched = &mut scratch.touched;

    for _ in 0..iters {
        let mut moved = false;
        for i in 0..n {
            let deg_i = dg.degrees[i] as f64;
            if deg_i == 0.0 {
                continue;
            }
            let cur = partition[i];
            let nbs = dg.neighbors(i);

            // Build edges_to flat counter (scratch.freq_f).
            touched.clear();
            for &nb in nbs {
                let c = partition[nb as usize] as usize;
                if edges_to[c] == 0.0 {
                    touched.push(c as u32);
                }
                edges_to[c] += 1.0;
            }
            let self_edges = edges_to[cur as usize];

            // Louvain ΔQ for moving i → c: gain(c) - gain_back(cur).
            // Scaled by 1/m; signs preserved for comparison.
            let tot_cur_removed = ctd[cur as usize] - deg_i;
            let mut best_gain = 0.0f64;
            let mut best_c = cur;
            for &c_u32 in touched.iter() {
                let c = c_u32 as CommunityId;
                if c == cur {
                    continue;
                }
                let k_ic = edges_to[c as usize];
                let tot_c = ctd[c as usize];
                let gain = (k_ic - self_edges) - deg_i * (tot_c - tot_cur_removed) * inv_m2;
                if gain > best_gain {
                    best_gain = gain;
                    best_c = c;
                }
            }

            // Reset edges_to via touched log.
            for &c_u32 in touched.iter() {
                edges_to[c_u32 as usize] = 0.0;
            }

            if best_c != cur {
                ctd[cur as usize] -= deg_i;
                ctd[best_c as usize] += deg_i;
                partition[i] = best_c;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}
