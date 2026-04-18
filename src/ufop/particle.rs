//! ufop/particle.rs
//! Extended SMPSO particle: EM-momentum velocity, adaptive χ, VNR mutation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::ufop::dense::{DenseGraph, Solution};
use crate::graph::CommunityId;
use rand::RngExt;
use rustc_hash::{FxBuildHasher, FxHashMap};

#[derive(Clone, Debug)]
pub struct Particle {
    pub current: Solution,
    pub velocity: Vec<f64>,
    /// Exponentially-averaged momentum of prior velocities.
    pub em_velocity: Vec<f64>,
    pub pbest: Solution,
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
        }
    }
}

/// Extended velocity update with EM momentum + adaptive χ.
///  v_new = χ · [ w·em[j] + c1·r1·𝟙[pb≠x] + c2·r2·𝟙[ld≠x] ]
///  em[j] = β·em[j] + (1-β)·v_new
///  move prob = em[j] (clamped to [0, v_max])
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
    let x = &mut particle.current.partition;
    let pb = &particle.pbest.partition;
    let ld = &leader.partition;

    let mut rng = rand::rng();
    let w = rng.random_range(inertia_low..=inertia_high);
    let c1 = rng.random_range(1.5..=2.5);
    let c2 = rng.random_range(1.5..=2.5);
    let r1: f64 = rng.random();
    let r2: f64 = rng.random();

    // Stage 1: EM velocity update.
    for j in 0..n {
        let d_pb = (pb[j] != x[j]) as u8 as f64;
        let d_ld = (ld[j] != x[j]) as u8 as f64;
        let v_raw =
            chi * (w * particle.em_velocity[j] + phi_scale * (c1 * r1 * d_pb + c2 * r2 * d_ld));
        let v_new = v_raw.clamp(0.0, v_max);
        particle.velocity[j] = v_new;
        particle.em_velocity[j] = beta * particle.em_velocity[j] + (1.0 - beta) * v_new;
    }

    // Stage 2: discrete move via neighbor-majority + swarm bias.
    let mut freq: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);
    let mut tied: Vec<CommunityId> = Vec::with_capacity(8);

    for j in 0..n {
        let v = particle.em_velocity[j].min(v_max);
        if v <= 0.0 || rng.random::<f64>() >= v {
            continue;
        }
        let current = x[j];
        let pb_j = pb[j];
        let ld_j = ld[j];

        let neighbors = &dg.adj[j];
        if neighbors.is_empty() {
            continue;
        }

        freq.clear();
        for &nb in neighbors {
            let comm = x[nb as usize];
            *freq.entry(comm).or_insert(0.0) += 1.0;
        }
        if freq.is_empty() {
            continue;
        }

        if pb_j != current {
            if let Some(f) = freq.get_mut(&pb_j) {
                *f += c1 * r1;
            }
        }
        if ld_j != current {
            if let Some(f) = freq.get_mut(&ld_j) {
                *f += c2 * r2;
            }
        }

        let mut best_score = f64::MIN;
        for &score in freq.values() {
            if score > best_score {
                best_score = score;
            }
        }
        tied.clear();
        for (&comm, &score) in freq.iter() {
            if (score - best_score).abs() < 1e-9 {
                tied.push(comm);
            }
        }
        let best = if tied.len() == 1 {
            tied[0]
        } else {
            tied[rng.random_range(0..tied.len())]
        };

        if best != current {
            x[j] = best;
        }
    }

    particle.current.objectives.clear();
    particle.current.rank = usize::MAX;
    particle.current.crowding_distance = f64::MAX;
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

/// Dense neighbor-majority mutation.
pub fn dense_mutate(partition: &mut [CommunityId], dg: &DenseGraph, rate: f64) {
    if rate <= 0.0 || dg.n == 0 {
        return;
    }
    let mut rng = rand::rng();
    let mut freq: FxHashMap<CommunityId, u32> =
        FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

    for j in 0..dg.n {
        if rng.random::<f64>() >= rate {
            continue;
        }
        let neighbors = &dg.adj[j];
        if neighbors.is_empty() {
            continue;
        }
        freq.clear();
        let mut max_count = 0u32;
        let current = partition[j];
        let mut best = current;
        for &nb in neighbors {
            let comm = partition[nb as usize];
            let entry = freq.entry(comm).or_insert(0);
            *entry += 1;
            if *entry > max_count {
                max_count = *entry;
                best = comm;
            }
        }
        if max_count > 0 && best != current {
            partition[j] = best;
        }
    }
}

/// Vulnerable Node Reassignment.
/// For each node with k_ext >= k_in, reassign to neighbor community with
/// maximum "reassignment rate" = edges_to_c / |V_c|. Prevents node leakage
/// in high-mixing regimes.
#[allow(dead_code)]
pub fn vnr_mutate(partition: &mut [CommunityId], dg: &DenseGraph) {
    if dg.n == 0 {
        return;
    }
    // Precompute community sizes.
    let mut size: FxHashMap<CommunityId, u32> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    for &c in partition.iter() {
        *size.entry(c).or_insert(0) += 1;
    }
    // Cap to prevent giant-community cascade at high μ.
    // Target community must be smaller than this to accept absorption.
    let size_cap = (dg.n as f64 * 0.25).ceil() as u32;

    let mut freq: FxHashMap<CommunityId, u32> =
        FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

    for j in 0..dg.n {
        let neighbors = &dg.adj[j];
        if neighbors.is_empty() {
            continue;
        }
        let cur = partition[j];
        let mut k_in = 0u32;
        freq.clear();
        for &nb in neighbors {
            let c = partition[nb as usize];
            *freq.entry(c).or_insert(0) += 1;
            if c == cur {
                k_in += 1;
            }
        }
        let k_ext = (neighbors.len() as u32) - k_in;
        // Strict "vulnerable": k_ext > k_in (not ≥) to avoid tie-cascades.
        if k_ext <= k_in {
            continue;
        }
        // Pick neighbor community with max rate = count / size, subject to size_cap.
        let mut best_rate = f64::MIN;
        let mut best_c = cur;
        for (&c, &cnt) in freq.iter() {
            if c == cur {
                continue;
            }
            let sz_u = *size.get(&c).unwrap_or(&1);
            if sz_u >= size_cap {
                continue;
            }
            let sz = sz_u.max(1) as f64;
            let rate = (cnt as f64) / sz;
            if rate > best_rate {
                best_rate = rate;
                best_c = c;
            }
        }
        if best_c != cur {
            if let Some(n_old) = size.get_mut(&cur) {
                if *n_old > 0 {
                    *n_old -= 1;
                }
            }
            *size.entry(best_c).or_insert(0) += 1;
            partition[j] = best_c;
        }
    }
}

/// Louvain-style local modularity refinement.
/// For each node, evaluate ΔQ for moving to every neighbor community; apply best positive move.
/// Repeats until no node changes or `iters` reached. Guaranteed non-decreasing in Q.
pub fn louvain_refine(partition: &mut [CommunityId], dg: &DenseGraph, iters: usize) {
    let n = dg.n;
    if n == 0 || dg.total_edges == 0 {
        return;
    }
    let m2 = 2.0 * dg.total_edges as f64;

    let mut comm_tot_deg: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(64, FxBuildHasher);
    for i in 0..n {
        *comm_tot_deg.entry(partition[i]).or_insert(0.0) += dg.degrees[i] as f64;
    }

    let mut edges_to: FxHashMap<CommunityId, f64> =
        FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

    for _ in 0..iters {
        let mut moved = false;
        for i in 0..n {
            let deg_i = dg.degrees[i] as f64;
            if deg_i == 0.0 {
                continue;
            }
            let cur = partition[i];

            // Edges from i to each community (neighbors only).
            edges_to.clear();
            let mut self_edges = 0.0;
            for &nb in &dg.adj[i] {
                let c = partition[nb as usize];
                *edges_to.entry(c).or_insert(0.0) += 1.0;
                if c == cur {
                    self_edges += 1.0;
                }
            }

            // "Remove" node i from cur: tot_cur - deg_i
            let tot_cur = *comm_tot_deg.get(&cur).unwrap_or(&deg_i);
            let tot_cur_removed = tot_cur - deg_i;

            // ΔQ for moving to target c: gain(c) - gain_back(cur)
            // gain(c)    = edges_to[c]/m - deg_i * tot[c]    / m2^2 * m2   (simplified)
            // Louvain standard: ΔQ_move_to_c  = k_{i,c}/m  - deg_i * tot[c] / (2 m^2)
            //                   ΔQ_leave_cur = k_{i,cur}/m - deg_i * (tot_cur - deg_i) / (2 m^2)
            //                   Net = k_{i,c} - k_{i,cur}  - deg_i*(tot[c] - tot_cur_removed) / (2m)
            // All scaled by 1/m but sign preserved.
            let mut best_gain = 0.0;
            let mut best_c = cur;
            let k_cur = self_edges;
            for (&c, &k_ic) in edges_to.iter() {
                if c == cur {
                    continue;
                }
                let tot_c = *comm_tot_deg.get(&c).unwrap_or(&0.0);
                let gain = (k_ic - k_cur) - deg_i * (tot_c - tot_cur_removed) / m2;
                if gain > best_gain {
                    best_gain = gain;
                    best_c = c;
                }
            }

            if best_c != cur {
                if let Some(v) = comm_tot_deg.get_mut(&cur) {
                    *v -= deg_i;
                }
                *comm_tot_deg.entry(best_c).or_insert(0.0) += deg_i;
                partition[i] = best_c;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}
