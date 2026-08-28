//! Micro-side offspring: community grafting plus neighbour mutation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::{Rng, RngExt};

use crate::core::algorithms::mmcomo::{Graph, Labels};

use super::mating::tournament;

/// One-way crossover: every node the donor parent puts in node `j`'s community
/// takes that community in the child.
fn graft(child: &mut Labels, donor_parent: &Labels, j: usize, n: usize) {
    let donor = donor_parent[j];
    for u in 0..n {
        if donor_parent[u] == donor {
            child[u] = donor;
        }
    }
}

/// Each node adopts a random neighbour's current label with probability `p_mut`.
fn neighbour_mutation(g: &Graph, child: &mut Labels, p_mut: f64, r: &mut impl Rng) {
    for i in 0..g.n {
        let nbrs = &g.adj[i];
        if !nbrs.is_empty() && r.random_bool(p_mut) {
            let t = nbrs[r.random_range(0..nbrs.len())];
            child[i] = child[t];
        }
    }
}

/// Micro offspring (Alg. 1 line 7): one-way crossover (prob `p_c`) + neighbour
/// mutation (rate 1/n).
pub fn micro_offspring(
    g: &Graph,
    parents: &[Labels],
    ranks: &[usize],
    crowd: &[f64],
    p_c: f64,
) -> Vec<Labels> {
    let pop = parents.len();
    if pop == 0 {
        return Vec::new();
    }
    let n = g.n;
    let p_mut = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    let mut r = rand::rng();
    let mut children: Vec<Labels> = Vec::with_capacity(pop);

    for _ in 0..pop {
        let a = tournament(ranks, crowd, &mut r);
        let mut child: Labels = parents[a].clone();

        if r.random_bool(p_c) && n > 0 {
            let b = tournament(ranks, crowd, &mut r);
            let j = r.random_range(0..n);
            graft(&mut child, &parents[b], j, n);
        }

        neighbour_mutation(g, &mut child, p_mut, &mut r);

        children.push(child);
    }
    children
}
