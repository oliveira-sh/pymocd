//! Sec. 4.3.2: one stochastic agglomerative merge chain per population slot.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;
use rand::rngs::StdRng;

use crate::core::algorithms::cdrme::chromosome::Relation;
use crate::core::algorithms::cdrme::objective::{objective, objective_if_merged};
use crate::core::algorithms::cdrme::sampling::{SALT_MERGE, slot_rng};
use crate::core::algorithms::cdrme::topology::Topology;

/// The best community set one chain ever held, with its Eq. (12) value.
pub struct Chain {
    pub labels: Vec<u32>,
    pub k: usize,
    pub objective: f64,
}

// Sec. 4.3.2 replaces C by C' "with a probability of 1 - 1/(ObjFunc(C') -
// ObjFunc(C))", which is a probability only when the difference is at least 1.
// This is that formula clamped into [0,1]; the perverse middle branch (a gain
// under 1 is refused) is the paper's, not a typo. README, "The Sec. 4.3.2
// acceptance rule", derives all three branches.
fn accept(delta: f64, rng: &mut StdRng) -> bool {
    if delta.is_nan() || delta <= 0.0 {
        return true;
    }
    if delta < 1.0 {
        return false;
    }
    rng.random_bool(1.0 - 1.0 / delta)
}

fn compress(base: &[u32], parent: &mut [u32], k: usize) -> (Vec<u32>, usize) {
    fn root(parent: &mut [u32], mut c: u32) -> u32 {
        while parent[c as usize] != c {
            parent[c as usize] = parent[parent[c as usize] as usize];
            c = parent[c as usize];
        }
        c
    }

    let mut dense = vec![u32::MAX; k];
    let mut next = 0;
    for c in 0..k as u32 {
        let r = root(parent, c);
        if dense[r as usize] == u32::MAX {
            dense[r as usize] = next;
            next += 1;
        }
    }
    let labels = base
        .iter()
        .map(|&c| {
            if c == u32::MAX {
                u32::MAX
            } else {
                dense[root(parent, c) as usize]
            }
        })
        .collect();
    (labels, next as usize)
}

/// Runs `attempts` merge attempts on one population slot and returns the set it
/// ends on.
///
/// Sec. 4.3.2 draws one random chromosome per iteration and merges it once, so
/// the chromosomes sit at different depths of their own merge chains while the
/// process runs; that spread is the population's only diversity, and it is what
/// Sec. 4.4.4 later picks from. A merge on one slot never reads another, so the
/// interleaving is immaterial and each slot runs its own attempts at once.
pub fn diversify(
    topology: &Topology,
    base: &[u32],
    k: usize,
    slot: usize,
    attempts: usize,
) -> Chain {
    let mut relation = Relation::from_labels(topology, base, k);
    let mut rng = slot_rng(SALT_MERGE, slot);
    let mut current = objective(&relation);
    let mut history: Vec<(u32, u32)> = Vec::new();

    for _ in 0..attempts {
        if relation.len() <= 1 {
            break;
        }
        let i = relation.live[rng.random_range(0..relation.len())];
        let neighbors = relation.neighbors(i);
        if neighbors.is_empty() {
            continue;
        }
        let j = neighbors[rng.random_range(0..neighbors.len())];
        let candidate = objective_if_merged(&relation, i, j);
        if accept(candidate - current, &mut rng) {
            relation.merge(i, j);
            history.push((i, j));
            current = candidate;
        }
    }

    let mut parent: Vec<u32> = (0..k as u32).collect();
    for &(i, j) in &history {
        parent[j as usize] = i;
    }
    let (labels, live) = compress(base, &mut parent, k);
    Chain {
        labels,
        k: live,
        objective: current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::cdrme::chromosome::NO_COMMUNITY;

    // four triangles in a path, bridged 2-3, 5-6, 8-9
    fn path_of_triangles() -> (Topology, Vec<u32>) {
        let nodes: Vec<i32> = (0..12).collect();
        let mut edges = Vec::new();
        for base in [0, 3, 6, 9] {
            edges.extend([(base, base + 1), (base + 1, base + 2), (base, base + 2)]);
        }
        edges.extend([(2, 3), (5, 6), (8, 9)]);
        let labels: Vec<u32> = (0..12u32).map(|v| v / 3).collect();
        (Topology::from_edges(&nodes, &edges), labels)
    }

    #[test]
    fn the_clamped_rule_keeps_both_branches_the_stop_condition_needs() {
        let mut rng = slot_rng(SALT_MERGE, 0);
        assert!(accept(-3.0, &mut rng));
        assert!(accept(0.0, &mut rng));
        assert!(accept(f64::NAN, &mut rng));
        assert!(!accept(0.5, &mut rng));
        assert!(!accept(0.999, &mut rng));
        let taken = (0..400).filter(|_| accept(2.0, &mut rng)).count();
        assert!((160..=240).contains(&taken), "{taken}");
    }

    #[test]
    fn a_chain_reports_the_objective_of_the_set_it_returns() {
        let (topology, base) = path_of_triangles();
        for slot in 0..16 {
            let chain = diversify(&topology, &base, 4, slot, slot % 5);
            let relation = Relation::from_labels(&topology, &chain.labels, chain.k);
            assert!(
                (objective(&relation) - chain.objective).abs() < 1e-12,
                "slot {slot}: reported {} but the labels score {}",
                chain.objective,
                objective(&relation)
            );
            assert!(chain.objective >= 1.0);
        }
    }

    #[test]
    fn different_slots_reach_different_sets() {
        let (topology, base) = path_of_triangles();
        let sizes: Vec<usize> = (0..32).map(|s| diversify(&topology, &base, 4, s, s / 8).k).collect();
        assert!(sizes.iter().any(|&k| k != sizes[0]), "the chains are frozen");
        assert!(sizes.iter().all(|&k| (1..=4).contains(&k)));
    }

    #[test]
    fn a_chain_is_reproducible_and_keeps_isolated_nodes_out() {
        let (topology, mut base) = path_of_triangles();
        base.push(NO_COMMUNITY);
        let topology = Topology::from_edges(
            &(0..13).collect::<Vec<i32>>(),
            &topology
                .active
                .iter()
                .flat_map(|&u| {
                    topology
                        .neighbors(u)
                        .iter()
                        .filter(move |&&v| u < v)
                        .map(move |&v| (u as i32, v as i32))
                })
                .collect::<Vec<(i32, i32)>>(),
        );
        let a = diversify(&topology, &base, 4, 3, 6);
        let b = diversify(&topology, &base, 4, 3, 6);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.labels[12], NO_COMMUNITY);
    }
}
