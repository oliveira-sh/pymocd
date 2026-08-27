//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::cdrme::topology::Topology;

pub struct Candidate {
    pub labels: Vec<u32>,
    pub k: usize,
    pub quality: f64,
    pub objective: f64,
}

pub fn modularity(topology: &Topology, labels: &[u32], k: usize) -> f64 {
    if topology.m == 0 {
        return 0.0;
    }
    let m = topology.m as f64;
    let mut internal = vec![0u32; k];
    let mut degree = vec![0u32; k];
    for &u in &topology.active {
        let community = labels[u as usize] as usize;
        degree[community] += topology.degree(u) as u32;
        for &v in topology.neighbors(u) {
            if u < v && labels[v as usize] as usize == community {
                internal[community] += 1;
            }
        }
    }
    (0..k)
        .map(|c| f64::from(internal[c]) / m - (f64::from(degree[c]) / (2.0 * m)).powi(2))
        .sum()
}

pub fn best(candidates: Vec<Candidate>) -> Option<Vec<u32>> {
    candidates
        .into_iter()
        .reduce(|best, next| {
            let better = next
                .quality
                .total_cmp(&best.quality)
                .then_with(|| next.objective.total_cmp(&best.objective))
                .then_with(|| best.k.cmp(&next.k))
                .then_with(|| best.labels.cmp(&next.labels));
            if better.is_gt() { next } else { best }
        })
        .map(|winner| winner.labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> Topology {
        Topology::from_edges(
            &(0..6).collect::<Vec<i32>>(),
            &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)],
        )
    }

    #[test]
    fn modularity_prefers_the_true_split() {
        let topology = two_triangles();
        let split = modularity(&topology, &[0, 0, 0, 1, 1, 1], 2);
        let merged = modularity(&topology, &[0, 0, 0, 0, 0, 0], 1);
        assert!((split - 0.35714285714285715).abs() < 1e-12, "{split}");
        assert!((merged - 0.0).abs() < 1e-12);
        assert!(split > merged);
    }

    #[test]
    fn the_highest_modularity_candidate_wins() {
        let picked = best(vec![
            Candidate {
                labels: vec![0, 0, 0, 0, 0, 0],
                k: 1,
                quality: 0.0,
                objective: 9.0,
            },
            Candidate {
                labels: vec![0, 0, 0, 1, 1, 1],
                k: 2,
                quality: 0.39,
                objective: 1.0,
            },
        ]);
        assert_eq!(picked, Some(vec![0, 0, 0, 1, 1, 1]));
        assert_eq!(best(Vec::new()), None);
    }

    #[test]
    fn ties_fall_through_to_the_objective_then_to_fewer_communities() {
        let picked = best(vec![
            Candidate {
                labels: vec![0, 1],
                k: 2,
                quality: 0.5,
                objective: 1.0,
            },
            Candidate {
                labels: vec![1, 0],
                k: 2,
                quality: 0.5,
                objective: 2.0,
            },
            Candidate {
                labels: vec![0, 0],
                k: 1,
                quality: 0.5,
                objective: 2.0,
            },
        ]);
        assert_eq!(picked, Some(vec![0, 0]));
    }
}
