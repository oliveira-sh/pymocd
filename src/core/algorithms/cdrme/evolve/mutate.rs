//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::cdrme::topology::Topology;

pub fn mutate(
    topology: &Topology,
    labels: &mut [u32],
    sim: &mut [f64],
    k: usize,
    alpha: f64,
    sweeps: usize,
) {
    let mut sums = vec![0.0f64; k];
    let mut seen = vec![false; k];
    let mut touched: Vec<u32> = Vec::new();

    for _ in 0..sweeps {
        let mut changed = false;
        for &v in &topology.active {
            if sim[v as usize] >= alpha {
                continue;
            }
            touched.clear();
            let mut total = 0.0;
            for &u in topology.neighbors(v) {
                let community = labels[u as usize] as usize;
                if !seen[community] {
                    seen[community] = true;
                    touched.push(community as u32);
                }
                sums[community] += sim[u as usize];
                total += sim[u as usize];
            }
            if total > 0.0 {
                let mut best = touched[0];
                for &c in &touched[1..] {
                    let sum = sums[c as usize];
                    let leader = sums[best as usize];
                    if sum > leader || (sum == leader && c < best) {
                        best = c;
                    }
                }
                let raised = sums[best as usize] / total;
                if best != labels[v as usize] && raised > sim[v as usize] {
                    labels[v as usize] = best;
                    sim[v as usize] = raised;
                    changed = true;
                }
            }
            for &c in &touched {
                sums[c as usize] = 0.0;
                seen[c as usize] = false;
            }
        }
        if !changed {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pendant() -> (Topology, Vec<u32>, Vec<f64>) {
        let topology = Topology::from_edges(&[0, 1, 2, 3], &[(0, 1), (1, 2), (0, 2), (2, 3)]);
        (topology, vec![0, 0, 0, 1], vec![0.9, 0.8, 0.7, 0.1])
    }

    #[test]
    fn a_weak_gene_joins_the_community_its_neighbours_favour() {
        let (topology, mut labels, mut sim) = pendant();
        mutate(&topology, &mut labels, &mut sim, 2, 0.5, 10);
        assert_eq!(labels, vec![0, 0, 0, 0]);
        assert!((sim[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_gene_above_the_threshold_is_never_a_candidate() {
        let (topology, mut labels, mut sim) = pendant();
        sim[3] = 0.6;
        mutate(&topology, &mut labels, &mut sim, 2, 0.5, 10);
        assert_eq!(labels, vec![0, 0, 0, 1]);
    }

    #[test]
    fn a_move_that_would_lower_the_similarity_is_refused() {
        let topology = Topology::from_edges(&[0, 1, 2, 3], &[(0, 1), (0, 2), (0, 3)]);
        let mut labels = vec![0, 1, 1, 2];
        let mut sim = vec![0.9, 0.5, 0.5, 0.5];
        mutate(&topology, &mut labels, &mut sim, 3, 0.95, 10);
        assert_eq!(labels[0], 0);
        assert!((sim[0] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn zero_similarity_neighbourhoods_are_left_alone() {
        let (topology, mut labels, _) = pendant();
        let mut sim = vec![0.0; 4];
        mutate(&topology, &mut labels, &mut sim, 2, 0.5, 10);
        assert_eq!(labels, vec![0, 0, 0, 1]);
    }
}
