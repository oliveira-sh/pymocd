//! Sec. 4.2.5's |V|-gene chromosome and the per-gene merge that folds the walks.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::cdrme::topology::Topology;

pub const NO_CENTER: u32 = u32::MAX;
pub const NO_COMMUNITY: u32 = u32::MAX;

/// One gene per node, holding `(center, similarity)`.
///
/// Sec. 4.2.5's sequential pairwise merge is a running element-wise argmax fold
/// here; Fig. 5 IS that argmax over the five Fig. 4 chromosomes, so the operator
/// is associative and only one chromosome is ever held.
pub struct Chromosome {
    pub center: Vec<u32>,
    pub sim: Vec<f64>,
}

impl Chromosome {
    pub fn new(n: usize) -> Self {
        Self {
            center: vec![NO_CENTER; n],
            sim: vec![0.0; n],
        }
    }

    /// "For each gene, we select the center node with high similarity";
    /// equal similarities go to the lower centre id so the fold is deterministic.
    pub fn absorb(&mut self, center: u32, sim: &[f64], active: &[u32]) {
        for &v in active {
            let slot = v as usize;
            let take = self.center[slot] == NO_CENTER
                || sim[slot] > self.sim[slot]
                || (sim[slot] == self.sim[slot] && center < self.center[slot]);
            if take {
                self.center[slot] = center;
                self.sim[slot] = sim[slot];
            }
        }
    }

    /// Sec. 4.3.1 indexes each community by its centre node, so a centre keeps
    /// its own gene even when another chromosome ties it at 1.0.
    pub fn pin(&mut self, center: u32) {
        self.center[center as usize] = center;
        self.sim[center as usize] = 1.0;
    }

    /// Dense community id per node (`NO_COMMUNITY` for isolated ones) and the
    /// number of primary communities.
    pub fn communities(&self, topology: &Topology) -> (Vec<u32>, usize) {
        let mut centers: Vec<u32> = topology
            .active
            .iter()
            .map(|&v| self.center[v as usize])
            .collect();
        centers.sort_unstable();
        centers.dedup();

        let mut labels = vec![NO_COMMUNITY; topology.n];
        for &v in &topology.active {
            let center = self.center[v as usize];
            labels[v as usize] = centers.partition_point(|&c| c < center) as u32;
        }
        (labels, centers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gene_goes_to_the_higher_similarity_and_ties_to_the_lower_centre() {
        let topology = Topology::from_edges(&[0, 1, 2], &[(0, 1), (1, 2)]);
        let mut chromosome = Chromosome::new(3);
        chromosome.absorb(2, &[0.4, 0.9, 1.0], &topology.active);
        chromosome.absorb(0, &[1.0, 0.9, 0.1], &topology.active);
        assert_eq!(chromosome.center, vec![0, 0, 2]);
        assert_eq!(chromosome.sim, vec![1.0, 0.9, 1.0]);
    }

    #[test]
    fn communities_are_dense_ids_in_centre_order() {
        let topology = Topology::from_edges(&[0, 1, 2, 3, 4], &[(0, 1), (1, 2), (2, 3)]);
        let mut chromosome = Chromosome::new(5);
        chromosome.absorb(3, &[0.1, 0.1, 0.9, 1.0, 0.0], &topology.active);
        chromosome.absorb(0, &[1.0, 0.9, 0.1, 0.1, 0.0], &topology.active);
        let (labels, k) = chromosome.communities(&topology);
        assert_eq!(k, 2);
        assert_eq!(labels, vec![0, 0, 1, 1, NO_COMMUNITY]);
    }
}
