//! A PESA-II archive member, its dominance test, and its evaluation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use crate::core::algorithms::mocd::locus::{self, Genome, NodeIndex};
use crate::core::algorithms::mocd::objectives::calculate_objectives;
use crate::core::graph::Graph;

#[derive(Clone, Debug)]
pub struct Solution {
    /// Compacted community ids, indexed by node position.
    pub labels: Vec<i32>,
    /// `[intra, inter]` (Shi Eqs. 3.5/3.6), both minimised.
    pub objectives: Vec<f64>,
}

impl Solution {
    pub fn dominates(&self, other: &Self) -> bool {
        let mut better = false;
        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a > b {
                return false;
            }
            if a < b {
                better = true;
            }
        }
        better
    }
}

pub struct Member {
    pub genome: Genome,
    pub solution: Solution,
    pub cell: usize,
}

pub fn evaluate(graph: &Graph, idx: &NodeIndex, degrees: &[usize], genome: Genome) -> Member {
    let labels = locus::decode(&genome);
    let metrics = calculate_objectives(graph, idx, &labels, degrees);
    Member {
        genome,
        solution: Solution {
            labels,
            objectives: vec![metrics.intra, metrics.inter],
        },
        cell: 0,
    }
}
