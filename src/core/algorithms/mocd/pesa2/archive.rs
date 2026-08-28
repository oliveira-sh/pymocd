//! External-archive (EP) maintenance: insertion and truncation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rand::RngExt;

use super::grid::{GRID_DIVISIONS, assign_cells};
use super::solution::Member;

pub fn insert_nondominated(ep: &mut Vec<Member>, candidate: Member) {
    if ep.iter().any(|m| m.solution.dominates(&candidate.solution)) {
        return;
    }
    ep.retain(|m| !candidate.solution.dominates(&m.solution));
    ep.push(candidate);
}

/// Squeeze-factor truncation to `epsize`, one member per pass: the grid is
/// rebuilt after every removal, so crowding is re-read rather than reused.
pub fn truncate(ep: &mut Vec<Member>, epsize: usize, rng: &mut impl rand::Rng) {
    let mut occ: Vec<usize> = Vec::new();
    while ep.len() > epsize {
        assign_cells(ep, GRID_DIVISIONS, &mut occ);
        let max_occ = *occ.iter().max().unwrap();
        let crowded_cells: Vec<usize> = occ
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == max_occ)
            .map(|(cell, _)| cell)
            .collect();
        let chosen_cell = crowded_cells[rng.random_range(0..crowded_cells.len())];

        let candidates: Vec<usize> = ep
            .iter()
            .enumerate()
            .filter(|(_, m)| m.cell == chosen_cell)
            .map(|(i, _)| i)
            .collect();
        let pick = candidates[rng.random_range(0..candidates.len())];
        ep.remove(pick);
    }
}
