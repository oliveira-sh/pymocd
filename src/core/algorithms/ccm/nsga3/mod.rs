//! CCM's own NSGA-III (Deb & Jain, IEEE TEC 18(4):577–601, 2014) over locus genomes.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod engine;
mod individual;
mod niching;
mod normalize;
mod reference_points;
mod survival;

pub(super) use engine::evolve;
pub(super) use individual::fast_non_dominated_sort;
