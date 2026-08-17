//! Eq. (4) node similarity and the Eq. (8) frequency-weighted walk similarity.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod pairwise;
mod walk_sim;

pub use walk_sim::avg_similarity;
