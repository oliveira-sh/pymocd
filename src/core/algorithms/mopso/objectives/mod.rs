//! The objective the swarm minimises: the Constant Potts Model, decomposed.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod cpm;

/// A point in objective space. Fixed at two entries: the CPM split has exactly
/// two terms, and a fixed-size array keeps every particle's objective off the
/// heap.
pub type Obj = [f64; 2];

pub(super) use cpm::{Counts, community_count, load_sizes, measure, obj_of};
