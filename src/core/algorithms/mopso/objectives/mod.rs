//! The objective the swarm minimises: the Constant Potts Model, decomposed.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod cpm;

pub type Obj = [f64; 2]; // a point in objective space; the CPM split has exactly two terms.

pub(super) use cpm::{Counts, community_count, load_sizes, measure, obj_of};
