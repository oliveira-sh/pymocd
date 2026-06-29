//! graph representations for community detection.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

use rustc_hash::FxHashMap;

pub type NodeId = i32;
pub type CommunityId = i32;
pub type Partition = FxHashMap<NodeId, CommunityId>;

mod adj;
mod csr;

pub use adj::{Graph, GraphMemoryStats, get_edges, get_nodes};
pub use csr::CsrGraph;
