//! Flat-array data structures for the PRISM hot path. See `../README.md`.
//!
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::graph::CommunityId;

pub type DensePartition = Vec<CommunityId>;

/// Edge endpoints packed into a u64. Low 32 bits hold u, high 32 bits hold v.
pub type EdgeUV = u64;

mod graph;
mod lpa;
mod objectives;
mod scratch;
mod solution;

pub use graph::DenseGraph;
pub use lpa::lpa_partition;
pub use objectives::evaluate_q_gamma;
pub use scratch::Scratch;
pub use solution::{Solution, crowding_distance, q_score};
