//! Pareto machinery: dominance, crowding, and the bounded external archive
//! that together give the swarm its shared memory of the front.
//!
//! There is deliberately no non-dominated sort here. An archive-based MOPSO
//! never ranks a mixed population — the archive is non-dominated by
//! construction, each particle's personal best is ranked by the scalar CPM of
//! its own resolution, and the final front is the archive itself — so
//! NSGA-II's sorting pass would have no caller.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

mod archive;
mod crowding;
mod dominance;

pub(super) use archive::Archive;
/// Only the archive uses dominance in anger; this re-export is what lets
/// the engine's tests assert the archive really is a front.
#[cfg(test)]
pub(super) use dominance::dominates;
