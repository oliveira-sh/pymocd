//! Default parameters for MOPSO; the shipped values are the measured winners.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

/// Swarm size. Matched to HP-MOCD and SMOCC so the comparison is like for like.
pub const DEFAULT_POP_SIZE: usize = 100;

pub const DEFAULT_NUM_GENS: usize = 100;

/// Inertia. The fraction of a node's instability that survives to the next
/// iteration, so a node the attractors have stopped pulling still moves for a
/// few iterations and then settles.
pub const DEFAULT_INERTIA: f64 = 0.4;

/// Cognitive weight: the pull toward the particle's own best partition.
pub const DEFAULT_COGNITIVE: f64 = 0.7;

/// Social weight: the pull toward the archive leader, drawn by binary
/// tournament on crowding distance so it comes from wherever the front is
/// still sparse rather than from the particle's own neighbourhood of it.
pub const DEFAULT_SOCIAL: f64 = 0.7;

/// Per-node rate of the resolution-directed CPM local move. This is the
/// operator that turns a per-node blend of two parents back into a partition;
/// every detector in this repository that dropped its local search collapsed at
/// high mixing.
pub const DEFAULT_LOCAL_RATE: f64 = 0.35;

/// Alternating node-move and community-merge rounds used to drive each particle
/// to its rung of the ladder. A merge sweep accepts a matching, so the
/// community count at worst halves per round; eight rounds is enough to reach
/// the coarse end from a scattered start, and the loop stops early once a
/// particle settles.
pub const DEFAULT_SEED_ROUNDS: usize = 8;

/// Selector: `1` is the resolution plateau, `0` the equal-weight sum, which is
/// CPM at `gamma` = density and the Leiden-CPM baseline's operating point.
pub const DEFAULT_SELECT_MODE: u8 = 1;
