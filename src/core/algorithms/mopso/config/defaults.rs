//! Default parameters for MOPSO; the shipped values are the measured winners.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

pub const DEFAULT_POP_SIZE: usize = 100; // swarm size, matched to HP-MOCD and SMOCC.

pub const DEFAULT_NUM_GENS: usize = 100;

pub const DEFAULT_INERTIA: f64 = 0.4; // fraction of a node's instability that survives to the next iteration.

pub const DEFAULT_COGNITIVE: f64 = 0.7; // pull toward the particle's own best partition.

pub const DEFAULT_SOCIAL: f64 = 0.7; // pull toward the archive leader.

pub const DEFAULT_LOCAL_RATE: f64 = 0.35; // per-node rate of the resolution-directed CPM local move.

pub const DEFAULT_SEED_ROUNDS: usize = 8; // alternating node-move and merge rounds driving a particle to its rung.
