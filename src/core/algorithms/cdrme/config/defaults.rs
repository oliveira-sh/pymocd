//! Shipped values for the CDRME parameters, none of which the paper pins down.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2026 - Guilherme Santos. If a copy of the MPL was not distributed with this
//! file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html

// Eq. (7)'s coefficient; the paper gives only "the range of 1 to 2".
pub const DEFAULT_ALPHA_WALK: f64 = 1.0;

// Algorithm 1's n_walk.
pub const DEFAULT_N_WALK: usize = 50;

// N_p (Sec. 4.3.1). Cost is linear in it; see README for the measurements.
pub const DEFAULT_POP_SIZE: usize = 300;

// N_sp <= N_p (Sec. 4.4.1). Eq. (12) grows with the number of communities, so
// ranking by it would drop exactly the coarse chromosomes Sec. 4.4.4 has to
// choose between; the shipped value is the upper end of the paper's own bound.
pub const DEFAULT_ELITE_SIZE: usize = DEFAULT_POP_SIZE;

// The mutation threshold alpha of Sec. 4.4.2, on the [0,1] similarity scale.
pub const DEFAULT_ALPHA_MUT: f64 = 0.5;

// Ceiling on the 4.4.2 <-> 4.4.3 loop, which also stops at its own fixed point.
pub const DEFAULT_MUT_SWEEPS: usize = 10;

// Total merge attempts per starting community (Sec. 4.3.2 states no bound). A
// merge that does not gain is always taken, so two per community lets the
// deepest slots run the ladder down to a single community.
pub const MERGE_ATTEMPTS_PER_COMMUNITY: usize = 2;
